from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GPU_QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(GPU_QUERY_FIELDS):
            raise RuntimeError(f"Unexpected nvidia-smi row: {line}")
        row = dict(zip(GPU_QUERY_FIELDS, values))
        for field in GPU_QUERY_FIELDS[3:]:
            try:
                row[field] = float(row[field])
            except (TypeError, ValueError):
                row[field] = math.nan
        row["index"] = int(row["index"])
        rows.append(row)
    return rows


def ensure_requested_gpus_are_free(indices: list[int]) -> list[dict[str, Any]]:
    rows = query_gpus()
    by_index = {int(row["index"]): row for row in rows}
    missing = sorted(set(indices) - set(by_index))
    if missing:
        raise RuntimeError(f"Requested GPUs not visible: {missing}")
    busy = {
        index: {
            "memory_used_mib": by_index[index]["memory.used"],
            "utilization_pct": by_index[index]["utilization.gpu"],
        }
        for index in indices
        if by_index[index]["memory.used"] > 512 or by_index[index]["utilization.gpu"] > 10
    }
    if busy:
        raise RuntimeError(f"Refusing to compete with existing GPU workloads: {busy}")
    return [by_index[index] for index in indices]


def run_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional

    import train_plumrac_loco as v2
    import train_plumrac_v4_phy as v4

    output_path = args.worker_output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "failed",
        "physical_gpu_index": args.physical_gpu_index,
        "model_dir": str(args.model_dir.resolve()),
        "requested_batch_size": args.batch_size,
        "requested_duration_seconds": args.duration_seconds,
        "started_at_utc": utc_now(),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside stress-test worker")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Each worker must see exactly one CUDA device through CUDA_VISIBLE_DEVICES; "
                f"observed {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        project = Path(__file__).resolve().parents[1]
        multimodal_dir = project / "data" / "processed" / "multimodal"
        model_dir = args.model_dir.resolve()
        bundle = json.loads((model_dir / "deployment_bundle.json").read_text(encoding="utf-8"))
        target_name = str(bundle["target"])
        trait = str(bundle["trait"])

        raw = np.load(multimodal_dir / "nir_c_absorbance.npy", mmap_mode="r")
        wavelength = np.load(multimodal_dir / "wavelength_nm.npy").astype(np.float32)
        row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
        row_lookup = pd.Series(
            np.arange(len(row_index), dtype=np.int64),
            index=row_index["sample_id"].astype(str),
        )
        fit = pd.read_parquet(model_dir / "training_fit_predictions.parquet")
        fit["sample_id"] = fit["sample_id"].astype(str)
        fit = fit.loc[fit["sample_id"].isin(row_lookup.index)].copy()
        fit["raw_row"] = row_lookup.loc[fit["sample_id"]].to_numpy(dtype=np.int64)
        fit = fit.loc[
            np.isfinite(pd.to_numeric(fit["y_true"], errors="coerce"))
            & np.isfinite(pd.to_numeric(fit["y_domain_pls_anchor"], errors="coerce"))
        ].reset_index(drop=True)
        if len(fit) < 32:
            raise RuntimeError(f"Too few real training examples for {trait}: {len(fit)}")

        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(fit), size=args.batch_size, replace=True)
        selected_fit = fit.iloc[selected]
        raw_batch_np = np.asarray(
            raw[selected_fit["raw_row"].to_numpy(dtype=np.int64)], dtype=np.float32
        ).copy()
        target_sd = float(bundle["target_sd"])
        target_mean = float(bundle["target_mean"])
        anchor_np = (
            selected_fit["y_domain_pls_anchor"].to_numpy(dtype=np.float32) - target_mean
        ) / target_sd
        residual_np = (
            selected_fit["y_true"].to_numpy(dtype=np.float32)
            - selected_fit["y_domain_pls_anchor"].to_numpy(dtype=np.float32)
        ) / target_sd

        config = v2.RACConfig(**bundle["config"])
        v4.CHANNEL_SET = "basic"
        v4.ARCHITECTURE = "multiscale"
        v4.MIXSTYLE_P = 0.0
        placeholder_mean = np.zeros((1, 3, len(wavelength)), dtype=np.float32)
        placeholder_sd = np.ones_like(placeholder_mean)
        channel_builder = v2.SpectralChannelBuilder(
            wavelength, placeholder_mean, placeholder_sd, config
        )
        channel_state = torch.load(
            model_dir / "channel_builder_state.pt", map_location="cpu", weights_only=True
        )
        channel_builder.load_state_dict(channel_state, strict=True)
        model = v4.V4PlumRACNet(
            config.width, config.blocks, config.dropout, config.attention_tail
        )
        model_state = torch.load(
            model_dir / "plumrac_state.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(model_state, strict=True)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        raw_batch = torch.from_numpy(raw_batch_np).to(device, non_blocking=True)
        anchor = torch.from_numpy(anchor_np).to(device, non_blocking=True)
        residual_target = torch.from_numpy(residual_np).to(device, non_blocking=True)
        channel_builder = channel_builder.to(device).train()
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(channel_builder.parameters()),
            lr=2e-4,
            weight_decay=2e-3,
        )

        def train_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            channels = channel_builder(raw_batch, augment=True)
            prediction = model(channels, anchor)
            loss = functional.mse_loss(prediction, residual_target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.detach().item()}")
            loss.backward()
            optimizer.step()
            return float(loss.detach().item())

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        warmup_losses = [train_step() for _ in range(args.warmup_steps)]
        torch.cuda.synchronize(device)

        iterations = 0
        last_loss = warmup_losses[-1]
        timed_started = time.perf_counter()
        while True:
            last_loss = train_step()
            iterations += 1
            if iterations % args.sync_interval == 0:
                torch.cuda.synchronize(device)
                if time.perf_counter() - timed_started >= args.duration_seconds:
                    break
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - timed_started
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
        device_properties = torch.cuda.get_device_properties(device)

        result.update(
            {
                "status": "passed",
                "finished_at_utc": utc_now(),
                "trait": trait,
                "target": target_name,
                "device_name": torch.cuda.get_device_name(device),
                "visible_cuda_devices": torch.cuda.device_count(),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "parameter_count": int(parameter_count),
                "spectral_points": int(raw_batch.shape[1]),
                "real_source_samples": int(len(fit)),
                "batch_size": int(args.batch_size),
                "warmup_steps": int(args.warmup_steps),
                "iterations": int(iterations),
                "elapsed_seconds": float(elapsed),
                "samples_processed": int(iterations * args.batch_size),
                "samples_per_second": float(iterations * args.batch_size / elapsed),
                "last_loss": float(last_loss),
                "finite_loss": bool(math.isfinite(last_loss)),
                "peak_memory_allocated_mib": float(peak_allocated),
                "peak_memory_reserved_mib": float(peak_reserved),
                "device_total_memory_mib": float(device_properties.total_memory / (1024**2)),
            }
        )
    except BaseException as error:
        result.update(
            {
                "finished_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def run_controller(args: argparse.Namespace) -> None:
    project = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    requested_gpus = [0, 1]
    before = ensure_requested_gpus_are_free(requested_gpus)
    (output_dir / "gpu_state_before.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )
    full_smi_before = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    ).stdout
    (output_dir / "nvidia_smi_before.txt").write_text(full_smi_before, encoding="utf-8")

    model_root = project / "results" / "v19" / "selected_production_models"
    assignments = {0: model_root / "SRF", 1: model_root / "LS"}
    processes: dict[int, subprocess.Popen[bytes]] = {}
    handles: dict[int, tuple[Any, Any]] = {}
    worker_outputs: dict[int, Path] = {}
    launched_at = time.monotonic()
    for physical_gpu, model_dir in assignments.items():
        worker_output = output_dir / f"worker_gpu{physical_gpu}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--physical-gpu-index",
            str(physical_gpu),
            "--model-dir",
            str(model_dir),
            "--worker-output",
            str(worker_output),
            "--duration-seconds",
            str(args.duration_seconds),
            "--batch-size",
            str(args.batch_size),
            "--warmup-steps",
            str(args.warmup_steps),
            "--sync-interval",
            str(args.sync_interval),
            "--seed",
            str(args.seed + physical_gpu),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", "2")
        environment.setdefault("OPENBLAS_NUM_THREADS", "2")
        stdout_handle = (output_dir / f"worker_gpu{physical_gpu}.stdout.log").open("wb")
        stderr_handle = (output_dir / f"worker_gpu{physical_gpu}.stderr.log").open("wb")
        process = subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        processes[physical_gpu] = process
        handles[physical_gpu] = (stdout_handle, stderr_handle)
        worker_outputs[physical_gpu] = worker_output
        print(
            f"launched {model_dir.name} stress worker on physical GPU{physical_gpu} "
            f"as PID {process.pid}",
            flush=True,
        )

    monitor_rows: list[dict[str, Any]] = []
    monitor_errors: list[str] = []
    while any(process.poll() is None for process in processes.values()):
        try:
            sampled_at = utc_now()
            elapsed = time.monotonic() - launched_at
            for row in query_gpus():
                if int(row["index"]) in requested_gpus:
                    monitor_rows.append(
                        {"sampled_at_utc": sampled_at, "elapsed_seconds": elapsed, **row}
                    )
        except BaseException as error:
            monitor_errors.append(f"{utc_now()} {type(error).__name__}: {error}")
        time.sleep(args.monitor_interval)

    return_codes: dict[int, int] = {}
    for gpu, process in processes.items():
        return_codes[gpu] = int(process.wait())
        handles[gpu][0].close()
        handles[gpu][1].close()

    monitor_path = output_dir / "gpu_monitor.csv"
    if monitor_rows:
        with monitor_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(monitor_rows[0]))
            writer.writeheader()
            writer.writerows(monitor_rows)
    if monitor_errors:
        (output_dir / "gpu_monitor_errors.log").write_text(
            "\n".join(monitor_errors) + "\n", encoding="utf-8"
        )

    try:
        after = query_gpus()
        full_smi_after = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True
        ).stdout
        after_error = None
    except BaseException as error:
        after = []
        full_smi_after = ""
        after_error = f"{type(error).__name__}: {error}"
    (output_dir / "gpu_state_after.json").write_text(
        json.dumps({"rows": after, "error": after_error}, indent=2), encoding="utf-8"
    )
    (output_dir / "nvidia_smi_after.txt").write_text(full_smi_after, encoding="utf-8")

    worker_results: dict[str, Any] = {}
    for gpu, path in worker_outputs.items():
        if path.exists():
            worker_results[str(gpu)] = json.loads(path.read_text(encoding="utf-8"))
        else:
            worker_results[str(gpu)] = {"status": "missing", "path": str(path)}

    hardware_summary: dict[str, Any] = {}
    for gpu in requested_gpus:
        active_rows = [
            row
            for row in monitor_rows
            if int(row["index"]) == gpu and float(row["memory.used"]) > 512
        ]
        hardware_summary[str(gpu)] = {
            "active_monitor_samples": len(active_rows),
            "utilization_pct": numeric_summary(
                [float(row["utilization.gpu"]) for row in active_rows]
            ),
            "memory_used_mib": numeric_summary(
                [float(row["memory.used"]) for row in active_rows]
            ),
            "temperature_c": numeric_summary(
                [float(row["temperature.gpu"]) for row in active_rows]
            ),
            "power_draw_w": numeric_summary(
                [float(row["power.draw"]) for row in active_rows]
            ),
            "sm_clock_mhz": numeric_summary(
                [float(row["clocks.sm"]) for row in active_rows]
            ),
        }

    samples_by_time: dict[str, set[int]] = {}
    for row in monitor_rows:
        if float(row["memory.used"]) > 512:
            samples_by_time.setdefault(str(row["sampled_at_utc"]), set()).add(int(row["index"]))
    active_timepoints = [indices for indices in samples_by_time.values() if indices]
    simultaneous_timepoints = sum(indices == {0, 1} for indices in active_timepoints)
    simultaneous_fraction = (
        simultaneous_timepoints / len(active_timepoints) if active_timepoints else 0.0
    )

    workers_passed = all(
        return_codes[gpu] == 0
        and worker_results[str(gpu)].get("status") == "passed"
        and worker_results[str(gpu)].get("finite_loss") is True
        for gpu in requested_gpus
    )
    after_visible = {int(row["index"]) for row in after} >= {0, 1}
    temperatures_safe = all(
        hardware_summary[str(gpu)]["temperature_c"]["max"] is not None
        and hardware_summary[str(gpu)]["temperature_c"]["max"] < 90
        for gpu in requested_gpus
    )
    passed = bool(
        workers_passed
        and after_visible
        and temperatures_safe
        and simultaneous_timepoints >= 5
        and not monitor_errors
    )
    summary = {
        "status": "passed" if passed else "failed",
        "protocol": (
            "Two independent native-Windows FP32 PlumSPECTRA training workers, one isolated "
            "per physical RTX 3090, using duplicated batches drawn from real project spectra. "
            "Diagnostic only; no production weights are overwritten."
        ),
        "started_at_utc": datetime.fromtimestamp(
            time.time() - (time.monotonic() - launched_at), tz=timezone.utc
        ).isoformat(),
        "finished_at_utc": utc_now(),
        "requested_duration_seconds_per_worker": args.duration_seconds,
        "batch_size_per_worker": args.batch_size,
        "return_codes": {str(key): value for key, value in return_codes.items()},
        "worker_results": worker_results,
        "hardware_summary": hardware_summary,
        "simultaneous_active_timepoints": simultaneous_timepoints,
        "active_timepoints": len(active_timepoints),
        "simultaneous_active_fraction": simultaneous_fraction,
        "monitor_errors": monitor_errors,
        "both_gpus_visible_after_test": after_visible,
        "maximum_safe_temperature_c": 90,
        "pass_criteria": {
            "both_workers_exit_zero_with_finite_loss": workers_passed,
            "both_gpus_visible_after_test": after_visible,
            "peak_temperature_below_90_c": temperatures_safe,
            "at_least_five_simultaneous_active_samples": simultaneous_timepoints >= 5,
            "no_nvidia_smi_monitor_errors": not monitor_errors,
        },
    }
    (output_dir / "stress_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Dual-GPU PlumSPECTRA stress test",
        "",
        f"- Overall status: **{summary['status'].upper()}**",
        f"- Duration requested per worker: {args.duration_seconds:.1f} s",
        f"- Batch size per worker: {args.batch_size}",
        f"- Simultaneous-active fraction: {simultaneous_fraction:.1%}",
        "- Production model files were read-only; all optimizer updates remained in memory.",
        "",
        "| GPU | Trait | Iterations | Samples/s | Peak reserved MiB | Mean util. | Peak util. | Peak temp. | Peak power |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gpu in requested_gpus:
        worker = worker_results[str(gpu)]
        hardware = hardware_summary[str(gpu)]
        report_lines.append(
            "| {gpu} | {trait} | {iterations} | {throughput:.1f} | {memory:.0f} | "
            "{mean_util:.1f}% | {max_util:.1f}% | {temperature:.0f} C | {power:.1f} W |".format(
                gpu=gpu,
                trait=worker.get("trait", "ERROR"),
                iterations=int(worker.get("iterations", 0)),
                throughput=float(worker.get("samples_per_second", 0.0)),
                memory=float(worker.get("peak_memory_reserved_mib", 0.0)),
                mean_util=float(hardware["utilization_pct"]["mean"] or 0.0),
                max_util=float(hardware["utilization_pct"]["max"] or 0.0),
                temperature=float(hardware["temperature_c"]["max"] or 0.0),
                power=float(hardware["power_draw_w"]["max"] or 0.0),
            )
        )
    report_lines.extend(["", "## Pass criteria", ""])
    for key, value in summary["pass_criteria"].items():
        report_lines.append(f"- [{'x' if value else ' '}] {key}")
    (output_dir / "STRESS_TEST_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded dual-GPU stress test using real PlumSPECTRA models and spectra"
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=90.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sync-interval", type=int, default=2)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.duration_seconds < 10:
        parser.error("--duration-seconds must be at least 10")
    if args.batch_size < 32:
        parser.error("--batch-size must be at least 32")
    if args.warmup_steps < 1 or args.sync_interval < 1:
        parser.error("warmup and sync settings must be positive")
    if args.worker:
        required = [args.physical_gpu_index, args.model_dir, args.worker_output]
        if any(value is None for value in required):
            parser.error("worker mode requires physical GPU, model directory, and worker output")
    elif args.output_dir is None:
        parser.error("controller mode requires --output-dir")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        run_controller(parsed)
