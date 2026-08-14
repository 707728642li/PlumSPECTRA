from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

import train_plumrac_v4_phy as v4
import train_plumrac_v5_auxpretrain as v5


BASE_V4 = v4.V4PlumRACNet
GLOBAL_CONTEXT = "transformer"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TransformerGlobalContext(nn.Module):
    def __init__(self, width: int, dropout: float, length: int = 228) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, length, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.layer_scale = nn.Parameter(torch.full((1, 1, width), 1e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        tokens = value.transpose(1, 2)
        if tokens.shape[1] != self.position.shape[1]:
            raise ValueError(
                f"Transformer positional length {self.position.shape[1]} does not match {tokens.shape[1]}"
            )
        contextual = self.encoder(tokens + self.position)
        return (tokens + self.layer_scale * (contextual - tokens)).transpose(1, 2)


class DualAttentionGlobalContext(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(width // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.Linear(width * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.Sigmoid(),
        )
        self.wavelength_gate = nn.Sequential(
            nn.Conv1d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.channel_gate[-2].weight)
        nn.init.zeros_(self.channel_gate[-2].bias)
        nn.init.zeros_(self.wavelength_gate[-2].weight)
        nn.init.zeros_(self.wavelength_gate[-2].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        summary = torch.cat([value.mean(dim=-1), value.amax(dim=-1)], dim=1)
        channel = 2.0 * self.channel_gate(summary).unsqueeze(-1)
        wavelength = 2.0 * self.wavelength_gate(value)
        return value * channel * wavelength


class V6GlobalPlumRACNet(BASE_V4):
    def __init__(self, width: int, blocks: int, dropout: float, attention_tail: bool = True) -> None:
        super().__init__(width, blocks, dropout, attention_tail)
        if GLOBAL_CONTEXT == "transformer":
            self.global_context = TransformerGlobalContext(width, dropout)
        elif GLOBAL_CONTEXT == "dual_attention":
            self.global_context = DualAttentionGlobalContext(width)
        else:
            raise ValueError(GLOBAL_CONTEXT)

    def encode(self, channels: torch.Tensor) -> torch.Tensor:
        return self.global_context(super().encode(channels))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--global-context", choices=["transformer", "dual_attention"], required=True)
    variant, remaining = parser.parse_known_args()
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = variant.global_context
    original_argv = sys.argv
    original_class = v4.V4PlumRACNet
    try:
        v4.V4PlumRACNet = V6GlobalPlumRACNet
        sys.argv = [sys.argv[0], *remaining]
        v5.main()
    finally:
        sys.argv = original_argv
        v4.V4PlumRACNet = original_class

    output_dir = Path(v5.argument_value(remaining, "--output-dir")).resolve()
    report = {
        "model": "PLUMRAC-GC V6",
        "global_context": GLOBAL_CONTEXT,
        "base_model": "PLUMRAC-MT V5.1",
        "one_final_model_per_trait": True,
        "final_model_outputs": 1,
        "source_only_auxiliary_pretraining": True,
        "provenance_sha256": {
            "v6_global_trainer": sha256_file(Path(__file__).resolve()),
            "v5_auxiliary_trainer": sha256_file(Path(v5.__file__).resolve()),
            "v4_multiscale_architecture": sha256_file(Path(v4.__file__).resolve()),
        },
        "claim_boundary": "Retrospective development candidate; requires frozen all-cultivar confirmation.",
    }
    (output_dir / "v6_global_context.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model"] = f"PLUMRAC-GC V6 {GLOBAL_CONTEXT}"
    summary["v6_global_context"] = report
    summary["provenance_sha256"]["src\\train_plumrac_v6_global.py"] = report[
        "provenance_sha256"
    ]["v6_global_trainer"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
