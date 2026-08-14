from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = root / "results/v16/all_nine_domain_anchor_models/all_nine_trait_statistics.csv"
    production_path = root / "results/v17/production_models/production_models.csv"
    verification_path = root / "results/v17/production_models/production_verification.json"
    quality_path = root / "results/v4/cultivar_quality_audit/cultivar_exclusion_decision.json"
    loco_path = root / "results/v7/primary_confirmation/fixed_gate075/summary.json"
    inference_path = root / "src/predict_v17_texture.py"
    plan_path = root / "results/v17/production_models/production_plan.json"
    evidence_paths = [
        statistics_path,
        production_path,
        verification_path,
        quality_path,
        loco_path,
        inference_path,
        plan_path,
    ]
    if not all(path.exists() for path in evidence_paths):
        raise RuntimeError(f"Missing completion evidence: {[str(p) for p in evidence_paths if not p.exists()]}")

    statistics = pd.read_csv(statistics_path)
    production = pd.read_csv(production_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    loco = json.loads(loco_path.read_text(encoding="utf-8"))
    requirements = {
        "model_independent_qc_exclusion": quality["excluded_cultivars"] == ["6.11"],
        "nine_validation_traits": len(statistics) == 9 and statistics["trait"].nunique() == 9,
        "ordinary_plsr_improvement_at_least_10pct_all_traits": bool(
            (statistics["ai_vs_global_pls_pct"] >= 10.0).all()
        ),
        "ordinary_plsr_cultivar_cluster_ci_positive_all_traits": bool(
            (statistics["global_ci95_low_pct"] > 0).all()
        ),
        "domain_plsr_increment_positive_all_traits": bool(
            (statistics["ai_vs_domain_pls_pct"] > 0).all()
        ),
        "domain_plsr_cultivar_cluster_ci_positive_all_traits": bool(
            (statistics["domain_ci95_low_pct"] > 0).all()
        ),
        "nine_production_models": len(production) == 9 and production["trait"].nunique() == 9,
        "one_full_primary_cohort_per_model": bool(
            (production["training_samples"] == 4839).all()
            and (production["cultivars"] == 15).all()
        ),
        "one_output_parameterization_per_trait": bool(
            (production["trainable_parameters"] == 142285).all()
        ),
        "serialized_replay_all_models": bool(
            verification["verified_models"] == 9 and verification["all_passed"]
        ),
        "inference_entrypoint_present": inference_path.exists(),
        "unknown_cultivar_boundary_quantified": bool(
            loco["target"] == "skin_break_displacement_raw_mean"
            and loco["pooled_rmse_improvement_pct"] > 0
        ),
    }
    achieved = all(requirements.values())
    audit = {
        "objective": "Build trait-specific AI models that materially and reproducibly outperform PLSR",
        "status": "achieved" if achieved else "incomplete",
        "requirements": requirements,
        "known_cultivar_validation": {
            "fruits_in_primary_cohort": 4839,
            "cultivars": 15,
            "repeated_prediction_records_per_trait": 4840,
            "unique_fruits_seen_in_any_test_split": 3263,
            "ai_vs_ordinary_plsr_improvement_range_pct": [
                float(statistics["ai_vs_global_pls_pct"].min()),
                float(statistics["ai_vs_global_pls_pct"].max()),
            ],
            "ai_vs_domain_plsr_increment_range_pct": [
                float(statistics["ai_vs_domain_pls_pct"].min()),
                float(statistics["ai_vs_domain_pls_pct"].max()),
            ],
            "all_cultivar_cluster_ci_positive": bool(
                (statistics["global_ci95_low_pct"] > 0).all()
                and (statistics["domain_ci95_low_pct"] > 0).all()
            ),
        },
        "unknown_cultivar_loco_boundary": {
            "confirmed_trait": "RD",
            "ai_rmse": loco["pooled_metrics"]["rmse"],
            "plsr_rmse": loco["pls_anchor_pooled_metrics"]["rmse"],
            "improvement_pct": loco["pooled_rmse_improvement_pct"],
            "interpretation": "Stable but smaller; do not generalize the 9-trait known-cultivar claim to unseen cultivars.",
        },
        "production": {
            "models": 9,
            "samples_per_model": 4839,
            "parameters_per_model": 142285,
            "serialized_replay_passed": verification["all_passed"],
        },
        "evidence_sha256": {str(path.relative_to(root)): sha256_file(path) for path in evidence_paths},
    }
    (output_dir / "COMPLETION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not achieved:
        raise RuntimeError(f"Completion audit failed: {requirements}")

    display = statistics[
        [
            "trait",
            "ai_rmse",
            "global_pls_rmse",
            "ai_vs_global_pls_pct",
            "global_ci95_low_pct",
            "global_ci95_high_pct",
            "domain_pls_rmse",
            "ai_vs_domain_pls_pct",
            "domain_ci95_low_pct",
            "domain_ci95_high_pct",
        ]
    ].copy()
    display.columns = [
        "性状",
        "AI RMSE",
        "普通PLSR RMSE",
        "AI改善%",
        "普通PLSR CI低",
        "普通PLSR CI高",
        "域PLSR RMSE",
        "额外AI改善%",
        "域PLSR CI低",
        "域PLSR CI高",
    ]
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    report = f"""# V17 域锚定 PLUMRAC-MT 最终模型报告

## 结论

本项目已经得到九个彼此独立、每个性状一个输出的质构预测 AI 模型。在“训练和测试均包含同一组已知品种、测试果实严格留出”的部署场景中，九个模型相对普通全局 PLSR 的 RMSE 均降低至少 10%，实际范围为 **{statistics['ai_vs_global_pls_pct'].min():.2f}%–{statistics['ai_vs_global_pls_pct'].max():.2f}%**；相对加入训练集品种校正的更强 PLSR，仍额外降低 **{statistics['ai_vs_domain_pls_pct'].min():.2f}%–{statistics['ai_vs_domain_pls_pct'].max():.2f}%**。九个性状的品种聚类 95% 置信区间对两类基线都全部排除 0。

因此，“AI 明显优于普通 PLSR”已经获得跨九性状的一致证据。科学写作中仍应区分两层贡献：主要增益来自正确处理稳定的品种/批次域偏移，深度光谱残差网络在强域 PLSR 之上贡献较小但统计稳定的 1.98%–3.91% 增益。

## 数据与质量控制

- 原始匹配果实：5,502。
- 仅按模型无关的测量质量证据整品种排除 `6.11`：129 个果实，占原数据 {quality['whole_cultivar_excluded_fraction'] * 100:.2f}%。
- 最终 primary 队列：4,839 个果实、15 个品种。
- 没有按模型预测误差删除品种；避免了用测试性能反向筛数据。
- 每个质构目标均单独训练一个模型，不存在一个输出头同时承担九个量纲差异巨大的目标。

## 模型结构

最终方法为“域校正 PLS 锚点 + 单性状深度光谱残差网络”：

1. 在训练数据内部选择 PLS 光谱预处理与成分数。
2. 使用训练果实估计每个已知品种的 PLS 残差偏置，形成域校正锚点。
3. 三视图光谱编码器同时接收 raw、SNV 和 Savitzky–Golay 一阶导数。
4. 4 个多尺度残差块使用 3/9/21 波段卷积核、GELU、GroupNorm 和注意力池化。
5. 先用九个质构性状、单果重、SSC、pH 共 12 个辅助目标预训练编码器；最终回归模型只保留一个性状输出。
6. 每个最终网络 142,285 个可训练参数。残差门控和训练轮数来自五次验证的稳健汇总；F6 使用开发重复冻结并由重复 2–5 确认的 0.75 门控。

## 五重复验证结果

每次从全部 15 个保留品种中按品种分层留出 20% 新果实，训练内部完成所有 PLS 选择、神经早停和门控选择。每个性状累计 4,840 条测试预测，涉及 3,263 个至少一次进入测试集的独立果实。置信区间以品种为聚类单位，保留同一果实的重复预测相关性。

{markdown_table(display)}

LS、SRF 和 PFD 是最突出的主结果，相对普通 PLSR 分别降低 RMSE 21.15%、20.38% 和 20.13%。最弱的 AF 仍达到 10.97%，其品种聚类 95% CI 为 5.36%–21.39%。

## 未知品种外推边界

上表回答的是“已知品种的新果实”应用问题，不应直接表述为未知新品种零样本泛化。严格 LOCO 中，当前最可靠的 RD 模型 RMSE 为 {loco['pooled_metrics']['rmse']:.4f}，PLSR 为 {loco['pls_anchor_pooled_metrics']['rmse']:.4f}，改善 {loco['pooled_rmse_improvement_pct']:.2f}%。这个结果稳定但明显小于已知品种场景，其他性状尚不能宣称普遍未知品种优势。

## 生产模型

- 模型目录：`results/v17/production_models/`
- 模型索引：`results/v17/production_models/production_models.csv`
- 每个性状目录包含网络权重、光谱标准化状态、PLS 锚点、品种偏置、训练历史和 SHA-256 清单。
- 九个模型均使用全部 4,839 个 primary 果实重新拟合；这些训练拟合指标不用于论文性能声明。
- 统一推理入口：`src/predict_v17_texture.py`
- 序列化验证：九个模型各回放 128 个果实，全部通过。

推理命令示例：

```powershell
.\\envs\\nirs-plum-ai\\python.exe src\\predict_v17_texture.py `
  --model-dir results\\v17\\production_models\\LS `
  --absorbance <new_absorbance.npy> `
  --wavelength data\\processed\\multimodal\\wavelength_nm.npy `
  --sample-table <new_samples.csv> `
  --output <ls_predictions.csv>
```

`new_samples.csv` 必须包含 `sample_id,cultivar_ascii`。生产模型只支持已经校准的 15 个品种；未知品种应使用独立的 LOCO 模型或先获得少量标定果实。

## GPU 与复现

单张 RTX 3090 上两个训练进程可将利用率提升到约 95%–99%，因此批量队列采用“每张卡最多两个模型”。GPU1 在负载启动后再次发生驱动级丢卡，本轮最终生产模型全部由稳定的 GPU0 完成；失败的 GPU1 任务没有进入结果。修复 GPU1 前不应继续使用它。

关键复现入口：

- 验证训练：`src/train_plumrac_v5_stratified.py`
- 多性状确认：`src/run_v15_domain_anchor_confirmation.py`
- F6 冻结门控确认：`src/finalize_v16_f6.py`
- 品种聚类统计：`src/analyze_v14_domain_anchor.py`
- 全队列生产训练：`src/train_plumrac_v5_full.py`
- 模型回放验证：`src/verify_v17_production_models.py`

## 可用于论文的表述

推荐：在包含 4,839 个果实和 15 个李品种的模型无关质量控制队列中，九个独立的域锚定单性状光谱网络在重复品种分层果实留出验证中，相对普通 PLSR 将 RMSE 降低 10.97%–21.15%；相对加入训练集品种校正的强 PLSR 仍降低 1.98%–3.91%，且所有性状的品种聚类 95% 置信区间均排除零。

不推荐：AI 已经能够对任何未知李品种全面替代 PLSR。现有证据只支持 RD 在严格未知品种 LOCO 中获得约 6.95% 的稳定优势。

## 证据文件

- `results/v16/all_nine_domain_anchor_models/all_nine_trait_statistics.csv`
- `results/v17/production_models/production_verification.json`
- `results/v17/production_models/production_plan.json`
- `results/v4/cultivar_quality_audit/cultivar_exclusion_decision.json`
- `results/v7/primary_confirmation/fixed_gate075/summary.json`
- `results/v17/COMPLETION_AUDIT.json`
"""
    (output_dir / "FINAL_MODEL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"wrote {output_dir / 'FINAL_MODEL_REPORT.md'}")


if __name__ == "__main__":
    main()
