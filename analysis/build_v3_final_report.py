from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chinese V3 final technical report.")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3" / "final_evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "NIRs_plum_V3_final_model_report_zh.md",
    )
    args = parser.parse_args()

    evidence = args.evidence_dir.resolve()
    metrics = pd.read_csv(evidence / "all_model_pooled_metrics.csv")
    transfer = pd.read_csv(evidence / "plumrac_x_transfer_summary.csv")
    selected = pd.read_csv(evidence / "model_selection_reading.csv")
    audit = json.loads((evidence / "final_evidence_audit.json").read_text(encoding="utf-8"))
    frozen = json.loads((PROJECT_ROOT / "configs" / "v3_plumrac_x_frozen.json").read_text(encoding="utf-8"))
    rd_full = json.loads(
        (PROJECT_ROOT / "results" / "v3" / "plumrac_x_rd_full" / "rd_full_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rd_confirmation = json.loads(
        (
            PROJECT_ROOT
            / "results"
            / "v3"
            / "plumrac_x_rd_confirmation_analysis"
            / "confirmation_summary.json"
        ).read_text(encoding="utf-8")
    )
    cohort = json.loads(
        (
            PROJECT_ROOT / "results" / "v2" / "tables" / "cohort" / "cohort_summary.json"
        ).read_text(encoding="utf-8")
    )
    operator = pd.read_csv(
        PROJECT_ROOT / "results" / "v3" / "rd_operator_suite" / "selection" / "variant_summary.csv"
    )
    optimization = pd.read_csv(
        PROJECT_ROOT / "results" / "v3" / "rd_optimization_suite" / "selection" / "variant_summary.csv"
    )

    model_order = ["PLSR", "Ridge", "PlumRAC-Net V2.2", "PLUMRAC-X V3"]
    zero = metrics.copy()
    zero["model"] = pd.Categorical(zero["model"], model_order, ordered=True)
    zero = zero.sort_values(["trait", "model"])
    zero_table = pd.DataFrame(
        {
            "性状": zero["trait"],
            "模型": zero["model"].astype(str),
            "RMSE": zero["rmse"].map(lambda value: f"{value:.4g}"),
            "R²": zero["r2"].map(lambda value: f"{value:.3f}"),
            "CCC": zero["ccc"].map(lambda value: f"{value:.3f}"),
            "相对PLSR": zero["rmse_improvement_vs_plsr_pct"].map(pct),
        }
    )
    transfer_table = pd.DataFrame(
        {
            "性状": transfer["trait"],
            "种子": transfer["seeds"].astype(int),
            "品种胜场": transfer["fold_wins_vs_plsr"].astype(int).astype(str) + "/16",
            "宏平均RMSE改善": transfer["macro_rmse_improvement_pct"].map(pct),
            "总体RMSE改善": transfer["pooled_rmse_improvement_pct"].map(pct),
            "品种bootstrap 95% CI": transfer.apply(
                lambda row: (
                    f"[{row['cultivar_bootstrap_ci95_lower_pct']:+.2f}%, "
                    f"{row['cultivar_bootstrap_ci95_upper_pct']:+.2f}%]"
                ),
                axis=1,
            ),
        }
    )
    selection_table = pd.DataFrame(
        {
            "性状": selected["trait"],
            "观察到的最低RMSE模型": selected["lowest_observed_rmse_model"],
            "RMSE": selected["rmse"].map(lambda value: f"{value:.4g}"),
            "R²": selected["r2"].map(lambda value: f"{value:.3f}"),
            "部署判断": selected["evidence_based_deployment_reading"],
        }
    )
    operator_view = operator.sort_values("macro_rmse_improvement_pct", ascending=False).head(10).copy()
    operator_view["trainable_parameters"] = operator_view["path"].map(
        lambda value: int(json.loads((Path(value) / "summary.json").read_text(encoding="utf-8"))["trainable_parameters"])
    )
    operator_table = pd.DataFrame(
        {
            "候选": operator_view["variant"],
            "参数量": operator_view["trainable_parameters"].astype(int),
            "品种胜场": operator_view["fold_wins"].astype(int).astype(str) + "/5",
            "宏平均改善": operator_view["macro_rmse_improvement_pct"].map(pct),
            "总体改善": operator_view["pooled_rmse_improvement_pct"].map(pct),
        }
    )
    optimization_view = optimization.sort_values("macro_rmse_improvement_pct", ascending=False).copy()
    optimization_table = pd.DataFrame(
        {
            "候选": optimization_view["variant"],
            "品种胜场": optimization_view["fold_wins"].astype(int).astype(str) + "/5",
            "宏平均改善": optimization_view["macro_rmse_improvement_pct"].map(pct),
            "总体改善": optimization_view["pooled_rmse_improvement_pct"].map(pct),
        }
    )

    architecture = frozen["architecture"]
    optim = frozen["optimization"]
    confirmation_ci = rd_confirmation["cultivar_cluster_bootstrap"]
    sign_flip = rd_confirmation["exact_paired_sign_flip"]
    final_figure = (evidence / "fig_v3_final_model_evidence.png").as_posix()
    dataset_total = int(cohort.get("linked_fruits", cohort.get("identity_linked_fruits", 5502)))
    analysis_total = int(cohort.get("analysis_fruits", cohort.get("analysis_cohort_fruits", 5430)))
    strict_total = int(cohort.get("strict_texture_fruits", cohort.get("strict_texture_cohort_fruits", 4952)))

    content = f"""# 李子近红外—质构无损预测：V3 最终模型技术报告

## 1. 结论先行

本轮没有得到一个在九个质构性状上都“吊打”PLSR的深度模型。严格留一品种验证给出的结论更重要也更可信：**模型容量不是当前的主要瓶颈，跨品种域偏移才是**。冻结的336,290参数PLUMRAC-X在RD的五个开发品种上宏平均RMSE改善{frozen['development_result']['macro_rmse_improvement_pct']:.2f}%，但在11个从未参与开发的确认品种上只改善{rd_confirmation['macro_rmse_improvement_pct']:.2f}%，其95%品种聚类区间为[{confirmation_ci['ci95_lower_pct']:.2f}%, {confirmation_ci['ci95_upper_pct']:.2f}%]，精确配对符号翻转检验双侧P={sign_flip['p_two_sided']:.3f}，未达到预设优越性规则。

RD仍是最有价值的AI端点。紧凑的72,530参数PlumRAC-Net V2.2将总体RMSE从0.7016降至0.6857，改善2.27%，R²从0.338升至0.367；它比更大的PLUMRAC-X总体RMSE 0.6893更好。因此最终科研结论不是“CNN一定胜过PLSR”，而是：**在严格跨品种条件下，RD存在可重复的非线性增量信号；其余端点主要受品种基线偏移、参考测量噪声和有限独立品种域数量限制。**

这并不削弱质构主线。相反，5,430个果实、16个品种、每果两次ARC曲线和九个机械端点，使本项目可以回答一个文献中更少被严格检验的问题：近红外能否跨遗传背景迁移预测果皮破裂、果肉阻力、机械功和黏附，而不仅是随机拆分下预测SSC。

## 2. 数据资产与质构定位

- 身份关联完整果实约{dataset_total:,}个；高置信主分析队列{analysis_total:,}个；严格质构队列约{strict_total:,}个。
- 16个品种或育种材料、19个采集批次、11,004条ARC重复曲线、228个近红外波段。
- 九个端点分别独立建模：SRF、RD、PFD、MFF、F6、LS、LW、PRW和AF。
- 质构仪测量是果实表型参考值，与单果重、SSC和pH处于同一研究层级；部署输入仍是质构测量之前采集的完整果实NIR。
- 主模型仅排除1.3%具有强技术异常证据的样本；10%严格队列用于数据发布和敏感性分析，避免通过删除难预测果实人为抬高模型分数。

## 3. 为什么采用留一品种验证

随机按果实拆分会把同一品种、同一批次的光谱基线同时放入训练集和测试集，严重高估未来新品种的性能。本项目以“性状×留出品种”为独立评价单元：每个端点建立自己的PLSR、Ridge和神经网络，共16个外层折。目标品种完全不参与光谱预处理、超参数选择、早停或门控选择。

虽然有5,430个果实，但决定跨品种泛化能力的独立域只有16个。因此深度网络面对的是“大量域内样本、极少独立域”的统计结构，而不是普通意义上的五千多个独立训练环境。这正是强低秩偏置的PLSR可能优于高自由度CNN的原因。

## 4. V2.2：单性状、残差锚定、安全回退

V2.2从一开始就遵循“一性状一模型”，不存在一个网络同时承担九个目标。每个模型先拟合嵌套PLSR锚点，再由一维残差网络学习PLSR未解释部分，最终预测为`PLSR + g × neural residual`。安全门控只允许g取0、0.25或0.50；如果源品种交叉验证不能证明残差在所有验证品种上都安全，则精确退回PLSR。

该设计的价值在于把AI的任务从“重建整条光谱—表型关系”缩小为“只学习可迁移的非线性增量”。RD获得明确正向结果，而F6等端点可以无损回退。它不是为了让每个性状都宣称AI胜出，而是为了防止AI在未知品种上造成大幅损失。

## 5. V3诊断：问题不只是隐藏层和激活函数

在预先指定的五个开发品种L313、CHL、KLD、WW和WX中，目标品种的理想截距校正可降低约14%—34%的RMSE，品种折偏差占平方误差约25%—56%。PLSR、Ridge和V2残差相关性高达0.98—1.00，说明简单堆叠模型缺少可供平均抵消的独立误差。

开发阶段还检验了单果重、光谱距离和无标签目标批次上下文。单果重没有形成稳定的跨品种增益；源域光谱距离不能可靠预测某个未知品种上的误差；无标签批次上下文只在F6和MFF上出现局部开发收益，在其余性状上恶化。由此排除了若干看似合理但证据不足的捷径。

## 6. PLUMRAC-X架构与参数

PLUMRAC-X全称为Plum Residual-Anchored Cross-cultivar eXpert。其输入为RAW吸光度、SNV和Savitzky–Golay一阶导数三个通道；主体为宽度{architecture['width']}、{architecture['residual_blocks']}个残差块、GELU激活和GroupNorm，带波长注意力尾部，共{architecture['trainable_parameters']:,}个可训练参数。训练使用AdamW，学习率{optim['learning_rate']:.1e}、权重衰减{optim['weight_decay']:.1e}、dropout={optim['dropout']:.2f}、batch={optim['batch_size']}、最多{optim['max_epochs']}轮，并使用物理合理的尺度、偏移、斜率和微噪声增强。

GELU并非凭直觉固定。受控消融比较了GELU、SiLU、ReLU，GroupNorm和LayerNorm，16×2、32×3和64×4容量，品种平衡采样，以及七通道多视图导数输入。结果如下。

{markdown_table(operator_table)}

64×4相较32×3的参数量约增加4.6倍，但宏平均收益只增加约0.3个百分点。这个结果同时说明原网络略有容量不足，也说明继续无约束堆参数不是解决域偏移的办法。

<!-- PAGE_BREAK -->

## 7. 冻结前的有限超参数优化

V3只在64×4获胜模型周围检查学习率2.5e-4/5e-4/1e-3、dropout 0.05/0.12、是否使用光谱增强、以及完全品种平衡采样。候选定义在打开11个确认品种前写入协议，没有在看到确认结果后追加候选。

{markdown_table(optimization_table)}

最终冻结的是GELU+GroupNorm、64×4、sampler power 1.0、学习率5e-4、dropout 0.12和增强开启的组合。

## 8. RD的开发—确认分离结果

五个开发品种上，PLUMRAC-X五折全胜，宏平均RMSE改善7.01%。然而11个封存确认品种上只获得6/11胜场，宏平均改善0.98%、总体改善0.79%，R²从0.365升至0.375。四个随机种子的宏平均改善均为正，但品种聚类置信区间跨零，预设确认优越性规则未通过。

把16个品种合并作描述性汇总时，PLUMRAC-X获得11/16胜场，宏平均改善{rd_full['macro_rmse_improvement_pct']:.2f}%、总体改善{rd_full['pooled_rmse_improvement_pct']:.2f}%；但是其中五个品种参与了模型选择，不能把这一行当作纯独立确认。V2.2虽然只有8/16胜场，却具有更低的总体RMSE。因此生产零样本RD模型仍推荐V2.2，而不是更大的V3。

## 9. 九性状完整零样本结果

{markdown_table(zero_table)}

这里的连续性状基线是PLSR，不是PLS-DA。所有模型在完全相同的留出品种果实上评价。负的“相对PLSR”表示模型恶化。

## 10. 冻结V3跨性状转移

{markdown_table(transfer_table)}

除RD外，其余八个性状只先运行一个冻结种子。只有同时满足宏平均改善>1%、总体改善>0、至少6/16品种获胜、最差品种恶化不超过3%，才允许自动追加三个种子。这个规则防止反复换随机种子寻找有利结果。

## 11. 最终模型判断

{markdown_table(selection_table)}

![图1 冻结模型相对PLSR的总体RMSE变化与PLUMRAC-X跨品种胜场]({final_figure})

“观察到的最低RMSE模型”来自已经完成的LOCO结果，只能作为当前数据上的描述性选择，不是新的独立测试。若后续用于产品部署，仍需用新年份或新果园做外部验证。对于RD，V2.2是当前最有依据的零样本AI；对PFD、LW和PRW，Ridge分别更有优势；其余端点大多应保留PLSR或把AI视为等效安全模型。

## 12. 为什么强AI没有全面击败PLSR

1. **光谱关系主要是低秩和平滑的。** PLSR的归纳偏置恰好适合高共线的一维近红外光谱。
2. **独立域太少。** 五千多个果实不能替代更多年份、果园、仪器和品种域；网络更容易学习训练品种基线。
3. **域均值偏移压过果实内排序。** 未见品种上的截距变化很大，而源域验证难以判断哪次残差校正对目标域安全。
4. **质构参考值含真实测量噪声。** 更大网络会同时拟合换人、批次和重复曲线差异，未必学习更可迁移的生物信号。
5. **安全门控的目标是避免灾难，不是制造胜场。** 大多数端点回退PLSR正是门控在工作；少数误判折说明还需外部域信息或目标域校准。

所以，CNN低于PLSR并不自动证明代码错误。相反，受控激活、归一化、容量、学习率、正则化、增强和多种子实验均指向同一结论：增加自由度只能放大开发折收益，不能凭空创造未知品种上的可迁移信息。

## 13. 真正可提升性能的下一步

- **新域数据优先于继续堆网络。** 采集跨年份、果园、成熟期、操作员和仪器日的独立域；每个域不必极大，但域数必须增加。
- **把5—20个目标品种标注果实作为正式部署模式。** 现有少样本实验表明仅5个果实的截距校准即可显著提升绝对R²；论文中应严格称为rapid calibration，而不是zero-shot。
- **重复质构曲线进入不确定性模型。** 让网络预测均值和测量方差，降低低一致性果实在损失中的权重，但不能直接删除难预测样本。
- **外部自监督预训练。** 只有获得更多未标注NIR域后，光谱MAE/对比学习才可能提供比当前5,430条光谱更稳健的表征；在同一16域内反复预训练不能替代外部证据。
- **层级/域随机效应与神经残差联合。** 若部署允许获取目标批次少量标签，显式估计品种或批次截距比盲目扩大卷积网络更有效。

## 14. 论文主张边界与亮点

可支持的亮点是：大规模单果近红外与双重复机械曲线配对；九个物理可解释质构端点；严格LOCO而非随机拆分；单性状残差AI；开发—确认分离；以及零样本与少样本部署的明确区分。RD可表述为“紧凑残差网络获得小但统计上有支持的总体增益”；PLUMRAC-X只能表述为“开发收益在封存品种上衰减为稳定小幅平均增益，未达到预设确认优越性”。

不能声称已经直接预测消费者口感或贮藏天数，因为本实验没有感官小组和货架寿命标签；质构端点应表述为与口感、搬运耐受和采后品质相关的机械代理性状。

## 15. 可复现性与审计

V2.2全部九性状均保存16折模型、预测、门控元数据、种子、环境锁和CPU推理重建审计。V3保存冻结配置、开发候选、封存确认预测、跨性状盲筛、每折结果和SHA-256。最终证据审计状态为`{audit['status']}`；V3模型参数量{audit['v3_trainable_parameters']:,}，V2.2参数量{audit['v2_trainable_parameters']:,}。

最终图位于`results/v3/final_evidence/fig_v3_final_model_evidence.png`，核心表位于同目录。完整研发边界和每次候选冻结记录位于`reports/V3_model_strategy_log.md`。
"""

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
