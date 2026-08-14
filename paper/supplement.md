# Supplementary information

## PlumSPECTRA: Cultivar-aware residual learning from near-infrared spectra for multidimensional intact-plum texture prediction

**Unit traceability note:** Several frozen supplementary diagnostic figures retain the force units recorded in the recovered instrument export (gram-force, gf). The manuscript and publication-facing summaries report force in newtons (N), using 1 gf = 0.00980665 N; this deterministic conversion does not alter any fitted model, ranking, residual or statistical inference.

This supplement provides figures, numerical tables and reproducibility identifiers for the analysis. Contrasts against nested RBF-SVR or the residual CNN are component comparisons because those predictors are the two branches averaged by PlumSPECTRA. Supplementary Fig. S25 uses the 60 primary CNN fits.

## Supplementary figures

### Supplementary Figure S1. Foldwise RMSE stability

![Supplementary Figure S1](../results/v26_claudecode_integration/supplementary_figures/figS01_foldwise_rmse_stability.png)

Five frozen outer-fold RMSE values for global PLSR, cultivar-aware PLSR, nested RBF-SVR and PlumSPECTRA across all 12 traits. The nested RBF-SVR is one of PlumSPECTRA’s two branches, not an independent method. Lines connect the same model, not repeated biological experiments.

### Supplementary Figure S2. Training-internal branch eligibility

![Supplementary Figure S2](../results/v26_claudecode_integration/supplementary_figures/figS02_quality_branch_selection.png)

Domain-SVR degradation relative to domain-PLSR in inner CV. The 5% eligibility threshold was fixed without outer-test labels. All final folds were eligible after baseline correction.

### Supplementary Figures S3–S14. Cultivar-resolved observed–predicted panels

![Supplementary Figure S3](../results/v26_claudecode_integration/supplementary_figures/figS03_FW_cultivar_observed_predicted.png)

![Supplementary Figure S4](../results/v26_claudecode_integration/supplementary_figures/figS04_SSC_cultivar_observed_predicted.png)

![Supplementary Figure S5](../results/v26_claudecode_integration/supplementary_figures/figS05_pH_cultivar_observed_predicted.png)

![Supplementary Figure S6](../results/v26_claudecode_integration/supplementary_figures/figS06_SRF_cultivar_observed_predicted.png)

![Supplementary Figure S7](../results/v26_claudecode_integration/supplementary_figures/figS07_RD_cultivar_observed_predicted.png)

![Supplementary Figure S8](../results/v26_claudecode_integration/supplementary_figures/figS08_PFD_cultivar_observed_predicted.png)

![Supplementary Figure S9](../results/v26_claudecode_integration/supplementary_figures/figS09_MFF_cultivar_observed_predicted.png)

![Supplementary Figure S10](../results/v26_claudecode_integration/supplementary_figures/figS10_F6_cultivar_observed_predicted.png)

![Supplementary Figure S11](../results/v26_claudecode_integration/supplementary_figures/figS11_LS_cultivar_observed_predicted.png)

![Supplementary Figure S12](../results/v26_claudecode_integration/supplementary_figures/figS12_LW_cultivar_observed_predicted.png)

![Supplementary Figure S13](../results/v26_claudecode_integration/supplementary_figures/figS13_PRW_cultivar_observed_predicted.png)

![Supplementary Figure S14](../results/v26_claudecode_integration/supplementary_figures/figS14_AF_cultivar_observed_predicted.png)

One figure per trait (FW, SSC, pH, SRF, RD, PFD, MFF, F6, LS, LW, PRW and AF). Every retained cultivar is shown; no group was removed using prediction residuals.

### Supplementary Figure S15. Conventional-quality PLSR VIP profiles

![Supplementary Figure S15](../results/v26_claudecode_integration/supplementary_figures/figS15_quality_pls_vip.png)

VIP profiles for fruit weight, SSC and pH under the frozen outer folds.

### Supplementary Figure S16. Authentic training dynamics

![Supplementary Figure S16](../results/v26_claudecode_integration/supplementary_figures/figS16_training_dynamics.png)

Recorded training and validation losses for all trait–fold fits. Curves are read from saved histories rather than simulated for illustration. ROC/AUC curves are not applicable because all 12 endpoints are continuous and every fitted task is regression rather than classification.

### Supplementary Figure S17. Full cultivar-cluster intervals

![Supplementary Figure S17](../results/v26_claudecode_integration/supplementary_figures/figS17_cluster_bootstrap_contrasts.png)

Paired cultivar-cluster 95% intervals for final-model RMSE reduction versus prespecified comparators. Positive values favour PlumSPECTRA.

### Supplementary Figure S18. Fruit-weight multiseed robustness

![Supplementary Figure S18](../results/v26_claudecode_integration/supplementary_figures/figS18_FW_multiseed_robustness.png)

Primary and two additional complete-pipeline seeds for each outer fold. Seed variation is displayed separately from outer-fold and cultivar-cluster uncertainty.

### Supplementary Figure S19. Held-batch few-shot recovery

![Supplementary Figure S19](../results/v26_claudecode_integration/supplementary_figures/figS19_heldbatch_fewshot_r2.png)

Pooled R² after 0, 5, 10, 20, 40 or 80 labelled reference fruit per held batch. The zero-shot point is the uncalibrated held-batch result and is recorded under the no-adapter series; the batch-mean null has no zero-shot value because a batch mean requires at least one label. Batch-macro R² and RMSE summaries are provided in the accompanying machine-readable tables.

### Supplementary Figure S20. Held-batch calibration gain by endpoint

![Supplementary Figure S20](../results/v26_claudecode_integration/supplementary_figures/figS20_heldbatch_fewshot_gain.png)

Trait-specific change after intercept or regularised affine calibration. The same calibration fruit are used for every comparator.

### Supplementary Figure S21. Cultivar mechanical profiles

![Supplementary Figure S21](../results/v26_claudecode_integration/supplementary_figures/figS21_cultivar_texture_profiles.png)

Robustly standardised cultivar medians for nine texture endpoints with descriptive Ward clustering. Groups are cultivar-associated and are not interpreted as genetic clusters.

### Supplementary Figure S22. Multiplicity-adjusted contrasts

![Supplementary Figure S22](../results/v26_claudecode_integration/supplementary_figures/figS22_multiplicity_adjusted_contrasts.png)

Benjamini–Hochberg, Holm and max-studentised results for the primary 36-comparison family and the post hoc 12-comparison strongest-baseline family.

### Supplementary Figure S23. Whole-cultivar QC decision

![Supplementary Figure S23](../results/v26_claudecode_integration/supplementary_figures/figS23_cultivar_qc_decision.png)

Independent measurement-quality domains, fruit counts and duplicate-measurement reliability for every source cultivar. The rule was fixed without model residuals.

### Supplementary Figure S24. Complete-pipeline optimisation stability

![Supplementary Figure S24](../results/v26_claudecode_integration/supplementary_figures/figS24_alltrait_multiseed_stability.png)

Primary versus three-seed prediction-mean fold SE, within-fold seed SD and cultivar-cluster intervals for all 12 traits. Random seeds are not biological replicates.

### Supplementary Figure S25. Current-model wavelength evidence

![Supplementary Figure S25](../results/v28_submission_strengthening/figures/Figure_S25_v28_wavelength_evidence.png)

**(a)** Median wavelength-attention allocation over the five frozen outer folds for each trait, expressed relative to uniform attention. Dashed vertical lines at 920 and 1685 nm identify instrument-edge caution regions. Enrichment is modest and all trait-wise maxima occur at one of the instrument endpoints; those maxima are treated as sensitivity warnings rather than biochemical markers. **(b)** Median within-cultivar Pearson correlation between SNV absorbance and phenotype at each wavelength, with cultivars receiving equal weight. Broad, trait-dependent association is more evident than a single common band. Attention is model based and correlation is observational; neither analysis supports a causal or compound-specific assignment.

All supplementary figures except the conventional-quality PLSR VIP panel (S15) can be regenerated from evidence tables in the reproducibility package. The VIP source is held in the project repository and will be deposited with the public code and data release. The figure-data bundle includes the endpoint PCA loadings underlying Table S12 and the attention and within-cultivar wavelength tables underlying Fig. S25.

## Supplementary tables

Machine-readable CSV, Parquet or JSON files accompany the rendered tables. Values were generated from fixed predictions, registries or programmatic audits.

### Table S1. Cultivar cohort counts

Source, strict-QC, analysis-tier, texture-modelling and complete-case counts by canonical cultivar.

### Table S2. Trait definitions

Endpoint code, source variable, algorithmic definition, reporting unit and modelling cohort. Recovered gram-force is converted to newtons for publication; position-derived endpoints retain archive position units (APU) because the Motor Steps/mm calibration constant is absent from the retained ARC records and no conversion to millimetres is asserted. The three-letter codes are compatibility codes: no rupture-event detection is performed.

### Table S3. Pooled OOF metrics

RMSE is in the archived modelling units used to fit each target. Table S41 provides publication-unit errors.

| Trait | n | RMSE | MAE | R² | RPIQ |
| --- | --- | --- | --- | --- | --- |
| FW | 4843 | 12.165 | 9.042 | 0.827 | 3.918 |
| SSC | 4843 | 1.576 | 1.181 | 0.629 | 1.903 |
| pH | 4843 | 0.361 | 0.288 | 0.544 | 2.381 |
| SRF | 4853 | 163.488 | 126.103 | 0.681 | 2.405 |
| RD | 4853 | 0.497 | 0.373 | 0.642 | 1.914 |
| PFD | 4853 | 158.658 | 121.401 | 0.645 | 2.367 |
| MFF | 4853 | 38.923 | 28.659 | 0.580 | 2.249 |
| F6 | 4853 | 45.633 | 30.875 | 0.502 | 1.976 |
| LS | 4853 | 70.502 | 52.303 | 0.719 | 2.715 |
| LW | 4853 | 534.945 | 408.014 | 0.547 | 1.983 |
| PRW | 4853 | 413.515 | 295.324 | 0.583 | 2.045 |
| AF | 4853 | 30.106 | 21.263 | 0.561 | 1.914 |

#### Table S3b. Pooled OOF RMSE of principal comparators

| Trait | Cultivar-aware PLSR | Global PLSR | No-neural B50 | Residual CNN |
| --- | --- | --- | --- | --- |
| FW | 12.392 | 25.058 | 12.403 | 12.018 |
| SSC | 1.632 | 1.808 | 1.593 | 1.590 |
| pH | 0.369 | 0.446 | 0.364 | 0.363 |
| SRF | 172.664 | 229.948 | 167.822 | 163.838 |
| RD | 0.521 | 0.586 | 0.508 | 0.500 |
| PFD | 163.624 | 219.936 | 161.119 | 158.985 |
| MFF | 40.372 | 48.399 | 39.528 | 39.081 |
| F6 | 46.896 | 55.072 | 46.297 | 45.573 |
| LS | 74.879 | 99.231 | 73.448 | 69.568 |
| LW | 553.641 | 662.746 | 546.348 | 532.238 |
| PRW | 436.404 | 528.197 | 423.546 | 414.292 |
| AF | 31.482 | 35.606 | 30.669 | 30.242 |

### Table S4. Foldwise metrics

Five outer-fold metrics for every trait and prespecified model.

### Table S5. Cultivar-centred metrics

Metrics after observed and predicted values were centred within cultivar.

| Trait | Cultivar-aware PLSR | Nested RBF-SVR | No-neural B50 | PlumSPECTRA | Residual CNN |
| --- | --- | --- | --- | --- | --- |
| FW | 0.020 | 0.011 | 0.019 | 0.057 | 0.083 |
| SSC | 0.331 | 0.340 | 0.363 | 0.378 | 0.370 |
| pH | -0.025 | 0.007 | -0.000 | 0.018 | 0.011 |
| SRF | -0.060 | 0.018 | -0.001 | 0.053 | 0.057 |
| RD | 0.115 | 0.158 | 0.161 | 0.200 | 0.203 |
| PFD | -0.027 | 0.016 | 0.004 | 0.037 | 0.039 |
| MFF | -0.009 | 0.019 | 0.033 | 0.063 | 0.058 |
| F6 | -0.035 | -0.007 | -0.009 | 0.021 | 0.026 |
| LS | -0.020 | 0.035 | 0.018 | 0.097 | 0.123 |
| LW | -0.032 | 0.006 | -0.005 | 0.037 | 0.050 |
| PRW | -0.031 | 0.030 | 0.029 | 0.078 | 0.085 |
| AF | 0.031 | 0.078 | 0.080 | 0.118 | 0.122 |

### Table S6. Cultivar-resolved metrics

Performance for each cultivar–trait–model cell, including difficult or negative cells.

### Table S7. Primary cultivar-cluster comparisons

Paired cluster-bootstrap effect estimates and percentile intervals.

### Table S8. Fold hyperparameters

PLSR preprocessing/components, RBF-SVR preprocessing/C/gamma/epsilon, selected residual gate and branch eligibility for all 60 folds.

### Table S9. Residual-gate audit

Internal selected gate, legacy sensitivity gate, checkpoint epoch, cross-fitted anchor folds and test-label-use assertions.

### Table S10. Final-cohort texture reliability

Duplicate ICC and replicate variability for nine endpoints in the modelling cohort.

| Trait | n | ICC(A,1) | Replicate r | Median CV (%) |
| --- | --- | --- | --- | --- |
| SRF | 4853 | 0.916 | 0.916 | 7.1 |
| RD | 4853 | 0.898 | 0.898 | 6.6 |
| PFD | 4853 | 0.857 | 0.858 | 9.4 |
| MFF | 4853 | 0.913 | 0.922 | 13.1 |
| F6 | 4851 | 0.801 | 0.805 | 17.2 |
| LS | 4853 | 0.984 | 0.985 | 3.5 |
| LW | 4853 | 0.921 | 0.925 | 7.8 |
| PRW | 4853 | 0.928 | 0.933 | 12.6 |
| AF | 4853 | 0.909 | 0.909 | 9.7 |

### Table S11. Endpoint correlation matrix

Pairwise texture-phenotype correlations.

### Table S12. Endpoint PCA variance and loadings

Explained variance and loading matrices used to quantify endpoint redundancy. The first three texture-only components explain 95.4% of variance.

### Table S13. Cultivar batch counts

Identifiable batch count and fruit count for every cultivar. Thirteen of 15 retained cultivars have one identifiable batch.

### Table S14. Within-cultivar batch effects

Descriptive batch differences for the two cultivars with multiple batches.

### Table S15. OOF cultivar-mean null and centred R²

Final, null and centred performance in one table.

### Table S16. Extended cluster comparisons

Deep-only, no-neural B50, cultivar-mean null and fixed-gate sensitivity comparisons (five candidate–baseline pairs × 12 traits). The deep-only row contrasts PlumSPECTRA with its own convolutional branch and is a component contrast; all twelve of those comparisons reach statistical parity.

### Table S17. Primary 36-comparison multiplicity family

Raw bootstrap probability, BH, Holm and simultaneous-interval status.

### Table S18. Strongest-baseline 12-comparison families

The rendered table gives the post hoc branch-excluded family: the best pooled OOF model among global PLSR, cultivar-aware PLSR and no-neural B50 was selected per trait. Both individual PlumSPECTRA branches were excluded, and 12/12 max-studentised simultaneous intervals were above zero. The machine-readable companion also retains the branch-inclusive sensitivity, which selected among global PLSR, cultivar-aware PLSR, nested RBF-SVR and B50 and supported 8/12 simultaneous contrasts.

| Trait | Selected baseline | Gain (%) | Simultaneous 95% CI (%) | Supported |
| --- | --- | --- | --- | --- |
| FW | Cultivar-aware PLSR | 1.83 | 0.58 to 3.09 | Yes |
| SSC | No-neural B50 | 1.09 | 0.27 to 1.91 | Yes |
| pH | No-neural B50 | 0.86 | 0.17 to 1.55 | Yes |
| SRF | No-neural B50 | 2.58 | 1.07 to 4.10 | Yes |
| RD | No-neural B50 | 2.09 | 0.35 to 3.83 | Yes |
| PFD | No-neural B50 | 1.53 | 0.49 to 2.57 | Yes |
| MFF | No-neural B50 | 1.53 | 0.48 to 2.58 | Yes |
| F6 | No-neural B50 | 1.43 | 0.58 to 2.29 | Yes |
| LS | No-neural B50 | 4.01 | 1.64 to 6.38 | Yes |
| LW | No-neural B50 | 2.09 | 0.49 to 3.68 | Yes |
| PRW | No-neural B50 | 2.37 | 0.61 to 4.12 | Yes |
| AF | No-neural B50 | 1.84 | 0.11 to 3.56 | Yes |

### Table S19. Equal-information 12-response PLS2 comparison

PLS2 metrics and paired comparisons on the common 4,843-fruit complete-case cohort.

### Table S20. Seed-by-fold metrics

Primary and two additional complete-pipeline seeds for deep-only and selected-final predictions.

### Table S21. Three-seed prediction-mean fold metrics

Predictions averaged within fruit across the three complete runs.

### Table S22. Multiseed cultivar-cluster comparisons

Prediction-mean cultivar-cluster contrasts for the selected final ensemble and for deep-only, each against global PLSR, cultivar-aware PLSR, nested RBF-SVR, the no-neural B50 ensemble and the cultivar-mean null (120 trait–candidate–baseline rows over 15 cultivar clusters). Only the cultivar-mean null is a null control.

### Table S23. Multiseed summary

Primary and prediction-mean fold effects, seed SD and cluster-support status for all 12 traits.

### Table S24. Additional seed metadata

Seed, checkpoint epoch, selected gate, anchor preprocessing/components, GPU and training protocol for 120 additional fits.

### Table S25. Same-cultivar held-batch pooled and macro metrics

Five held batches from two cultivars; pooled and batch-macro summaries are both reported.

### Table S26. Held-batch per-batch metrics

Absolute performance for each trait, model and batch.

### Table S27. Held-batch descriptive bootstrap comparisons

Five-batch cluster intervals; these are descriptive because only five clusters are available.

### Table S28. Held-batch train-internal branch choices

Eligibility and hyperparameters fitted without held-batch labels.

### Table S29. Few-shot calibration summary

Spectral ensemble and matched no-spectra batch-mean null after 0–80 labelled fruit.

### Table S30. Minimum calibration size

First shot count at which each prespecified criterion is met, where applicable.

### Table S31. Held-batch claim audit

Machine-readable claim disposition for direct transfer and calibrated updating.

### Table S32. LOCO PLSR fold metrics

Nine targets × 15 held-out cultivars for the separate leave-one-cultivar-out PLSR transfer test, not for PlumSPECTRA. Cultivar-macro R² is negative for every endpoint, and 124 of the 135 cultivar–endpoint cells are below zero.

### Table S33. LOCO PLSR selected hyperparameters

Train-internal preprocessing and component selection for each held-out cultivar. Only PLSR preprocessing and component counts are recorded, because this transfer test fits a PLSR rather than PlumSPECTRA.

### Table S34. Cultivar-exclusion performance sensitivity

Primary cohort compared with prespecified cultivar-retention/removal sensitivities. This table is not used to choose the reported cohort.

### Table S35. Cultivar 6.11 repeatability sensitivity

Endpoint-level duplicate reliability with and without the excluded cultivar.

### Table S36. Cultivar 6.11 domain decomposition

Spectral signal and acquisition-session variables reported in separate domains.

### Table S37. Wide-epsilon replacement audit

Strict-superset validation and archived-source identity for 15 earlier epsilon-boundary folds.

### Table S38. Baseline metadata repair audit

Audit of recorded search-space literals; no prediction was altered by literal-only repairs.

### Table S39. ARC coordinate-unit audit

Marker presence, retained numeric calibration status and kinematic evidence.

### Table S40. Scientific release checks

Programmatic checks of OOF uniqueness, truth matching, train-only selection, search-boundary convergence, row counts, transfer analyses and QC invariants.

### Table S41. Native-unit prediction and repeatability context

Publication units are g for fruit weight, percentage points for SSC, pH units for pH, N for force endpoints, APU for peak-referenced position, N·APU⁻¹ for loading stiffness and N·APU for work endpoints.

| Trait | Unit | Median AE | 80th-pct AE | Within 0.5 IQR (%) | R² | RPIQ |
| --- | --- | --- | --- | --- | --- | --- |
| FW | g | 6.638 | 14.852 | 94.3 | 0.827 | 3.918 |
| SSC | percentage points | 0.928 | 1.848 | 71.1 | 0.629 | 1.903 |
| pH | pH units | 0.245 | 0.458 | 76.7 | 0.544 | 2.381 |
| SRF | N | 0.977 | 1.970 | 79.0 | 0.681 | 2.405 |
| RD | APU | 0.291 | 0.584 | 70.8 | 0.642 | 1.914 |
| PFD | N | 0.932 | 1.911 | 78.8 | 0.645 | 2.367 |
| MFF | N | 0.207 | 0.453 | 78.3 | 0.580 | 2.249 |
| F6 | N | 0.213 | 0.476 | 77.6 | 0.502 | 1.976 |
| LS | N APU^-1 | 0.379 | 0.847 | 83.8 | 0.719 | 2.715 |
| LW | N APU | 3.126 | 6.328 | 71.3 | 0.547 | 1.983 |
| PRW | N APU | 2.057 | 4.585 | 76.4 | 0.583 | 2.045 |
| AF | N | 0.149 | 0.322 | 74.9 | 0.561 | 1.914 |

#### Table S41b. Prediction relative to duplicate reliability

| Trait | ICC(A,1) | Median duplicate CV (%) | R² / ICC (%) |
| --- | --- | --- | --- |
| SRF | 0.916 | 7.1 | 74.4 |
| RD | 0.898 | 6.6 | 71.5 |
| PFD | 0.857 | 9.4 | 75.2 |
| MFF | 0.913 | 13.1 | 63.6 |
| F6 | 0.801 | 17.2 | 62.7 |
| LS | 0.984 | 3.5 | 73.1 |
| LW | 0.921 | 7.8 | 59.4 |
| PRW | 0.928 | 12.6 | 62.8 |
| AF | 0.909 | 9.7 | 61.7 |

### Table S42. Deployment model card

Machine-readable intended use, required input, supported calibration domain, unsupported zero-shot uses, batch-update evidence, reference-fruit guidance, strongest and weakest texture outputs, archive-unit limitation and monitoring triggers. This table converts the transfer results into bounded operational guidance without asserting external validation.

## Reproducibility identifiers

- Source ledger: 5,502 fruit.
- Strict release tier: 4,967 fruit.
- Analysis tier: 5,430 fruit.
- High-confidence technical exclusions: 72 fruit.
- Texture cohort: 4,853 fruit × 9 endpoints = 43,677 OOF predictions.
- Conventional complete-case cohort: 4,843 fruit × 3 traits = 14,529 OOF predictions.
- Integrated primary prediction table: 58,206 rows, 58,206 unique fruit–trait pairs.
- Texture manifest SHA-256: `363ad2174d53d7eb2dcbeb8f2cecfb3cb32da98db3b3ba6176da32f43bf29a69`.
- Conventional manifest SHA-256: `7f859700cf7386e571f48305b53b922922b7d07338d7e8051b281351b59ad155`.
- Primary models: 60 trait–fold fits.
- Additional seed models: 120 fits; 180 complete pipeline instances including primary.
- Same-cultivar held-batch: 1,256 unique fruit, five batches, two cultivars, 11,304 trait predictions.
- LOCO: 43,677 predictions, nine traits, 15 held-out cultivars.
- PLS2: 58,116 predictions on 4,843 common complete cases.
- Current-model wavelength interpretation: 60 primary fits, 12 traits and 228 wavelengths.
- Within-cultivar wavelength association: 12 traits × 228 wavelengths with equal cultivar weighting.

## Claim boundary

The primary estimand is interpolation among 15 retained cultivars in the observed acquisition domain. Five held-batch tests cover Konglongdan and Weiwang, the only cultivars with replicate batches, so their scope is batch transfer within two cultivars. LOCO PLSR is a negative unseen-cultivar sensitivity analysis on a separate model family. Transfer across years, orchards, instruments and prospectively balanced operators remains to be tested.
