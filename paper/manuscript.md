# PlumSPECTRA: Cultivar-aware residual learning from near-infrared spectra for multidimensional intact-plum texture prediction

**Running title:** Cultivar-aware NIR prediction of plum texture  
**Authors:** Anonymous for peer review  
**Affiliations:** Withheld for peer review  
**Corresponding author:** Withheld for peer review

## Abstract

Near-infrared (NIR) fruit models usually target soluble solids or one firmness value, while random evaluation can confound spectral information with cultivar identity. We developed PlumSPECTRA, which combines a cross-fitted cultivar-aware partial least-squares anchor, a trait-specific residual convolutional network and a nonlinear kernel expert. Spectra, mass, duplicate texture curves, soluble solids and pH were linked for 5,502 plums; 4,853 fruit from 15 cultivars passed quality control for nine texture endpoints. Frozen five-fold evaluation produced 58,206 out-of-fold predictions. R² was 0.827 for fruit weight, 0.629 for soluble solids, 0.544 for pH and 0.502–0.719 for texture. RMSE fell by 12.8–51.5% against global partial least squares, although that reference was weaker than a spectrum-free cultivar-mean predictor for 11 of 12 traits. In a post hoc branch-excluded comparison, gains over the best of global partial least squares, cultivar-aware partial least squares and a no-neural ensemble were 0.9–4.0%; all 12 simultaneous intervals excluded zero. The ensemble and its convolutional branch remained statistically indistinguishable. Cultivar-centred R² fell to 0.018–0.378. In five held batches from two cultivars, affine updating with 20–40 labelled fruit reduced median texture error by 15.4–16.5%; at 40 fruit, pooled R² was 0.187 versus 0.164 for a batch-mean control. Leave-one-cultivar-out regression failed for every texture endpoint on cultivar-macro evaluation. PlumSPECTRA supports multidimensional texture screening inside a monitored calibration domain, with labelled reference fruit required for transfer.

**Keywords:** near-infrared spectroscopy; residual learning; fruit texture; calibration transfer; deep learning; chemometrics

## Introduction

Rapid quality sensing is useful only when its targets reflect product quality and its validation resembles deployment. Fruit mass affects grading, soluble solids concentration (SSC) contributes to perceived sweetness, and pH records juice hydrogen-ion activity. None describes how fruit tissue deforms under load or behaves during probe withdrawal. Those mechanical responses matter for eating texture and handling, yet automated near-infrared (NIR) systems commonly reduce texture to one peak-force or firmness value.

Texture-analyser phenotyping is informative but labour intensive and locally destructive. Here, spectra were acquired from intact fruit before weighing, duplicate penetration tests, SSC and pH measurements. The spectrum can therefore estimate traits measured later in the workflow. Postharvest NIR studies have concentrated on SSC, dry matter and single firmness endpoints (Nicolaï et al., 2007; Walsh et al., 2020). Firmness prediction has been reported for apple, peach and plum (Lu et al., 2000; Fu et al., 2008; Paz et al., 2008; Louw and Theron, 2010; Pérez-Marín et al., 2010; Uwadaira et al., 2018), and recent work has extended portable sensing and multi-trait modelling (Posom et al., 2020; Scalisi and O’Connell, 2021; Masuda et al., 2023; Minas et al., 2023; Zhou et al., 2023; Bu et al., 2025). Deep networks have also been fitted directly to agricultural NIR data (Yang et al., 2021; Zhang et al., 2024). Far fewer studies connect pre-penetration spectra to a repeatability-audited panel that includes force, position, stiffness, work and adhesion.

Ordered spectra suit one-dimensional convolution (Malek et al., 2018), but their collinearity also favours partial least-squares regression (PLSR), and nonlinear kernels can be competitive. Cultivar identity adds another source of signal because it can dominate both spectral structure and phenotype means. Cross-batch and cross-cultivar prediction are consequently different problems from random-split accuracy (Li et al., 2021; Wu et al., 2023; Wang et al., 2024; Liu et al., 2025), and grouped observations require structured validation (Roberts et al., 2017). A comparison with global PLSR alone would mix fruit-level prediction with cultivar-associated baselines. We instead used a cross-fitted cultivar-aware anchor, trait-specific residual learning and separate tests of same-session interpolation, within-cultivar batch transfer and unseen-cultivar transfer.

The study links 11,004 archived texture curves to 5,502 fruit records. Its modelling design uses cross-fitted cultivar-aware residual targets, one compact convolutional model per trait, a train-internally selected kernel expert and equal-information controls. Frozen cultivar-stratified folds cover 12 targets. Held-batch and leave-one-cultivar-out audits then test where the fitted system stops transferring. The resulting resource places nine repeatability-audited texture endpoints alongside fruit weight, SSC and pH, with the deployment claim confined to a monitored calibration domain.

## Materials and methods

### Fruit workflow and identity reconciliation

The approximate measurement sequence was intact fruit → NIR spectrum → single-fruit mass → duplicate texture analysis → SSC → pH. Source filenames, cultivar aliases, batch labels and fruit numbers were reconciled into a canonical English identifier. Conflicts and corrections were preserved in machine-readable registries rather than overwritten silently.

### NIR spectra and preprocessing

Each retained primary `c`-code export comprised 228 absorbance bands on a common grid from 901.032 to 1700.792 nm. Representative headers identified instrument serial 5490277, a configured 900–1700-nm range, 5.080-ms exposure and six declared repeats. The archive did not retain the instrument make and model, optical geometry, fruit position or white/dark reference schedule, so `c` is used only as an archive scan code. Temperature, humidity and lamp-photodiode fields supported acquisition-session quality control and were withheld from the predictors. Prespecified spectral views were raw absorbance, within-spectrum standard normal variate (SNV), first-derivative Savitzky–Golay smoothing (11-point window, second-order polynomial; Savitzky and Golay, 1964), and combinations of these views. Scalers, preprocessing parameters and models were fitted inside the relevant training partition.

### Conventional and mechanical phenotypes

Fruit mass, SSC and pH are reported in grams, percentage points and pH units. The source field informally described as acidity was a pH measurement, not titratable acidity. Manufacturer, model and detailed assay protocols for the balance, refractometer and pH meter were not preserved in the source records.

Two Stable Micro Systems Exponent ARC curves per fruit represented nominal penetration positions without anatomical labels. The nine prespecified endpoints are defined in Table S2. Force required an archive-specific conversion: every ARC included an Exponent preview labelled `Force (g)`, and the preview peak matched the decoded channel multiplied by 1000 in one file sampled from each of 19 acquisition batches. Those files recorded Exponent 8.0.16.0 and a 30,000-g load-cell capacity. We used `force_gf = force_raw × 1000` for this archive family and converted gram-force to newtons for presentation (1 gf = 0.00980665 N). The files contained a `Motor Steps / mm` marker but no numeric calibration. Position, stiffness and work therefore retain archive position units (APU, N·APU⁻¹ and N·APU).

### Quality control and frozen cohorts

Quality control combined file and identity failures, severe acquisition anomalies, spectral signal, session conditions, duplicate reliability and chemically implausible values. Moderate flags marked the highest 20% of evidence ranks and required corroboration; a biological extreme alone did not trigger exclusion. Removing an entire cultivar required failure in at least two independent measurement domains. Neither model residuals nor test-set performance entered an exclusion decision.

The texture modelling cohort contained 4,853 fruit from 15 cultivars; 4,843 also had complete fruit-weight, SSC and pH records. Five outer folds were stratified by cultivar and frozen before fitting. Every model for a target used the same fold manifest and test fruit.

### Nested PLSR, RBF-SVR and PLS2 baselines

PLSR tuning minimised root-mean-square error (RMSE) in four-fold cultivar-stratified inner cross-validation. The search combined four prespecified preprocessing choices with 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 or 24 components. Cultivar-aware PLSR then added the mean training residual for the corresponding cultivar. Radial-basis-function support vector regression (RBF-SVR) standardised predictors and response and used three inner folds. Across all 60 outer folds, C ranged from 0.3 to 300 and gamma multipliers from 0.005 to 4, with a one-axis extension when a selected value met a grid edge. The standardised epsilon grid initially spanned 0.005–0.8 with a hard limit of 1.5. All five pH folds selected that limit, prompting a training-only extension to 10. The 15 texture folds that had also selected epsilon = 1.5 were reselected against the strict superset {2.4, 4, 8, 10}; all retained 1.5. No final C, gamma or epsilon selection lay on a grid boundary. A 12-response partial least-squares model (PLS2) used the common complete-case cohort and selected its component count within outer training.

### Trait-specific residual convolutional model

One residual convolutional neural network (CNN) was trained per target. Four-fold cross-fitted cultivar-aware PLSR predictions defined the residual target for outer-training fruit. The 142,285-parameter encoder used a width-48 stem, four residual blocks, depthwise kernels of 3, 9 and 21, dilations of 1, 2, 4 and 8, GroupNorm, GELU, 0.12 dropout, segment pooling and attention pooling. Twelve epochs of masked 12-trait auxiliary pretraining used outer-training fruit only; fine-tuning retained one scalar output for the target trait.

Fine-tuning used AdamW with learning rate 5 × 10⁻⁴, weight decay 2 × 10⁻³, batch size 256, cosine decay, mixed precision and gradient clipping. Maximum training length was 48 epochs, with a minimum of eight and patience of eight during train-internal selection. Cultivar-balanced sampling used inverse-frequency power 0.5. Raw-space augmentation applied multiplicative scale, offset, slope and wavelength-noise standard deviations of 0.015, 0.004, 0.004 and 0.0015, respectively, after which derived channels were recomputed. The objective combined Smooth-L1 residual loss with cultivar-centred and pairwise-ranking terms weighted 0.20 and 0.08; ranking temperature was 0.30. A cultivar-stratified internal validation set selected the epoch and residual gate from {0, 0.25, 0.50, 0.75, 1.00}; the final model was refitted on the complete outer-training set for the selected epoch count. Fold-level choices and seeds are supplied in Tables S8, S9 and S24.

### Ensemble and negative controls

Kernel eligibility required domain-SVR inner-CV RMSE to be within 5% of domain-PLSR. Every final fold was eligible; PlumSPECTRA therefore averaged residual-CNN and domain-SVR predictions with weights 0.5/0.5. Deep-only and a no-neural 0.5 domain-PLSR/domain-SVR average were retained. The cultivar-mean null predicted each test fruit using labels from the same cultivar in other outer folds only. No test label contributed to its prediction.

### Wavelength association analysis and practical error context

Wavelength evidence came from the 60 primary CNN fits (12 traits × 5 outer folds). For each held-out fold, mean attention-pooling allocation at every wavelength was divided by uniform allocation and fold values were summarised by their median. This measure quantifies where the fitted pooling layer allocated weight. For a model-independent check, spectra were SNV transformed and absorbance–phenotype Pearson correlations were calculated separately by cultivar and wavelength. Cultivar coefficients received equal weight in the median summary, and adjacent wavelengths were aggregated into 50-nm windows. Instrument-edge regions below 920 nm and above 1685 nm were excluded from the non-edge consensus search. Attention and correlation were used as association maps because collinearity prevents compound-specific attribution.

Native-unit context included RMSE, median and 80th-percentile absolute error, ratio of performance to interquartile range (RPIQ), and the fraction of predictions within one-half of the observed interquartile range. For texture outcomes, R² divided by final-cohort absolute-agreement intraclass correlation coefficient (ICC) gives a descriptive measure of performance relative to replicate reliability.

### Robustness, held-batch and leave-one-cultivar-out protocols

Two additional complete-pipeline seeds were fitted for each trait and fold. Predictions were averaged within fruit only after every seed completed. Seed SD, fold SE and cultivar-cluster uncertainty were reported separately.

The held-batch manifest left out KLD_B01, KLD_B02, KLD_B03, WW_B01 or WW_B02 while retaining other batches from the same cultivar in training. Few-shot calibration used 0, 5, 10, 20, 40 or 80 labelled fruit per batch and 500 deterministic matched resamples at each positive budget. Spectral and no-spectra controls received the same calibration fruit, which were removed from evaluation. The regularised affine adapter estimated an intercept and a slope penalised towards one with prespecified strength 5. For Fig. 6d,e, each resample was reduced to the median RMSE gain across nine texture endpoints relative to zero-shot prediction; the plotted line and band are the median and 2.5th–97.5th percentiles across resamples. Continuous two-segment models with candidate knots at 5, 10, 20 and 40 fruit were ranked by residual sum of squares to locate a descriptive elbow within the five observed batches. Leave-one-cultivar-out (LOCO) PLSR held out each cultivar in turn and selected preprocessing and components by macro-normalised inner-cross-validation error across the remaining cultivars.

### Metrics and inference

Primary accuracy was pooled OOF RMSE. Secondary metrics and definitions are listed with Table S3; they include MAE, bias, R², Pearson correlation, Lin concordance correlation (Lin, 1989), RPD and RPIQ. The same metrics were recomputed after centring observations and predictions within cultivar. Fold- and cultivar-level values were retained.

Cultivars were resampled as paired clusters for 1,000,000 draws in the primary contrast analysis. A separate 200,000-draw analysis covered null, deep-only, B50 and PLS2 comparisons. The 36-comparison analysis family and post hoc 12-trait strongest-baseline sensitivities were evaluated with Benjamini–Hochberg, Holm and max-studentised simultaneous procedures. Held-batch intervals used the five batches as descriptive clusters. Random seeds measured optimisation stability and were kept separate from biological replication.

### Visualisation and computation

Figures were drawn in R 4.4.3 with ggplot2 4.0.3 and patchwork 1.3.2, using Arial and a common `theme_classic2`-style system. Main figures were 10.5 inches wide except Fig. 2 (13.125 inches) and were exported as editable-text PDF with 450-dpi PNG counterparts. The minimum 10.5-pt source text reproduces at approximately 7.2 pt at 183 mm. Neural fitting used mixed precision on two NVIDIA RTX 3090 graphics processors. Independent trait–fold–seed jobs were assigned with `CUDA_VISIBLE_DEVICES`; each model occupied one device, and independent fits ran concurrently when memory permitted. Seeds and device assignments were stored in the run metadata.

## Results

### Cohort assembly and identity reconciliation

Identity reconciliation linked 5,502 fruit records. Each usable final-model NIR file contained 228 absorbance bands from 901.032 to 1700.792 nm, and the archive contained 11,004 texture-analyser curves, normally two penetration positions per fruit. Cultivar aliases were standardised to an English registry; the author-confirmed selection A181 is reported as LA191.

Quality control considered spectral signal, acquisition-session conditions, curve acquisition, duplicate reliability and chemical plausibility. The strict release tier retained 4,967 fruit (90.28%); the broader analysis tier retained 5,430 and identified 72 fruit (1.31%) as high-confidence technical exclusions. Cultivar 6.11 was the only cultivar to fail two independent domains, with poor duplicate reliability and an abnormal acquisition session. Distinctive but internally reliable cultivars were retained.

Fruit-level eligibility and the cultivar decision left 15 cultivars, ranging from 86 Fengwei Huanghou to 778 Konglongdan fruit. Ten texture-cohort fruit lacked at least one valid conventional measurement. Outer folds contained 970–971 texture fruit and 968–969 complete cases, with manifests fixed before modelling.

![Figure 1](../results/v26_claudecode_integration/figures_integrated/Figure_1_v26.png)

### Texture endpoint reliability and correlation structure

Each duplicate load–unload pair yielded maximum penetration force (SRF; a legacy code), peak-referenced position (RD), post-peak force drop (PFD), mean flesh force (MFF), force at six archive position units (F6), loading stiffness (LS), loading work (LW), post-peak work (PRW) and adhesive force (AF). The first three principal components of these nine endpoints explained 95.4% of their variance (Table S12). Together they form a compact multidimensional mechanical profile.

Absolute-agreement ICC ranged from 0.801 to 0.984. Median duplicate coefficient of variation (CV) was 3.5–17.2% and highest for F6. Pooled PlumSPECTRA R² reached 59.4–75.2% of the corresponding ICC under a simple attenuation interpretation, leaving both measurement variability and model error in the gap.

The `Motor Steps / mm` marker occurred in every ARC file, but the numeric calibration was absent. Figure 2a therefore reports position in APU, stiffness in N·APU⁻¹ and work in N·APU.

Mechanical profiles differed strongly among cultivar-labelled cohorts. Thirteen of the 15 cultivars came from one identifiable batch, leaving cultivar, batch and operator partly confounded. We use “cultivar-associated” for these differences and examine them again with cultivar-centred, held-batch and unseen-cultivar analyses.

![Figure 2](../results/v26_claudecode_integration/figures_integrated/Figure_2_v26.png)

### Model configuration and training-internal selection

One model was fitted for each target. All 60 outer-fold fits passed the training-internal kernel eligibility rule, so their predictions used the planned equal-weight residual-CNN/RBF-SVR ensemble. The deep-only and no-neural PLSR–SVR rows remained explicit comparators. Recorded preprocessing, hyperparameter, gate and checkpoint decisions matched their training-partition provenance.

![Figure 3](../results/v26_claudecode_integration/figures_integrated/Figure_3_v26.png)

### Baseline search coverage

Thirty-five of the 60 cultivar-aware PLSR folds selected one component. Global PLSR instead selected 8, 12 or 16 components. Neither search reached the 24-component upper bound, and the RBF-SVR selections lay inside the final C, gamma and epsilon ranges. These choices show that the reported comparisons were not set by a truncated hyperparameter grid.

The equal-information PLS2 comparator jointly used the same 12 response labels on the complete-case cohort, isolating the contribution of multi-trait labels from the convolutional representation and ensemble.

### Out-of-fold accuracy and comparison with baselines

PlumSPECTRA RMSE/R² were 12.165 g/0.827 for fruit weight, 1.576 percentage points/0.629 for SSC and 0.361/0.544 for pH. Texture R² ranged from 0.502 for F6 to 0.719 for LS; values for SRF, RD, PFD, MFF, LW, PRW and AF were 0.681, 0.642, 0.645, 0.580, 0.547, 0.583 and 0.561.

Absolute errors place these coefficients in an operational scale without imposing an arbitrary percentage threshold. Median out-of-fold absolute error was 6.64 g for fruit weight, 0.93 percentage points for SSC and 0.25 pH units. Among texture endpoints it was 0.98 N for maximum penetration force, 0.93 N for post-event force drop and 0.38 N·APU⁻¹ for loading stiffness. Depending on endpoint, 70.8–83.8% of texture predictions fell within one-half of the observed interquartile range and RPIQ was 1.91–2.72. Predictive R² represented 59.4–75.2% of duplicate-measurement ICC, so loading stiffness, maximum penetration force and post-event force drop were the clearest texture candidates for calibrated screening, whereas F6 remained the least precise. Complete native-unit errors and reliability-normalised context are reported in Supplementary Table S41.

Against ordinary global PLSR, relative RMSE reduction ranged from 12.8% for SSC to 51.5% for fruit weight, with 12/12 max-studentised simultaneous intervals above zero. Global PLSR is a deliberately plain reference: in 11 of 12 traits it was less accurate than the spectrum-free cultivar-mean predictor. The large percentage range mainly captures the value of cultivar-aware nonlinear modelling over a global linear model, rather than the contribution of spectral measurements alone.

The more demanding post hoc comparison excluded both individual PlumSPECTRA branches and selected the lowest pooled RMSE among global PLSR, cultivar-aware PLSR and no-neural B50 for each trait. The final model improved RMSE by 0.86% for pH to 4.01% for LS, and all 12 simultaneous intervals remained above zero. No-neural B50 was selected for 11 traits and cultivar-aware PLSR for fruit weight. A separate branch-inclusive sensitivity gave the smaller 0.5–3.2% range and 8/12 simultaneous intervals. This distinction keeps component contrasts separate from comparisons with branch-excluded baselines.

Across the 36 primary contrasts, cultivar-cluster resampling supported 34 after Benjamini–Hochberg or Holm adjustment and 26 with max-studentised simultaneous intervals. Baseline-specific simultaneous families supported 12/12 contrasts against global PLSR, 10/12 against cultivar-aware PLSR and 8/12 against the nested RBF-SVR branch. Both strongest-baseline analyses were post hoc and rest on 15 cultivar clusters.

![Figure 4](../results/v26_claudecode_integration/figures_integrated/Figure_4_v26.png)

### Optimisation stability across complete-pipeline seeds

Two additional complete training runs for each trait and fold brought the total to 180 pipeline instances. Every repeat reran epoch and gate selection, yielding three predictions per OOF fruit–trait pair.

The three-seed mean improved on cultivar-aware PLSR in all 60 trait–fold comparisons. Cultivar-cluster RMSE reduction ranged from 1.81% for fruit weight (95% CI 1.16–2.75%) to 5.71% for loading stiffness (3.36–7.87%), with positive lower limits for all 12 traits. Mean within-fold seed SD was 0.27–1.48 percentage points. Averaging reduced foldwise SE for most high-variance targets, while outer-fold heterogeneity remained larger than optimisation noise.

### Cultivar-centred performance and the cultivar-mean null

Cultivar-centred PlumSPECTRA R² was positive but lower than pooled R² for every trait, from 0.018 for pH to 0.378 for SSC. Texture-centred R² ranged from 0.021 for F6 to 0.200 for RD. Fruit-level signal remained, but cultivar-associated means contributed substantially to pooled discrimination.

The difference between PlumSPECTRA and cultivar-mean-null pooled R² was 0.227 for SSC, 0.089 for RD and 0.059 for AF, but only 0.010–0.013 for pH, fruit weight and F6; nine differences were below 0.05. These are predictive contrasts between non-nested models. At cultivar–endpoint resolution, 53 of 180 centred R² values were negative and nine also had a negative correlation with the observations. The five largest and five smallest cultivars each contributed 16 failures, so sample count alone did not explain the pattern.

![Figure 5](../results/v26_claudecode_integration/figures_integrated/Figure_5_v26.png)

### Wavelength association analysis

Wavelength evidence was computed from the 60 primary CNN fits. The most emphasised wavelength for each trait received 1.05–1.14 times uniform attention-pooling allocation, and every maximum occurred at an instrument endpoint. These weak edge maxima are most useful as sensitivity warnings.

The cultivar-resolved SNV analysis was similarly diffuse, with trait-wise maximum median absolute correlations of 0.14–0.28 (pH 0.14; RD 0.28). Recurrent non-edge associations occurred in 1250–1350 and 1550–1600 nm windows, especially for peak-referenced position, loading stiffness and adhesive force; SSC also appeared in the latter window. These bands overlap broad O–H and C–H overtone and combination structure, but correlated chemistry and scattering preclude a compound-specific reading (Nicolaï et al., 2007; Walsh et al., 2020; Uwadaira et al., 2018). The attention and correlation maps together support a distributed spectral representation (Supplementary Fig. S25).

### Held-batch and unseen-cultivar transfer

Only Konglongdan and Weiwang contained multiple identifiable batches. Five leave-one-batch-out tests covered 1,256 unique fruit and 11,304 fruit–trait predictions. Baselines and PlumSPECTRA were refitted without the held batch, and branch eligibility remained training internal.

Direct held-batch pooled R² was negative for every texture endpoint (−0.256 to −0.092). On batch-macro RMSE, PlumSPECTRA improved on cultivar-aware PLSR and domain-aware RBF-SVR for 9/9 endpoints, but on global PLSR for only 4/9; global PLSR was better for AF, F6, LW, MFF and PRW. Cultivar offsets helped interpolation. They did not survive every batch shift. These results cover five batches within two cultivars.

Regularised affine updating reduced median RMSE by 9.7%, 13.5%, 15.4%, 16.5% and 17.0% at 5, 10, 20, 40 and 80 reference fruit. The two-segment fit placed its descriptive elbow at 10 fruit, while the 40-to-80 increment was 0.48 percentage points. Within these five batches, 10 fruit captured the steep early return and 20–40 fruit provided a more stable operating window.

At 40 fruit with intercept updating, median pooled R² was 0.187 for PlumSPECTRA and 0.164 for the batch-mean control; RD and SRF still favoured the control under pooled and batch-macro RMSE. Only shrunken-affine LW met the strict dual lower-bound criterion, at 80 fruit. Eleven of 18 trait–adapter combinations met a weaker directional criterion at some budget, whereas seven met neither criterion (Table S30). The intervals describe calibration-fruit selection within the observed batches.

In the separate zero-shot LOCO PLSR test, pooled R² ranged from −0.183 to 0.217 and cultivar-macro R² was −4.761 to −1.525 for all nine texture endpoints. Of 135 cultivar–endpoint cells, 124 were negative; adhesive force in Naili was the only usable transfer (R² = 0.42). The fitted calibration domain did not extend to unseen cultivars.

![Figure 6](../results/v26_claudecode_integration/figures_integrated/Figure_6_v26.png)

## Discussion

Intact-fruit spectra were paired with duplicate mechanical measurements and conventional quality assays at a scale rarely available for plum. The nine-endpoint panel treats texture as a primary phenotype and preserves more of the load–unload curve than a single firmness value.

Ordinary global PLSR produced the largest contrast, but it also lost to the spectrum-free cultivar-mean null on 11 traits. The 12.8–51.5% reduction should therefore be read as the combined benefit of cultivar-aware and nonlinear modelling over a deliberately plain chemometric reference. The branch-excluded comparison is more demanding: PlumSPECTRA reduced RMSE by 0.9–4.0% over the best of global PLSR, cultivar-aware PLSR and no-neural B50, with positive simultaneous lower limits for all traits. Both PLSR variants could select 1–24 components, and the kernel search ended away from its boundaries. Comparator capacity, rather than a restricted search, explains the contraction in effect size.

The residual CNN alone had lower pooled RMSE for FW, F6, LS and LW; the equal-weight ensemble led on the other eight targets. Ensemble and CNN intervals overlapped throughout. Their error correlations of 0.926–0.981 explain this behaviour: under an equal-error-variance approximation, equal weighting can lower RMSE by only 0.5–1.9% at those correlations. For the four CNN-favoured targets, fixed averaging is a robustness choice rather than an accuracy choice. Compact residual learning is the main source of the nonlinear gain; ensembling adds a modest stabilising layer.

Several design choices limit artefactual advantage. Preprocessing, hyperparameters, gates and checkpoints were selected from outer-training fruit. The 12-response PLS2 control received the same auxiliary labels as the network, while deep-only and no-neural rows separate convolution from averaging. Cultivar-mean and batch-mean controls quantify group identity, and complete-pipeline seed repeats separate optimisation noise from fold-to-fold biological variation. Contrasts against nested RBF-SVR and the residual CNN are reported as component analyses; the headline strongest-baseline family excludes both individual branches.

Plum softening accompanies changes in cell-wall integrity and pectin organisation (Geng et al., 2020), while direct firmness tracks ripening stage and consumer acceptance in plum and related stone fruit (Crisosto et al., 2004; Valero et al., 2007; Usenik et al., 2014). The present endpoints partition this mechanical phenotype beyond peak force. Loading stiffness was the most predictable texture measure (R² = 0.719; RPIQ = 2.72), followed by maximum penetration force and post-peak force drop (R² = 0.681 and 0.645). Their median absolute errors were 0.38 N·APU⁻¹, 0.98 N and 0.93 N. These values suit cohort screening and ranking; sensory and storage experiments are still needed to connect the mechanical profile to eating quality and shelf life.

Attention did not concentrate on a narrow interior band. Equal-cultivar correlations instead revealed broad associations around 1250–1350 and 1550–1600 nm for selected texture endpoints and SSC. Water status, soluble constituents, tissue scattering and cell-wall change all covary during ripening, consistent with evidence that optical firmness prediction need not arise from one constituent (Uwadaira et al., 2018; Geng et al., 2020). Agreement between the model-based and cultivar-resolved maps supports distributed spectral phenotyping, while the edge maxima caution against biochemical attribution.

Cultivar and batch are confounded for 13 of 15 cultivars, and the held-batch audit contains five batches from only two cultivars. No independent year, orchard or instrument was available. The archive lacks the NIR instrument model, optical geometry, fruit position, reference schedule and complete protocols for conventional assays; physical texture-position calibration is also missing. Direct sensory quality, consumer liking and shelf life were not measured. Cultivar-dependent shelf-life trajectories require longitudinal postharvest measurements (Guo et al., 2022; Bohinc et al., 2026), so the present mechanical predictions should be tested prospectively against those outcomes.

The archive headers imply 30.5 ms of detector integration for six declared repeats, but they do not record positioning, referencing, file transfer or operator time. End-to-end throughput cannot be reconstructed reliably. In practice, the model is best positioned as an upstream screen: most fruit receive NIR measurements, while a smaller reference set undergoes conventional and texture assays for batch monitoring. Ten labelled fruit captured the steepest early recovery in the held-batch audit, and 20–40 fruit formed the more stable range. This calibration burden is modest relative to measuring every fruit, but it must be budgeted for each new batch and cultivar. A prospective engineering trial should measure scan cycle time, reference-labour cost and drift across years, orchards, operators and instruments.

## Conclusions

PlumSPECTRA predicts fruit weight, SSC, pH and nine mechanical texture endpoints from a 228-band intact-fruit spectrum. Within the 15-cultivar calibration domain, texture R² reached 0.502–0.719. The 12.8–51.5% RMSE reduction over global PLSR partly reflects the weakness of that reference, whereas the branch-excluded post hoc comparison gave a more credible 0.9–4.0% margin with positive simultaneous lower limits for all traits. Loading stiffness, maximum penetration force and post-peak force drop combined the strongest accuracy and repeatability. Transfer required calibration: 20–40 labelled reference fruit captured most of the observed within-cultivar batch benefit, and zero-shot prediction did not extend to unseen cultivars. The study provides a calibration-aware screening framework and a large multidimensional texture resource for prospective validation.

## Data availability

Final OOF predictions, split manifests, cultivar and endpoint registries, fold-level training metadata, figure data, analysis code and audit outputs are supplied in the accompanying reproducibility package. A public archival record and DOI will replace the anonymous review location at publication. Raw texture-analyser and spectral files are available from the corresponding author subject to institutional governance and file-volume constraints. The analysis seed was 20260806.

## Acknowledgments

Funding and acknowledgements are withheld for anonymous review and must be restored before submission.

## Funding

Funding information and grant identifiers are withheld for anonymous review and must be restored before submission.

## CRediT authorship contribution statement

The CRediT statement is withheld for anonymous review and must be restored before submission.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the writing process

During manuscript preparation, the authors used OpenAI Codex and Anthropic Claude Code for language editing, code review, numerical cross-checks and document formatting. The authors reviewed the analyses, code changes, numerical results and written claims and take responsibility for the article.

## Supplementary information

Supplementary Figures S1–S25 and Tables S1–S42 accompany this article, together with frozen split manifests, machine-readable figure data and reproducibility identifiers.

## Figure legends

### Figure 1. Paired single-fruit acquisition workflow, cohort ledger and per-cultivar composition

**(a)** Measurement sequence. The 228-band near-infrared (NIR) spectrum (901–1,701 nm) and fruit mass were acquired before the dashed destructive boundary; duplicate penetration tests, soluble solids content (SSC) and pH followed on the same fruit. Texture served as a reference phenotype alongside SSC and pH. **(b)** Cohort ledger: 5,502 source fruit, 5,430 in the analysis tier, 4,967 in the strict release tier, 4,853 in the texture cohort and 4,843 complete cases. The 72 technical exclusions used measurement-quality evidence; the final ten fruit lacked a valid mass, SSC or pH value. The modelling cohorts generated 58,206 OOF fruit–trait predictions from 60 baseline folds, 60 primary neural folds and 120 seed refits. **(c)** Retained fruit per cultivar and identifiable acquisition batches. Konglongdan and Weiwang were the only cultivars with replicate batches, yielding five held-batch clusters. Codes: KLD, Konglongdan; WD, Weidi; WW, Weiwang; QCL, Qingcuili; WJ, Weijin; CHL, Cuihongli; FRL, Furongli; ZSKLD, Zaoshu Konglongdan; WX, Weixin; FTL, Fengtangli; NL, Naili; FWHH, Fengwei Huanghou. LA191 (registry name A181), L313 (registry name 3.13) and L31 are breeding selections. Panel a is schematic.

Alt text: A three-part figure describing how the dataset was built. The top panel is a horizontal timeline of the six measurements made on each individual fruit, from left to right: NIR spectrum, fruit mass, two penetration tests, soluble solids content and pH. A heavy dashed vertical line sits between fruit mass and the first penetration test and is labelled the destructive boundary; the spectrum and the mass are the two measurements to its left, taken while the fruit is intact. A bracket above the whole sequence states that 4,853 fruit were measured end to end, one fruit per data row. The lower left panel is a horizontal bar chart of the cohort sizes, descending from 5,502 fruit in the source ledger to 4,843 in the complete-case cohort. The lower right panel is a horizontal bar chart of retained fruit for each of fifteen cultivars, ranging from 778 down to 86, with a narrow column of batch counts showing that thirteen cultivars came from a single acquisition batch.

### Figure 2. Phenotype distributions, endpoint correlation structure, measurement reliability and the fruit-level texture landscape

Observed phenotypes only. **(a)** Cultivar-resolved distributions of all 12 traits. Points represent fruit and white boxes show the median, interquartile range and 1.5-IQR whiskers. APU denotes archive position unit; RD is reported in APU, LS in N·APU⁻¹, and LW/PRW in N·APU. Endpoint codes are compatibility labels: SRF, maximum penetration force; RD, peak-referenced position; PFD, post-peak force drop; MFF, mean flesh force; F6, force at 6 APU; LS, loading stiffness; LW, loading work; PRW, post-peak work; AF, adhesive force. **(b)** Pearson correlations among 12 traits, clustered on 1 − |r|; white dots mark |r| > 0.90. **(c)** Loadings of the first three principal components of all 12 standardised traits. They explain 60.4%, 12.1% and 10.5% (82.9% cumulative); the participation ratio is 2.51. **(d)** Absolute-agreement ICC(A,1) and Pearson replicate correlation for the two penetration positions. Markers coincide where the measures differ by less than 0.002 (SRF, RD, PFD, LS and AF). The duplicate cohort contains 4,853 fruit for eight endpoints and 4,851 for F6. **(e)** Scores and cultivar centroids from the 12-trait PCA, with whole-cohort density contours. The display window retains 4,565 fruit (94.3%); all fruit remain in the calculations. Grey leaders join labels to centroids.

Alt text: A five-part figure describing the phenotype data before any modelling. Panel a is one three-row by four-column grid of all twelve traits. Each facet contains thousands of semi-transparent, cultivar-coloured single-fruit points with a white boxplot for each of fifteen cultivar codes; all facets have x- and y-axis lines, but smaller 45-degree cultivar labels appear only on the bottom row. Panels b through e occupy a 20%-taller row beneath it and include FW, SSC, pH and nine texture endpoints. Panel b is a twelve-by-twelve correlation heatmap showing a strongly correlated texture block and its associations with conventional quality. Panel c contains three lollipop plots with dashed zero-reference lines for all twelve traits; the first component is mainly mechanical while fruit weight and soluble solids load strongly on the second. Panel d shows the nine texture endpoints with joined intraclass-correlation and replicate-correlation markers; FW, SSC and pH are omitted because no duplicate measurements exist. Panel e presents a cropped central window of the joint PCA map, with 4,565 fruit visible as pale points, light-plum whole-cohort density contours and fifteen enlarged labelled cultivar centroids; all 4,843 complete fruit remain included in the PCA, centroids and density calculations, and no cultivar confidence ellipse is drawn.

### Figure 3. Residual model architecture, nested cross-validation protocol and equal-information comparator design

A schematic of the analysis protocol. **(a)** The 228-band spectrum enters as raw, standard-normal-variate (SNV) and first-derivative Savitzky–Golay (SG1) views. A cultivar-aware PLSR anchor supplies cross-fitted residual targets. Twelve-trait auxiliary pretraining uses outer-training fruit, after which a residual CNN and nested RBF-SVR pass through a training-internal gate into a 0.5/0.5 ensemble. One model is fitted per trait. **(b)** Five cultivar-stratified outer folds; the highlighted cell is held out. The lists identify training-internal selections and prohibited information channels. **(c)** Eight models compared across four capabilities. Deep-only is the residual CNN; no-neural B50 averages cultivar-aware PLSR and nested RBF-SVR; the cultivar-mean null uses no spectrum. The 12-response PLS2 control receives the same auxiliary labels and isolates multi-trait information from convolutional representation. Nested RBF-SVR and residual CNN are PlumSPECTRA branches.

Alt text: A three-part schematic of the modelling protocol containing no measured data. The top panel is a left-to-right flow diagram: the 228-band spectrum feeds both a cultivar-aware PLSR anchor and a twelve-trait auxiliary pretraining stage, both of which feed a residual spectral convolutional network; that network and a separately tuned kernel regressor pass through an eligibility gate into a fixed half-and-half ensemble producing twelve outputs. Small teal squares on several boxes mark steps configured using training data only. The lower left panel shows a five-by-five grid representing five cross-validation folds with one held-out cell per row, followed by two ruled lists: what is selected inside each training block, and three crossed-out information channels the design forbids. The lower right panel is a table of eight comparator models against four capabilities — sees twelve traits, sees cultivar, nonlinear, ensemble — with filled circles where a model has the capability and open circles where it does not, a right-hand column naming each model's role, and the equal-information PLS2 control and the final model highlighted.

### Figure 4. Out-of-fold accuracy, improvement over ordinary PLSR and sensitivity to stronger comparators

**(a)** OOF agreement for 58,206 fruit–trait pairs. Shading gives fruit per bin on a log scale; the dashed line is identity. Axes are clipped to the 0.5th–99.5th percentiles, excluding 644 points from bins but not statistics. **(b)** Relative RMSE reduction against global PLSR (left) and five stronger comparators (right). Points and bars are estimates and 95% cultivar-cluster intervals; filled symbols exclude zero. Arrows mark effects above the 8% display limit. Nested RBF-SVR and residual CNN are component contrasts. **(c)** Supported contrasts under raw, Benjamini–Hochberg, Holm and max-studentised simultaneous criteria. The bottom row compares the final model with the strongest branch-excluded baseline per trait. All families are analysis-stage or post hoc and use 15 cultivar clusters. **(d)** Trait-wise comparison with the lower-RMSE model among global PLSR, cultivar-aware PLSR and no-neural B50. All 12 simultaneous intervals exclude zero; gains are 0.86–4.01%.

Alt text: Four panels summarise prediction accuracy. Twelve density plots compare observed and predicted values with R-squared from 0.50 to 0.83. Forest plots show the large contrast with global PLSR and smaller contrasts with stronger baselines and model branches. A five-row multiplicity matrix reports supported counts; the branch-excluded bottom row is 12 of 12 under every correction. The final forest plot shows positive simultaneous intervals for all traits against the strongest branch-excluded baseline, selected from global PLSR, cultivar-aware PLSR and no-neural B50.

### Figure 5. Performance relative to the cultivar-mean null, within-cultivar signal and component complementarity

**(a)** Pooled OOF R² of the cultivar-mean null (grey) and the PlumSPECTRA difference (purple). The +0.010 to +0.227 contrast compares non-nested predictors. **(b)** Within-cultivar centred R² for cultivar-aware PLSR, nested RBF-SVR, no-neural B50, residual CNN and PlumSPECTRA. Global PLSR is omitted because it reaches −0.58. **(c)** Within-cultivar R² for 180 cultivar–endpoint cells. Diagonals mark 53 negative cells, boxes identify the nine that also have negative correlation, and white dots mark values outside the colour scale. **(d)** CNN–SVR error correlation (filled) and opposite-sign error fraction (open). The dashed 0.5 line applies to open symbols only and is the expected opposite-sign fraction for independent predictors.

Alt text: Four panels separate pooled and within-cultivar performance. A stacked bar chart compares PlumSPECTRA with the cultivar-mean null. The adjacent plot shows five models, including no-neural B50, after cultivar means are removed. A twelve-by-fifteen heatmap contains 53 negative cells, nine of which also have negative correlation. The final panel shows CNN–SVR error correlations of 0.926–0.981 and opposite-sign errors for 5.3–13.5% of fruit.

### Figure 6. Predictive performance across same-session, held-batch and unseen-cultivar evaluation regimes

**(a)** Pooled R² for nine texture endpoints. Lines connect same-session and held-batch PlumSPECTRA results; open, unconnected points show the separate unseen-cultivar LOCO PLSR test. The dashed horizontal line is zero. **(b)** LOCO PLSR R² for 135 held-out cultivar–endpoint cells. White dots mark 28 cells below −3; a dark open circle marks the only cell above 0.3 (AF in Naili, R² = 0.42). **(c)** Held-batch macro RMSE change against three baselines. Points and bars give descriptive 95% intervals over five batch clusters; filled symbols exclude zero. **(d)** Median RMSE reduction after affine updating with 0–80 labelled reference fruit; the ribbon is the 2.5th–97.5th percentile range across 500 matched resamples, and the shaded region marks the 20–40-fruit operating window. **(e)** Marginal gain per ten additional fruit. The descriptive two-segment fit placed the elbow at 10 fruit; moving from 40 to 80 added 0.48 percentage points. Panels c–e cover five batches from two cultivars.

Alt text: Five panels compare evaluation regimes and calibration effort. Same-session and held-batch PlumSPECTRA values are connected, while the separate LOCO PLSR points are open and unconnected. The LOCO heatmap has 124 negative cells among 135 and marks both low and high off-scale values. A forest plot gives held-batch contrasts. Two calibration plots show a steep early gain, a resampling interval, a 20–40-fruit operating window and diminishing marginal return after ten fruit.

## References

Bohinc, K., Trebar, M., Tomić, J., Glišić, I., Pešaković, M., Štukelj, R., Abram, A., Van de Velde, N.W., Jerman, I., Vidrih, R., 2026. Cultivar differences in postharvest quality of plums: Changes in metabolic and cuticle biophysical properties during shelf life. *Scientia Horticulturae* 360, 114793. https://doi.org/10.1016/j.scienta.2026.114793.

Bu, Y., Luo, J., Tian, Q., Li, J., Cao, M., Yang, S., Guo, W., 2025. Nondestructive detection of internal quality in multiple peach varieties by Vis/NIR spectroscopy with multi-task CNN method. *Postharvest Biology and Technology* 227, 113579. https://doi.org/10.1016/j.postharvbio.2025.113579.

Crisosto, C.H., Garner, D., Crisosto, G.M., Bowerman, E., 2004. Increasing ‘Blackamber’ plum (*Prunus salicina* Lindell) consumer acceptance. *Postharvest Biology and Technology* 34, 237–244. https://doi.org/10.1016/j.postharvbio.2004.06.003.

Fu, X., Ying, Y., Zhou, Y., Xie, L., Xu, H., 2008. Application of NIR spectroscopy for firmness evaluation of peaches. *Journal of Zhejiang University Science B* 9, 552–557. https://doi.org/10.1631/jzus.B0720018.

Geng, Y., Zhang, Y., Liu, Y., Hu, B., Wang, J., He, J., Liang, M., 2020. Quality attributes and microstructure of cell walls in ‘Suli’ plum fruit (*Prunus salicina* Lindl.) during softening. *Food Science and Technology Research* 26, 281–292. https://doi.org/10.3136/fstr.26.281.

Guo, H., Yan, F., Li, P., Li, M., 2022. Determination of storage period of harvested plums by near-infrared spectroscopy and quality attributes. *Journal of Food Processing and Preservation* 46, e16504. https://doi.org/10.1111/jfpp.16504.

Li, X., Li, Z., Yang, X., He, Y., 2021. Boosting the generalization ability of Vis-NIR-spectroscopy-based regression models through dimension reduction and transfer learning. *Computers and Electronics in Agriculture* 186, 106157. https://doi.org/10.1016/j.compag.2021.106157.

Lin, L.I.-K., 1989. A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 45, 255–268. https://doi.org/10.2307/2532051.

Liu, S., Zhao, X., Zhu, Q., Huang, M., Guo, X., 2025. A feature-enhanced approach based on joint domain alignment and multi-order derivative spectral reconstruction for predicting apple firmness using Vis-NIR spectroscopy. *Food Chemistry* 476, 143457. https://doi.org/10.1016/j.foodchem.2025.143457.

Louw, E.D., Theron, K.I., 2010. Robust prediction models for quality parameters in Japanese plums (*Prunus salicina* L.) using NIR spectroscopy. *Postharvest Biology and Technology* 58, 176–184. https://doi.org/10.1016/j.postharvbio.2010.07.001.

Lu, R., Guyer, D.E., Beaudry, R.M., 2000. Determination of firmness and sugar content of apples using near-infrared diffuse reflectance. *Journal of Texture Studies* 31, 615–630. https://doi.org/10.1111/j.1745-4603.2000.tb01024.x.

Malek, S., Melgani, F., Bazi, Y., 2018. One-dimensional convolutional neural networks for spectroscopic signal regression. *Journal of Chemometrics* 32, e2977. https://doi.org/10.1002/cem.2977.

Masuda, K., Uchida, R., Fujita, N., Miyamoto, Y., Yasue, T., Kubo, Y., Ushijima, K., Uchida, S., Akagi, T., 2023. Application of deep learning diagnosis for multiple traits sorting in peach fruit. *Postharvest Biology and Technology* 201, 112348. https://doi.org/10.1016/j.postharvbio.2023.112348.

Minas, I.S., Anthony, B.M., Pieper, J.R., Sterle, D.G., 2023. Large-scale and accurate non-destructive visual to near infrared spectroscopy-based assessment of the effect of rootstock on peach fruit internal quality. *European Journal of Agronomy* 143, 126706. https://doi.org/10.1016/j.eja.2022.126706.

Nicolaï, B.M., Beullens, K., Bobelyn, E., Peirs, A., Saeys, W., Theron, K.I., Lammertyn, J., 2007. Nondestructive measurement of fruit and vegetable quality by means of NIR spectroscopy: A review. *Postharvest Biology and Technology* 46, 99–118. https://doi.org/10.1016/j.postharvbio.2007.06.024.

Paz, P., Sánchez, M.-T., Pérez-Marín, D., Guerrero, J.-E., Garrido-Varo, A., 2008. Nondestructive determination of total soluble solid content and firmness in plums using near-infrared reflectance spectroscopy. *Journal of Agricultural and Food Chemistry* 56, 2565–2570. https://doi.org/10.1021/jf073369h.

Pérez-Marín, D., Paz, P., Guerrero, J.-E., Garrido-Varo, A., Sánchez, M.-T., 2010. Miniature handheld NIR sensor for the on-site non-destructive assessment of post-harvest quality and refrigerated storage behavior in plums. *Journal of Food Engineering* 99, 294–302. https://doi.org/10.1016/j.jfoodeng.2010.03.002.

Posom, J., Klaprachan, J., Rattanasopa, K., Sirisomboon, P., Saengprachatanarug, K., Wongpichet, S., 2020. Predicting Marian plum fruit quality without environmental condition impact by handheld visible–near-infrared spectroscopy. *ACS Omega* 5, 27909–27921. https://doi.org/10.1021/acsomega.0c03203.

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schröder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F., Dormann, C.F., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40, 913–929. https://doi.org/10.1111/ecog.02881.

Savitzky, A., Golay, M.J.E., 1964. Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry* 36, 1627–1639. https://doi.org/10.1021/ac60214a047.

Scalisi, A., O’Connell, M.G., 2021. Application of visible/NIR spectroscopy for the estimation of soluble solids, dry matter and flesh firmness in stone fruits. *Journal of the Science of Food and Agriculture* 101, 2100–2107. https://doi.org/10.1002/jsfa.10832.

Usenik, V., Stampar, F., Kastelec, D., 2014. Indicators of plum maturity: When do plums become tasty? *Scientia Horticulturae* 167, 127–134. https://doi.org/10.1016/j.scienta.2014.01.002.

Uwadaira, Y., Sekiyama, Y., Ikehata, A., 2018. An examination of the principle of non-destructive flesh firmness measurement of peach fruit by using VIS-NIR spectroscopy. *Heliyon* 4, e00531. https://doi.org/10.1016/j.heliyon.2018.e00531.

Valero, C., Crisosto, C.H., Slaughter, D., 2007. Relationship between nondestructive firmness measurements and commercially important ripening fruit stages for peaches, nectarines and plums. *Postharvest Biology and Technology* 44, 248–253. https://doi.org/10.1016/j.postharvbio.2006.12.014.

Walsh, K.B., McGlone, V.A., Han, D.H., 2020. The uses of near infrared spectroscopy in postharvest decision support: A review. *Postharvest Biology and Technology* 163, 111139. https://doi.org/10.1016/j.postharvbio.2020.111139.

Wang, J., Yang, Y., Li, S., Zeng, S., Chi, Q., Guo, W., 2024. Calibration transfer of cross soluble solids content of different kiwifruit cultivars based on two-stage TrAdaBoost.R2. *Postharvest Biology and Technology* 210, 112783. https://doi.org/10.1016/j.postharvbio.2024.112783.

Wu, X., Li, G., Fu, X., Wu, W., 2023. Robustness of calibration model for prediction of lignin content in different batches of snow pears based on NIR spectroscopy. *Frontiers in Plant Science* 14, 1128993. https://doi.org/10.3389/fpls.2023.1128993.

Yang, J., Wang, J., Lu, G., Fei, S., Yan, T., Zhang, C., Lu, X., Yu, Z., Li, W., Tang, X., 2021. TeaNet: Deep learning on near-infrared spectroscopy data for the assurance of tea quality. *Computers and Electronics in Agriculture* 190, 106431. https://doi.org/10.1016/j.compag.2021.106431.

Zhang, W., Sanaeifar, A., Ji, X., Luo, X., Guo, H., He, Q., Luo, Y., Huang, F., Yan, P., Li, X., He, Y., 2024. Data-driven optimization of nitrogen fertilization and quality sensing across tea bud varieties using near-infrared spectroscopy and deep learning. *Computers and Electronics in Agriculture* 222, 109071. https://doi.org/10.1016/j.compag.2024.109071.

Zhou, C., Zhang, X., Liu, Y., Ni, X., Wang, H., Liu, Y., 2023. Research on hyperspectral regression method of soluble solids in green plum based on one-dimensional deep convolution network. *Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy* 303, 123151. https://doi.org/10.1016/j.saa.2023.123151.
