
# PlumSPECTRA

Public reproducibility repository for a cultivar-aware near-infrared framework that
predicts fruit weight, soluble solids content, pH and nine mechanical texture
endpoints from intact plums.

> **Release status:** public `1.0.0-rc.1`. The version DOI, reuse licenses and final
> deployment weights are not yet claimed as complete.

## Study design and retained cohort

![Figure 1. Study design and retained cohort](docs/Figure_1.png)

## What is frozen

- 5,502 identity-linked fruit in the source ledger.
- 4,853 quality-controlled fruit from 15 cultivars for texture modelling.
- 12 independently fitted trait targets and five cultivar-stratified outer folds.
- 58,206 out-of-fold fruit-trait predictions.
- Final R²: 0.827 for fruit weight, 0.629 for SSC, 0.544 for pH and 0.502–0.719 for texture.
- Branch-excluded RMSE gains of 0.86–4.01% over the strongest independent baseline;
  all 12 simultaneous intervals exclude zero.

## Evidence boundary

The primary result is interpolation inside the registered 15-cultivar acquisition
domain. It is not external orchard, year, instrument or unseen-cultivar validation.
Five held batches from two cultivars support labelled adaptation only; zero-shot
leave-one-cultivar-out transfer failed for every texture endpoint.

## Model architecture and leakage controls

![Figure 3. PlumSPECTRA architecture and leakage controls](docs/Figure_3.png)

## Repository map

- `analysis/`: analysis, training and audit scripts retained for provenance.
- `configs/`: cultivar, trait and model registries.
- `docs/Figure_1.png`: authorised study-design and cohort figure.
- `docs/Figure_3.png`: authorised model-architecture figure.
- `evidence/`: frozen predictions, metric tables and audit outputs.
- `tools/audit_release.py`: repository integrity and privacy audit.
- `tests/`: clean-environment checks for the frozen evidence.

Manuscript files, all other publication figures and figure-specific source data are
intentionally excluded. Figures 1 and 3 are the only manuscript-derived images
authorised for this repository.

Related public repositories:

- [`PlumSPECTRA-data`](https://github.com/707728642li/PlumSPECTRA-data): analysis-ready spectra, phenotypes, folds and prediction records.
- [`PlumSPECTRA-models`](https://github.com/707728642li/PlumSPECTRA-models): model card, configuration registry and deployment-weight release status.

## Reviewer quick check

```bash
python -m pip install pandas pyarrow pytest pyyaml
python tools/audit_release.py
pytest -q
```

The check recomputes file hashes, cohort sizes and selected headline metrics. It does
not retrain the neural networks. Full retraining requires the public analysis-ready
dataset and substantial GPU time.

## Data, model and code availability

The public data repository contains a path-free analysis table and the frozen OOF
predictions. Raw ARC archives and instrument exports are not copied into Git history.
The model repository does not mislabel the older nine-trait production bundle as the
paper's final 12-trait system; final refit weights remain a release blocker.

## Citation and license

The version DOI and public reuse licenses will be added after institutional
confirmation. Repository access is public, but reuse rights remain reserved until
the corresponding license files are issued.
