
# Reproducibility contract

The sample, not the spectrum or texture replicate, is the observational unit. Outer
folds are cultivar-stratified and frozen. Test fruit cannot contribute to scaling,
preprocessing, feature selection, hyperparameter selection or early stopping.
Comparisons use identical held-out fruit. Primary uncertainty resamples cultivars as
clusters. The branch-excluded strongest-baseline family excludes the residual CNN and
nested RBF-SVR because both are components of the final ensemble.

The repository supports exact recomputation of reported metrics from frozen OOF
predictions. It does not claim that frozen predictions substitute for an external
orchard or instrument validation.
