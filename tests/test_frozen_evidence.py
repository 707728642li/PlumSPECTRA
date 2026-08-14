
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def test_prediction_contract():
    p=pd.read_parquet(ROOT/'evidence/final_analysis/v25_integrated_predictions.parquet')
    assert len(p)==58206
    assert set(p.trait.unique())=={'FW','SSC','pH','SRF','RD','PFD','MFF','F6','LS','LW','PRW','AF'}
    assert p.y_final.notna().all()
    assert p.groupby('trait').outer_fold.nunique().eq(5).all()

def test_branch_excluded_family():
    x=pd.read_csv(ROOT/'evidence/final_revision/multiplicity_branch_excluded_strongest_family.csv')
    assert len(x)==12
    assert x.relative_rmse_improvement_pct.between(0.86,4.02).all()
    assert x.supported_simultaneous_0_05.all()
