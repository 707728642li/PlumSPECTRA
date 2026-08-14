
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "release/repository_manifest.json").read_text(encoding="utf-8"))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
bad=[rel for rel,meta in manifest["files"].items() if not (ROOT/rel).exists() or (ROOT/rel).stat().st_size!=meta["bytes"] or sha(ROOT/rel)!=meta["sha256"]]
pred=pd.read_parquet(ROOT / "evidence/final_analysis/v25_integrated_predictions.parquet")
assert len(pred)==58206 and pred["trait"].nunique()==12 and pred["outer_fold"].nunique()==5
pooled=pd.read_csv(ROOT / "evidence/final_analysis/pooled_metrics.csv")
final=pooled[pooled["model"].eq("plumspectra_corrected")].set_index("trait")
assert abs(float(final.loc["FW","r2"])-0.827457)<5e-6
assert abs(float(final.loc["SSC","r2"])-0.628828)<5e-6
assert abs(float(final.loc["pH","r2"])-0.543648)<5e-6
assert not bad, bad
print(json.dumps({"status":"PASS","manifest_files":len(manifest["files"]),"prediction_rows":len(pred),"traits":pred["trait"].nunique()},indent=2))
