from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def cultivar_registry() -> pd.DataFrame:
    """Return the authoritative V2 cultivar display-code registry."""
    table = pd.read_csv(CONFIG_DIR / "v2_nomenclature.csv", dtype=str)
    if table["cultivar_ascii"].duplicated().any():
        raise ValueError("V2 cultivar registry contains duplicate source labels")
    if table["cultivar_code"].duplicated().any():
        raise ValueError("V2 cultivar registry contains duplicate publication codes")
    if not table["cultivar_code"].str.fullmatch(r"[A-Z]+[A-Z0-9]*").all():
        raise ValueError("V2 cultivar codes must contain uppercase letters followed by uppercase letters/digits")
    return table


def trait_registry() -> pd.DataFrame:
    """Return the authoritative V2 trait abbreviation registry."""
    table = pd.read_csv(CONFIG_DIR / "v2_trait_registry.csv", keep_default_na=False)
    if table["target"].duplicated().any() or table["abbreviation"].duplicated().any():
        raise ValueError("V2 trait registry contains duplicate targets or abbreviations")
    return table


def cultivar_code_map() -> dict[str, str]:
    table = cultivar_registry()
    return dict(zip(table["cultivar_ascii"], table["cultivar_code"], strict=True))


def trait_abbreviation_map() -> dict[str, str]:
    table = trait_registry()
    return dict(zip(table["target"], table["abbreviation"], strict=True))


def add_cultivar_code(frame: pd.DataFrame, source_column: str = "cultivar_ascii") -> pd.DataFrame:
    """Add publication-facing cultivar_code while retaining the traceable source label."""
    if source_column not in frame.columns:
        raise KeyError(f"Missing cultivar source column: {source_column}")
    result = frame.copy()
    result["cultivar_code"] = result[source_column].map(cultivar_code_map())
    missing = sorted(result.loc[result["cultivar_code"].isna(), source_column].astype(str).unique())
    if missing:
        raise ValueError(f"Cultivar labels missing from V2 registry: {missing}")
    return result


def abbreviated_trait(target: str) -> str:
    mapping = trait_abbreviation_map()
    if target not in mapping:
        raise KeyError(f"Trait missing from V2 registry: {target}")
    return mapping[target]
