"""Load the GFP (avGFP) fluorescence landscape dataset.

Source: Sarkisyan et al. 2016, "Local fitness landscape of the green
fluorescent protein" (Nature). Curated by TAPE and mirrored on HuggingFace.
See data/README.md for the full story and the split logic.

Typical use:
    from aurora.data import load_gfp, train_test
    df = load_gfp()                 # full dataset as a DataFrame
    train_df, test_df = train_test()  # TAPE's edit-distance split
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Primary source. If this ever 404s, these mirrors carry the same TAPE data:
#   "cradle-bio/tape-fluorescence", "SaProtHub/Dataset-Fluorescence-TAPE"
HF_DATASET = "biomap-research/fluorescence_prediction"

# Different mirrors use slightly different column names, so we detect them.
_SEQ_KEYS = ("primary", "sequence", "seq")
_LABEL_KEYS = ("log_fluorescence", "target", "label", "fluorescence")
_NMUT_KEYS = ("num_mutations", "n_mutations", "mutations")

# Repo-root-relative cache. This file lives at <repo>/src/aurora/data.py,
# so parents[2] is the repo root. The data/ folder is gitignored.
_CACHE = Path(__file__).resolve().parents[2] / "data" / "processed" / "gfp.csv"


def _pick(colnames, candidates, what):
    """Return the first candidate present in colnames, else a clear error."""
    for c in candidates:
        if c in colnames:
            return c
    raise KeyError(
        f"Couldn't find the {what} column. Looked for {candidates}, but the "
        f"dataset columns are {list(colnames)}. Update the *_KEYS lists in data.py."
    )


def _to_float(series: pd.Series) -> pd.Series:
    """Some TAPE mirrors store the label as a 1-element list; unwrap it."""
    def one(v):
        if isinstance(v, (list, tuple)):
            return float(v[0])
        return float(v)
    return series.map(one)


def _download() -> pd.DataFrame:
    # Imported lazily so `import aurora.data` stays fast and dependency-light.
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET)  # a DatasetDict: train / valid / test

    frames = []
    for split_name, split in ds.items():
        part = split.to_pandas()
        part["split"] = split_name
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)

    seq_col = _pick(raw.columns, _SEQ_KEYS, "sequence")
    label_col = _pick(raw.columns, _LABEL_KEYS, "log-fluorescence")

    out = pd.DataFrame(
        {
            "sequence": raw[seq_col].astype(str),
            "log_fluorescence": _to_float(raw[label_col]),
            "split": raw["split"].astype(str),
        }
    )

    # Keep num_mutations if the mirror ships it (TAPE does); otherwise leave
    # it blank rather than guessing a wild-type and computing it wrong.
    try:
        nmut_col = _pick(raw.columns, _NMUT_KEYS, "num_mutations")
        out["num_mutations"] = raw[nmut_col].astype("Int64")
    except KeyError:
        out["num_mutations"] = pd.array([pd.NA] * len(out), dtype="Int64")
        print("[data] note: num_mutations not found in this mirror; left blank.")

    return out


def load_gfp(force_download: bool = False) -> pd.DataFrame:
    """Full GFP dataset with columns: sequence, log_fluorescence, num_mutations, split.

    Downloads from HuggingFace on first call, then caches to data/processed/gfp.csv
    so later calls are instant and work offline. Pass force_download=True to refresh.
    """
    if _CACHE.exists() and not force_download:
        return pd.read_csv(_CACHE)
    df = _download()
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE, index=False)
    return df


def train_test(df: pd.DataFrame | None = None):
    """TAPE's edit-distance split.

    train = TAPE train + valid  (variants <=3 mutations from wild-type)
    test  = TAPE test           (variants >=4 mutations from wild-type)

    The gap between them is intentional: it measures how well a model trained on
    the local neighborhood extrapolates to the far, mostly-dark part of the
    landscape. Don't re-shuffle these together, or you'll leak and fool yourself.
    """
    if df is None:
        df = load_gfp()
    train = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    df = load_gfp(force_download=True)
    print(f"Loaded {len(df):,} GFP variants -> cached at {_CACHE}")
    print(df["split"].value_counts())
    print(df.head())
