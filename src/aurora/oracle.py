"""Oracle v1: frozen ESM-2 embeddings + a small head that predicts brightness.

The plan is deliberately cheap where it can be. ESM-2 is frozen — we run every
sequence through it exactly once, mean-pool to a single vector, and cache those
vectors to disk. After that the head is trained on cached features, so trying a
new head (or a new alpha) costs seconds, not another pass over 54k sequences.

Two heads, on purpose:
  - Ridge  — the honest linear baseline. Near-instant, hard to overfit. If a
             linear probe already ranks well, that tells you the embeddings
             carry the signal and the head barely matters.
  - MLP    — a small nonlinear head. The interesting question is whether it
             actually *helps on test* (>=4 mutations, out of distribution) or
             just fits the train neighborhood better and buys nothing.

The number that counts is TEST-set Spearman, and the split is sacred (see
data.py / data/README.md). We train on `train`, pick hyperparameters on `valid`
(both <=3 mutations, in-distribution), and never let `test` (>=4 mutations)
touch anything until the final read. Spearman because the oracle's job in the
loop is to *rank* candidate sequences, not to nail their absolute brightness.

Run the whole thing:
    uv run python -m aurora.oracle
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aurora.data import load_gfp

# 650M is the big honest checkpoint — 1280-dim embeddings, the best number we'll
# get on an M-series Mac without fine-tuning. Swap the tag for a smaller one
# (esm2_t30_150M_UR50D, esm2_t12_35M_UR50D) if you want to iterate faster.
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# Cache next to the dataset. data/ is gitignored, so these never hit the repo.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


# --------------------------------------------------------------------------- #
# Embeddings                                                                    #
# --------------------------------------------------------------------------- #

def _device():
    """MPS on Apple Silicon, CUDA if it's there, else CPU. No fuss."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_sequences(
    sequences: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    half: bool = False,
) -> np.ndarray:
    """Run sequences through a frozen ESM-2 and mean-pool to one vector each.

    Returns an (n, hidden) float32 array in the *same order* as `sequences`.

    We sort by length before batching so each batch pads to roughly its own
    longest member instead of the longest in the whole set — a big win when a
    few sequences are long and most aren't. The result is un-sorted back to the
    caller's order before returning, so alignment with the dataframe is safe.

    half=True runs the model in float16 — roughly half the time and memory on
    MPS. It barely moves Spearman here, but it's off by default so the cached
    features are the exact-precision ones.
    """
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device).eval()
    if half:
        model = model.half()

    # Longest-first isn't necessary; we just want like-length things together.
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    out = np.empty((len(sequences), model.config.hidden_size), dtype=np.float32)

    n_batches = (len(order) + batch_size - 1) // batch_size
    with torch.no_grad():
        for b, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start : start + batch_size]
            batch = [sequences[i] for i in idx]

            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state  # (B, L, H)

            # Mean over *residues only*. ESM wraps each sequence in <cls>...<eos>,
            # and pads to batch length; we want none of those in the average.
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                                   # drop <cls>
            lengths = enc["attention_mask"].sum(dim=1)
            mask[torch.arange(mask.size(0)), lengths - 1] = 0  # drop <eos>

            mask = mask.unsqueeze(-1).to(hidden.dtype)       # (B, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            for j, i in enumerate(idx):
                out[i] = pooled[j].float().cpu().numpy()

            if b % 25 == 0 or b == n_batches - 1:
                print(f"  embedded batch {b + 1}/{n_batches}", flush=True)

    return out


def get_embeddings(
    df: pd.DataFrame | None = None,
    model_name: str = MODEL_NAME,
    force: bool = False,
    **kw,
) -> np.ndarray:
    """Embeddings for every row of `df`, cached to disk and reused forever.

    Cache is keyed by the checkpoint name and row count, so switching model size
    (or growing the dataset) invalidates it cleanly instead of silently handing
    back stale vectors.
    """
    if df is None:
        df = load_gfp()

    tag = model_name.split("/")[-1]
    cache = _CACHE_DIR / f"emb_{tag}.npy"

    if cache.exists() and not force:
        X = np.load(cache)
        if len(X) == len(df):
            print(f"[oracle] loaded cached embeddings: {cache.name} {X.shape}")
            return X
        print("[oracle] cache row count doesn't match df; re-embedding.")

    print(f"[oracle] embedding {len(df):,} sequences with {tag} "
          f"(one-time cost)...")
    X = embed_sequences(df["sequence"].tolist(), model_name=model_name, **kw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    print(f"[oracle] cached -> {cache}")
    return X


# --------------------------------------------------------------------------- #
# Heads                                                                         #
# --------------------------------------------------------------------------- #

def _spearman(pred, true) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(pred, true).correlation)


def _rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def fit_ridge(Xtr, ytr, Xva, yva, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Standardize features, sweep alpha, keep whatever ranks `valid` best.

    Selection is on valid Spearman, not valid MSE — we're optimizing the thing
    we actually report, and rank quality and squared error don't always agree.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    best = None
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr_s, ytr)
        rho = _spearman(model.predict(Xva_s), yva)
        print(f"  ridge alpha={a:<8g} valid Spearman={rho:.4f}")
        if best is None or rho > best[0]:
            best = (rho, a, model)

    _, alpha, model = best
    print(f"  -> picked alpha={alpha:g}")

    def predict(X):
        return model.predict(scaler.transform(X))

    return predict


def fit_mlp(Xtr, ytr, Xva, yva, hidden=256, dropout=0.2, lr=1e-3,
            max_epochs=200, patience=15, batch_size=256, seed=0):
    """A small ReLU head, MSE loss, early-stopped on valid Spearman.

    Features are standardized on train stats; the target is standardized too,
    purely for optimization stability — we invert it before scoring so RMSE
    lands back in log-fluorescence units. (Spearman doesn't care either way,
    it's rank-only, but RMSE has to be on the real scale to mean anything.)
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    device = _device()

    xs = StandardScaler().fit(Xtr)
    y_mean, y_std = float(np.mean(ytr)), float(np.std(ytr)) or 1.0

    def to_x(X):
        return torch.tensor(xs.transform(X), dtype=torch.float32, device=device)

    def to_y(y):
        return torch.tensor((np.asarray(y) - y_mean) / y_std,
                            dtype=torch.float32, device=device)

    Xtr_t, ytr_t = to_x(Xtr), to_y(ytr)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def predict(X):
        model.eval()
        with torch.no_grad():
            p = model(to_x(X)).squeeze(-1).cpu().numpy()
        return p * y_std + y_mean

    best_rho, best_state, waited = -np.inf, None, 0
    n = len(Xtr_t)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]).squeeze(-1), ytr_t[b])
            loss.backward()
            opt.step()

        rho = _spearman(predict(Xva), yva)
        if rho > best_rho:
            best_rho, waited = rho, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  early stop at epoch {epoch} "
                      f"(best valid Spearman={best_rho:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return predict


# --------------------------------------------------------------------------- #
# Eval                                                                          #
# --------------------------------------------------------------------------- #

def evaluate(predict, splits: dict[str, tuple]) -> pd.DataFrame:
    """Spearman + RMSE on each split. `splits` maps name -> (X, y)."""
    rows = []
    for name, (X, y) in splits.items():
        p = predict(X)
        rows.append({"split": name, "n": len(y),
                     "spearman": _spearman(p, y), "rmse": _rmse(p, y)})
    return pd.DataFrame(rows).set_index("split")


def _scatter(predict, Xte, yte, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = predict(Xte)
    plt.figure(figsize=(5, 5))
    plt.scatter(yte, p, s=4, alpha=0.25)
    lo, hi = min(yte.min(), p.min()), max(yte.max(), p.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("true log-fluorescence")
    plt.ylabel("predicted")
    plt.title(f"{title}\ntest Spearman = {_spearman(p, yte):.3f}")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    print(f"[oracle] wrote {path}")


# --------------------------------------------------------------------------- #
# Run it                                                                        #
# --------------------------------------------------------------------------- #

def main():
    df = load_gfp()
    X = get_embeddings(df)
    y = df["log_fluorescence"].to_numpy()
    split = df["split"].to_numpy()

    # Respect the split. valid is held out for selection; test is untouched.
    def part(name):
        m = split == name
        return X[m], y[m]

    Xtr, ytr = part("train")
    Xva, yva = part("valid")
    Xte, yte = part("test")
    print(f"[oracle] train {len(ytr):,}  valid {len(yva):,}  test {len(yte):,}")

    splits = {"train": (Xtr, ytr), "valid": (Xva, yva), "test": (Xte, yte)}
    results_dir = Path(__file__).resolve().parents[2] / "results"

    print("\n=== Ridge ===")
    ridge = fit_ridge(Xtr, ytr, Xva, yva)
    print(evaluate(ridge, splits).round(4))
    _scatter(ridge, Xte, yte, "Oracle v1 — Ridge on ESM-2",
             results_dir / "oracle_ridge_test.png")

    print("\n=== MLP ===")
    mlp = fit_mlp(Xtr, ytr, Xva, yva)
    print(evaluate(mlp, splits).round(4))
    _scatter(mlp, Xte, yte, "Oracle v1 — MLP on ESM-2",
             results_dir / "oracle_mlp_test.png")


if __name__ == "__main__":
    main()
