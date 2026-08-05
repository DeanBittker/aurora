"""The closed loop. Directed evolution driven by the oracle, graded by ground truth.

This is where the oracle gets used the way it would be used for real: score many
candidate sequences, keep the best, repeat. The loop is greedy directed evolution.

  1. Start from the wild-type sequence.
  2. Propose mutated candidates (proposers.propose).
  3. Score them with the ORACLE and keep the top few.
  4. Repeat.

The point is not just to climb. Each round we track two numbers: what the oracle
THINKS the best sequence scores, and what a separate ground-truth model says it
ACTUALLY scores. The oracle is trained only on the near region (<=3 mutations), so
as the loop pushes further out it starts trusting a model that is guessing. The
ground-truth model is trained on the full landscape, so it is a more honest stand-in
for reality. When the two curves split apart, that is reward hacking: the optimizer
winning on the oracle while losing in truth.

Ground truth here is still a model, a proxy for reality, not reality. That is the
standard way to study this in-silico and it is worth saying plainly.

Run it:
    uv run python -m aurora.loop
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from aurora.data import load_gfp
from aurora.oracle import MODEL_NAME, get_embeddings
from aurora.proposers import propose

_RESULTS = Path(__file__).resolve().parents[2] / "results"


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_head(X, y, alpha=10.0):
    """Standardize features, fit Ridge, return a predict(embeddings) -> scores fn."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    return lambda Xnew: model.predict(scaler.transform(Xnew))


def _make_embedder(model_name=MODEL_NAME):
    """Load frozen ESM once and return an embed(seqs) -> array function.

    Loading the 650M model is slow, so we load it a single time and reuse it every
    round instead of reloading. The masked mean-pool matches the one used to build
    the cached training embeddings, so the heads trained on those work here too.
    """
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
    model = model.to(device).eval()

    @torch.no_grad()
    def embed(seqs, batch_size=16):
        out = []
        for i in range(0, len(seqs), batch_size):
            batch = list(seqs[i:i + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0
            lengths = enc["attention_mask"].sum(dim=1)
            mask[torch.arange(mask.size(0)), lengths - 1] = 0
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out)

    return embed


def _wild_type(df):
    """Starting sequence: the 0-mutation wild-type if present, else the brightest
    training sequence.
    """
    mask = (df["num_mutations"] == 0).fillna(False)
    wt = df[mask]
    if len(wt):
        return wt["sequence"].iloc[0]
    train = df[df["split"] == "train"]
    return train.loc[train["log_fluorescence"].idxmax(), "sequence"]


def run_loop(rounds=15, children_per_parent=20, n_mut=1, keep=10, seed=0):
    # Two heads on the cached ESM embeddings from step 2. The oracle sees only the
    # near region (the realistic case). Ground truth sees the whole landscape.
    df = load_gfp()
    X_all = get_embeddings(df)
    y = df["log_fluorescence"].to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    print("[loop] fitting oracle (train-only) and ground-truth (all-data) heads...")
    oracle_head = _fit_head(X_all[is_train], y[is_train])
    gt_head = _fit_head(X_all, y)

    embed = _make_embedder()
    rng = random.Random(seed)

    population = [_wild_type(df)]
    history = []
    for r in range(rounds):
        # Mutate the current population, pool with parents so a good one can survive.
        children = propose(population, children_per_parent, n_mut, rng)
        pool = list(dict.fromkeys(population + children))   # dedup, keep order

        X = embed(pool)                      # embed once, reuse for both heads
        oracle_scores = oracle_head(X)
        true_scores = gt_head(X)

        order = np.argsort(oracle_scores)[::-1]   # rank by ORACLE, the optimizer's view
        population = [pool[i] for i in order[:keep]]

        best = order[0]
        history.append((r, float(oracle_scores[best]), float(true_scores[best])))
        print(f"round {r:2d}: oracle {oracle_scores[best]:.3f}   "
              f"true {true_scores[best]:.3f}", flush=True)

    _plot(history)
    return history


def _plot(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [h[0] for h in history]
    oracle = [h[1] for h in history]
    true = [h[2] for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(rounds, oracle, marker="o", label="oracle-predicted best")
    plt.plot(rounds, true, marker="o", label="ground-truth of that best")
    plt.xlabel("round")
    plt.ylabel("log-fluorescence")
    plt.title("Closed loop: what the oracle thinks vs what is true")
    plt.legend()
    plt.tight_layout()
    _RESULTS.mkdir(parents=True, exist_ok=True)
    path = _RESULTS / "loop_best_vs_round.png"
    plt.savefig(path, dpi=140)
    print(f"[loop] wrote {path}")


def main():
    run_loop()


if __name__ == "__main__":
    main()
