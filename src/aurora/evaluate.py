"""Step 6. Reward hacking, and a fix.

Step 4 showed the loop optimizing the oracle runs away from truth: the oracle's
predicted brightness climbs far past anything real while the true score lags. This
measures that gap and then reduces it with a trust region.

The pieces:

  1. The oracle. An ensemble of Ridge heads on bootstrap resamples of the train set,
     giving a mean prediction. This is the thing the loop optimizes, and it is only
     trained on the near region, so it is untrustworthy far out.

  2. An honest ground truth. A random forest on all the data. A linear model
     extrapolates without bound out of distribution and predicts brightness past the
     brightest protein ever measured. A forest only averages real training labels, so
     a far design gets scored like the real variants nearest it, which out there are
     mostly dark. That is a believable stand-in for reality where a linear model is
     not.

  3. The fix: a trust region on distance from the training data. We first tried a
     penalty on the ensemble's own uncertainty, but its spread was too small to hold
     the line against an oracle mean running off to 11. Distance from the training
     embeddings is a stronger, monotonic "how far out am I" signal: it grows steadily
     as the design drifts, so penalizing it keeps the search inside the region the
     oracle can be trusted. Select by mean - lam * distance. lam = 0 is the greedy,
     reward-hacking loop.

We sweep lam and look at two things per round: the gap between oracle score and true
score, and the true brightness of the design. The greedy run should show a big gap
and dim designs. A large enough lam should shrink the gap and pull the designs back
up to genuinely bright.

Run it:
    uv run python -m aurora.evaluate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aurora.data import load_gfp
from aurora.oracle import get_embeddings
from aurora.proposers import propose
from aurora.loop import _make_embedder, _wild_type

_RESULTS = Path(__file__).resolve().parents[2] / "results"


def build_oracle_ensemble(Xtr, ytr, k=20, alpha=10.0, seed=0):
    """Ensemble of Ridge heads on bootstrap resamples. Returns mean(X) -> scores."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = len(ytr)
    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)

    coefs, intercepts = [], []
    for _ in range(k):
        idx = rng.integers(0, n, size=n)
        model = Ridge(alpha=alpha).fit(Xs[idx], ytr[idx])
        coefs.append(model.coef_)
        intercepts.append(model.intercept_)
    coefs = np.stack(coefs)
    intercepts = np.array(intercepts)

    def mean(X):
        return (scaler.transform(X) @ coefs.T + intercepts).mean(axis=1)

    return mean


def build_ground_truth(X_all, y_all, n_trees=200, seed=0):
    """Random forest on all the data. Cannot extrapolate past the real label range."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=n_trees, max_depth=None, n_jobs=-1, random_state=seed,
    ).fit(X_all, y_all)
    return model.predict


def build_distance(Xtr, k=10, sample=3000, seed=0):
    """Distance from the training embeddings, a direct out-of-distribution signal.

    For a candidate, take the mean distance to its k nearest training sequences,
    normalized by the typical distance among training points. In-distribution
    candidates land near 1, and it grows as candidates drift away from the data.
    """
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=k).fit(Xtr)

    def raw(X):
        d, _ = knn.kneighbors(X)
        return d.mean(axis=1)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(sample, len(Xtr)), replace=False)
    scale = float(np.median(raw(Xtr[idx]))) or 1.0

    def distance(X):
        return raw(X) / scale

    return distance


def evolve(oracle_mean, distance, ground_truth, embed, start, lam,
           rounds=12, children_per_parent=20, n_mut=1, keep=10, seed=0):
    """One directed-evolution run selecting by mean - lam * distance.

    Records (round, oracle_mean, distance, true) for the selected best each round.
    """
    import random

    rng = random.Random(seed)
    population = [start]
    history = []
    for r in range(rounds):
        children = propose(population, children_per_parent, n_mut, rng)
        pool = list(dict.fromkeys(population + children))

        X = embed(pool)
        mean = oracle_mean(X)
        dist = distance(X)
        selection = mean - lam * dist            # the trust region
        order = np.argsort(selection)[::-1]
        population = [pool[i] for i in order[:keep]]

        best = order[0]
        true = float(ground_truth(X[best:best + 1])[0])
        history.append((r, float(mean[best]), float(dist[best]), true))
    return history


def main(lambdas=(0.0, 2.0, 5.0, 10.0)):
    df = load_gfp()
    X_all = get_embeddings(df)
    y = df["log_fluorescence"].to_numpy()
    is_train = (df["split"] == "train").to_numpy()

    print("[eval] building bootstrap-ensemble oracle (train only)...")
    oracle_mean = build_oracle_ensemble(X_all[is_train], y[is_train])
    print("[eval] building random-forest ground truth (all data)...")
    ground_truth = build_ground_truth(X_all, y)
    print("[eval] building distance-to-training-data trust region...")
    distance = build_distance(X_all[is_train])

    embed = _make_embedder()
    start = _wild_type(df)

    runs = {}
    for lam in lambdas:
        print(f"\n=== lambda = {lam} ===")
        hist = evolve(oracle_mean, distance, ground_truth, embed, start, lam)
        runs[lam] = hist
        last = hist[-1]
        gap = last[1] - last[3]
        print(f"  final: oracle {last[1]:.2f}  dist {last[2]:.2f}  "
              f"true {last[3]:.2f}  gap {gap:.2f}")

    _plot(runs)


def _plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for lam, hist in runs.items():
        rounds = [h[0] for h in hist]
        gap = [h[1] - h[3] for h in hist]
        true = [h[3] for h in hist]
        ax1.plot(rounds, gap, marker="o", label=f"lambda={lam}")
        ax2.plot(rounds, true, marker="o", label=f"lambda={lam}")

    ax1.set_xlabel("round")
    ax1.set_ylabel("oracle score minus true score")
    ax1.set_title("Reward-hacking gap vs trust-region strength")
    ax1.legend()

    ax2.set_xlabel("round")
    ax2.set_ylabel("true (ground-truth) brightness of the design")
    ax2.set_title("What the designs are actually worth")
    ax2.legend()

    plt.tight_layout()
    _RESULTS.mkdir(parents=True, exist_ok=True)
    path = _RESULTS / "reward_hacking.png"
    plt.savefig(path, dpi=140)
    print(f"[eval] wrote {path}")


if __name__ == "__main__":
    main()
