"""Oracle v2. Fine-tune ESM-2 end to end, with a deep ensemble for uncertainty.

v1 froze ESM-2 and trained a head on cached embeddings. It ranked the far test set
about right (Spearman ~0.62) but its absolute predictions fell apart out there. v2
does two things about that.

  1. Fine-tune the backbone. The ESM weights move on the fluorescence task instead
     of staying fixed. This should lift the headline number.
  2. Train an ensemble. Three fine-tunes from different seeds. The spread of their
     predictions is an uncertainty estimate. The question that matters is whether
     that spread gets larger out on the far, mostly-dark test set the models never
     saw. If it does, the oracle knows where it is ignorant. That is the property a
     naive optimizer will try to exploit later, in step 6.

Same split rules as always. Train on `train`, pick the checkpoint on `valid`,
report `test`. The target is standardized on train stats for stable optimization
and un-standardized before scoring, so RMSE stays in log-fluorescence units.

Run it:
    uv run python -m aurora.oracle_v2

This is a real fine-tune. Expect roughly 20 to 40 minutes per epoch per model on an
M-series Mac, times three models. Fine to leave running.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aurora.data import load_gfp

# 150M is the size that actually fine-tunes on a laptop. v1 used frozen 650M, so
# the honest comparison is frozen-150M vs fine-tuned-150M for the fine-tuning
# effect, plus the frozen-650M number from v1 for reference.
MODEL_NAME = "facebook/esm2_t30_150M_UR50D"

_RESULTS = Path(__file__).resolve().parents[2] / "results"


def _device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _spearman(pred, true) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(pred, true).correlation)


def _rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


# --------------------------------------------------------------------------- #
# Model                                                                         #
# --------------------------------------------------------------------------- #

def _build_model(model_name, head_dropout=0.2):
    import torch.nn as nn
    from transformers import EsmModel

    class ESMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            # add_pooling_layer=False drops the unused CLS pooler. We mean-pool
            # residues ourselves, same as v1.
            self.esm = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
            h = self.esm.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(h, 1))

        def forward(self, input_ids, attention_mask):
            import torch

            hidden = self.esm(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
            # Mean over residues only. Drop <cls> at the front and <eos> at the end.
            mask = attention_mask.clone()
            mask[:, 0] = 0
            lengths = attention_mask.sum(dim=1)
            mask[torch.arange(mask.size(0)), lengths - 1] = 0
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.head(pooled).squeeze(-1)

    return ESMRegressor()


# --------------------------------------------------------------------------- #
# Train one member                                                              #
# --------------------------------------------------------------------------- #

def train_member(seq_tr, y_tr, seq_va, y_va, seed, model_name=MODEL_NAME,
                 epochs=4, batch_size=16, lr_backbone=2e-5, lr_head=1e-3,
                 patience=2):
    """Fine-tune one ESM regressor. Early-stopped on valid Spearman.

    Returns a predict(list_of_sequences) -> np.ndarray function that already
    un-standardizes back to log-fluorescence units.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    device = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = _build_model(model_name).to(device)

    # Standardize the target on train stats only. Spearman doesn't care, but it
    # keeps the loss well scaled and lets RMSE come back in real units.
    y_mean, y_std = float(np.mean(y_tr)), float(np.std(y_tr)) or 1.0

    class SeqData(Dataset):
        def __init__(self, seqs, ys):
            self.seqs = list(seqs)
            self.ys = np.asarray(ys, dtype=np.float32)

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, i):
            return self.seqs[i], self.ys[i]

    def collate(batch):
        seqs, ys = zip(*batch)
        enc = tok(list(seqs), return_tensors="pt", padding=True,
                  truncation=True, max_length=1024)
        return enc, torch.tensor(ys, dtype=torch.float32)

    y_tr_std = (np.asarray(y_tr) - y_mean) / y_std
    loader = DataLoader(SeqData(seq_tr, y_tr_std), batch_size=batch_size,
                        shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW(
        [{"params": model.esm.parameters(), "lr": lr_backbone},
         {"params": model.head.parameters(), "lr": lr_head}],
        weight_decay=0.01,
    )
    loss_fn = nn.MSELoss()

    @torch.no_grad()
    def predict(seqs, infer_batch=32):
        model.eval()
        out = []
        for i in range(0, len(seqs), infer_batch):
            chunk = list(seqs[i:i + infer_batch])
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            p = model(enc["input_ids"], enc["attention_mask"]).cpu().numpy()
            out.append(p)
        return np.concatenate(out) * y_std + y_mean

    best_rho, best_state, waited = -np.inf, None, 0
    n_batches = len(loader)
    for epoch in range(epochs):
        model.train()
        for b, (enc, yb) in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(enc["input_ids"], enc["attention_mask"])
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            if b % 100 == 0:
                print(f"    seed {seed} epoch {epoch} batch {b}/{n_batches} "
                      f"loss {loss.item():.3f}", flush=True)

        rho = _spearman(predict(seq_va), y_va)
        print(f"  seed {seed} epoch {epoch}: valid Spearman {rho:.4f}", flush=True)
        if rho > best_rho:
            best_rho, waited = rho, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  seed {seed} early stop (best valid {best_rho:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return predict


# --------------------------------------------------------------------------- #
# Calibration                                                                   #
# --------------------------------------------------------------------------- #

def calibration(y_true, mean, std):
    """How honest is the uncertainty on this split?

    - err_vs_std: Spearman between the ensemble std and the actual absolute error.
      Positive means bigger uncertainty tends to mean bigger error, which is what
      we want.
    - cov68 / cov90: fraction of true values that land inside mean +/- z*std,
      treating the ensemble spread as a Gaussian sigma. Ideal is 0.68 and 0.90.
      A K=3 ensemble underestimates sigma, so expect these to run low. The useful
      read is the trend across splits, not the exact number.
    """
    err = np.abs(np.asarray(mean) - np.asarray(y_true))
    std = np.asarray(std)
    return {
        "mean_std": float(std.mean()),
        "err_vs_std": _spearman(std, err),
        "cov68": float(np.mean(np.abs(np.asarray(y_true) - mean) <= 1.0 * std)),
        "cov90": float(np.mean(np.abs(np.asarray(y_true) - mean) <= 1.645 * std)),
    }


def _plots(y_te, mean_te, std_te, y_va, std_va):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _RESULTS.mkdir(parents=True, exist_ok=True)

    # Predicted vs true on test, colored by uncertainty.
    plt.figure(figsize=(5.4, 5))
    sc = plt.scatter(y_te, mean_te, c=std_te, s=6, alpha=0.4, cmap="viridis")
    lo, hi = min(y_te.min(), mean_te.min()), max(y_te.max(), mean_te.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.colorbar(sc, label="ensemble std")
    plt.xlabel("true log-fluorescence")
    plt.ylabel("ensemble mean prediction")
    plt.title(f"Oracle v2 test. Spearman {_spearman(mean_te, y_te):.3f}")
    plt.tight_layout()
    plt.savefig(_RESULTS / "oracle_v2_test.png", dpi=140)

    # The money plot. Is the oracle more uncertain out of distribution?
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, max(std_va.max(), std_te.max()), 40)
    plt.hist(std_va, bins=bins, alpha=0.6, density=True, label="valid (<=3 mut)")
    plt.hist(std_te, bins=bins, alpha=0.6, density=True, label="test (>=4 mut)")
    plt.xlabel("ensemble std (uncertainty)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Does uncertainty rise out of distribution?")
    plt.tight_layout()
    plt.savefig(_RESULTS / "oracle_v2_uncertainty.png", dpi=140)
    print(f"[oracle_v2] wrote plots to {_RESULTS}")


# --------------------------------------------------------------------------- #
# Run it                                                                        #
# --------------------------------------------------------------------------- #

def main(k=3, limit=None):
    df = load_gfp()
    if limit:
        # Smoke test. Subsample each split so a full pass takes minutes.
        parts = [g.sample(min(len(g), limit), random_state=0) for _, g in df.groupby("split")]
        import pandas as pd
        df = pd.concat(parts, ignore_index=True)

    def part(name):
        g = df[df["split"] == name]
        return g["sequence"].tolist(), g["log_fluorescence"].to_numpy()

    seq_tr, y_tr = part("train")
    seq_va, y_va = part("valid")
    seq_te, y_te = part("test")
    print(f"[oracle_v2] train {len(y_tr):,}  valid {len(y_va):,}  test {len(y_te):,}")

    # Train the ensemble. One full fine-tune per seed.
    preds = {"train": [], "valid": [], "test": []}
    for seed in range(k):
        print(f"\n=== ensemble member {seed + 1}/{k} (seed {seed}) ===")
        predict = train_member(seq_tr, y_tr, seq_va, y_va, seed=seed)
        preds["train"].append(predict(seq_tr))
        preds["valid"].append(predict(seq_va))
        preds["test"].append(predict(seq_te))

    # Ensemble = mean prediction. Uncertainty = spread across members.
    def stack(name):
        arr = np.stack(preds[name])          # (k, n)
        return arr.mean(0), arr.std(0)

    mean_tr, _ = stack("train")
    mean_va, std_va = stack("valid")
    mean_te, std_te = stack("test")

    print("\n=== Oracle v2 (ensemble mean) ===")
    for name, mean, y in [("train", mean_tr, y_tr),
                          ("valid", mean_va, y_va),
                          ("test", mean_te, y_te)]:
        print(f"  {name:>5}: Spearman {_spearman(mean, y):.4f}  "
              f"RMSE {_rmse(mean, y):.4f}")
    print("  (v1 frozen-650M test Spearman was ~0.62 for reference)")

    print("\n=== Calibration ===")
    for name, mean, std, y in [("valid", mean_va, std_va, y_va),
                               ("test", mean_te, std_te, y_te)]:
        c = calibration(y, mean, std)
        print(f"  {name:>5}: mean_std {c['mean_std']:.3f}  "
              f"err_vs_std {c['err_vs_std']:.3f}  "
              f"cov68 {c['cov68']:.2f}  cov90 {c['cov90']:.2f}")

    _plots(y_te, mean_te, std_te, y_va, std_va)


if __name__ == "__main__":
    main()
