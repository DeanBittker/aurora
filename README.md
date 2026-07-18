# Aurora

In-silico closed-loop protein design. The aim is to design brighter GFP variants by
iterating: propose sequences, score them with a learned oracle, keep the best ones,
and propose again from those.

The oracle is imperfect. An optimizer that leans on it too hard will find sequences
that score well under the oracle without being bright, by exploiting the oracle's
errors. A large part of this project is measuring that gap and reducing it. The
failure modes involved are distribution shift and reward hacking.

## Setup

Python is managed with uv.

```bash
uv sync
uv run python -m aurora.data      # download and cache the GFP dataset
```

Code is src-layout, so it imports as `import aurora`.

## Layout

```
src/aurora/
  data.py        GFP loader and the train/test split.
  oracle.py      oracle v1. frozen ESM-2 embeddings into a regression head.
  oracle_v2.py   oracle v2. fine-tuned ESM-2 with an ensemble for uncertainty.
  proposers.py   sequence proposers. todo.
  loop.py        the closed loop. todo.
  evaluate.py    ground-truth eval and reward-hacking analysis. todo.
notebooks/       exploration
NOTES.md         running log of the modeling work and results
```

## Data and the split

GFP fluorescence landscape from Sarkisyan et al. 2016, via TAPE. 54k point mutants
of avGFP, each with a brightness label (`log_fluorescence`, higher is brighter).

The split groups variants by edit distance from the wild-type. Train and valid hold
variants with 3 mutations or fewer. Test holds variants with 4 or more. Train is
about 82% bright and test is about 32%. The oracle learns from the bright local
neighborhood and is then measured on the farther, mostly-dark region it did not see
during training, which measures extrapolation. If train and test are combined and
reshuffled, near-duplicate variants end up on both sides of the boundary and the
reported scores are inflated, so the split is kept as is. More in `data/README.md`.

## Status

- [x] Step 1. Data pipeline and EDA.
- [x] Step 2. Oracle v1: frozen ESM-2 (650M) embeddings, Ridge and MLP heads. Test
  Spearman about 0.62. Ranking mostly holds under the distribution shift, absolute
  calibration does not, and RMSE roughly doubles on the far set. That gap is the
  subject of step 6.
- [x] Step 3. Oracle v2: fine-tuned ESM-2 (150M) end to end, three-model ensemble for
  uncertainty. Test Spearman about 0.67 and lower error on the far set than v1. The
  ensemble uncertainty is larger on test than on valid, so it grows on the region the
  models did not train on. See NOTES.md.
- [ ] Step 4. Closed loop: directed-evolution proposer plus a held-out ground-truth
  evaluator.
- [ ] Step 5. Smarter proposals: ESM masked-LM-guided mutations compared to random
  evolution.
- [ ] Step 6. Reward hacking. Measure the gap between oracle score and true score on
  designed sequences, show an optimizer exploiting it, and reduce it with a trust
  region or an uncertainty penalty.
- [ ] Step 7. Benchmark against FLEXS or AdaLead.
- [ ] Step 8. Package it up, short writeup, maybe a Gradio demo.
