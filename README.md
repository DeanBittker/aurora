# Aurora

In-silico closed-loop protein design. The goal is brighter GFP variants, found by
iterating: propose sequences, score them with a learned oracle, keep the best,
propose again.

The oracle is wrong, and that is the valuable part. A naive optimizer will
exploit its errors and give you sequences that score high but aren't actually
bright. Measuring that failure (distribution shift, reward hacking) and fixing it
is the point.

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
  data.py        GFP loader and the train/test split. don't reshuffle it.
  oracle.py      oracle v1. frozen ESM-2 embeddings into a regression head.
  proposers.py   sequence proposers. todo.
  loop.py        the closed loop. todo.
  evaluate.py    ground-truth eval and reward-hacking analysis. todo.
notebooks/       exploration
```

## Data and the split

GFP fluorescence landscape from Sarkisyan et al. 2016, via TAPE. 54k point mutants
of avGFP, each with a brightness label (`log_fluorescence`, higher is brighter).

The split is by edit distance from wild-type, not random. Train and valid are 3
mutations or fewer. Test is 4 or more. Train is about 82% bright. Test is about 32%.
So the oracle learns a bright local neighborhood and gets graded on the far,
mostly-dark region it never saw. That is a real extrapolation test, set up that way
on purpose. Reshuffle train and test together and you leak near-duplicates across
the boundary, which inflates every number. So don't. More in `data/README.md`.

## Status

- [x] Step 1. Data pipeline and EDA.
- [x] Step 2. Oracle v1: frozen ESM-2 (650M) embeddings, Ridge and MLP heads. Test
  Spearman about 0.62. Ranking mostly survives the distribution shift. Absolute
  calibration does not. RMSE roughly doubles on the far set. That gap is what week 6
  is about.
- [ ] Step 3. Fine-tune ESM end to end, add uncertainty (ensemble or MC-dropout),
  check calibration.
- [ ] Step 4. MVP closed loop: directed-evolution proposer plus a held-out
  ground-truth evaluator.
- [ ] Step 5. Smarter proposals: ESM masked-LM-guided mutations vs random evolution.
- [ ] Step 6. Reward hacking, the centerpiece. Measure the oracle-vs-truth gap on
  designed sequences, show the optimizer exploiting it, fix it with a trust region
  or uncertainty penalty.
- [ ] Step 7. Benchmark against FLEXS or AdaLead.
- [ ] Step 8. Package it up, short writeup, maybe a Gradio demo.
