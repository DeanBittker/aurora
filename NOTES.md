# Notes

Running log of the modeling work. Newest step at the bottom. This is the reasoning
and the results, not the how-to. Setup and commands live in the READMEs.

## The setup everything depends on

GFP fluorescence, 54k point mutants of avGFP, label is `log_fluorescence`. The split
is by edit distance from wild-type: train and valid are 3 mutations or fewer, test is
4 or more. Train is about 82% bright, test about 32%. So the model learns a bright
local neighborhood and gets graded on a far, mostly-dark region it never saw. This is
an extrapolation test on purpose. Never reshuffle train and test together. It leaks
near-duplicate variants across the boundary and inflates every number.

Metric is test-set Spearman. Rank correlation, not absolute error, because the
oracle's job is to rank candidate sequences and pick the best, not to nail exact
brightness. Protocol: train on train, select on valid, report test. Test does not get
touched until the final number.

## Step 2. Oracle v1: frozen ESM-2 plus a head

Idea: borrow a pretrained protein language model instead of training from scratch. Run
every sequence through ESM-2 650M, frozen, mean-pool to one vector, cache the vectors,
then train a cheap head on them. Embed once, reuse forever.

Two heads on purpose. Ridge as an honest linear baseline, a small MLP as the learned
head. If Ridge already ranks well, the signal is in the embeddings and the head barely
matters.

Results:
- test Spearman 0.61 (Ridge), 0.63 (MLP).
- valid Spearman about 0.71 to 0.73. So the drop from valid (in-distribution) to test
  (out-of-distribution) is about 0.10. Ranking mostly survives the shift.
- RMSE tells a sharper story. valid RMSE about 0.44 to 0.47, test RMSE about 0.84 to
  0.96. Roughly double. The model ranks the far set about right but its absolute
  predictions there are badly off, biased high toward the bright training mean.
- MLP beats Ridge by about 0.01 on test but about 0.05 on train. So the embeddings
  carry the signal, the head is a minor lever, and the extra nonlinearity mostly fits
  the training neighborhood without transferring.

Takeaway that sets up everything later: the oracle ranks fine but is confidently
miscalibrated in the region it never saw. That gap is exactly what a naive optimizer
would exploit.

## Step 3. Oracle v2: fine-tune plus uncertainty

Two upgrades aimed at v1's weakness.

Fine-tune the backbone instead of freezing it. ESM-2 150M, full fine-tune, the size
that actually trains on the Mac. Different learning rates, small on the backbone
(2e-5), larger on the head (1e-3).

Add uncertainty with a deep ensemble. Three fine-tunes from different seeds, and the
spread across them is the uncertainty. This is epistemic uncertainty (the model's own
ignorance) on purpose, not aleatoric (noise in the data). Epistemic is what flags the
out-of-distribution region and what the step 6 optimizer will try to game.

Results:
- test Spearman 0.67, up from v1's 0.62. Fine-tuned 150M beat frozen 650M. A smaller
  model that adapts to the task beats a bigger one held fixed.
- test RMSE 0.51, down from v1's 0.84 to 0.96. Absolute predictions on the far set are
  much less biased now. Fine-tuning fixed a lot of the OOD bias, not just the ranking.
- the extrapolation gap is still there. valid Spearman 0.77, test 0.67, about a 0.10
  drop. Fine-tuning lifts the whole curve but does not erase the shift. Expected.
- calibration, the point of this step:
  - mean ensemble std is higher on test (0.187) than valid (0.141). The ensemble is
    more unsure out in the dark region it never trained on. The oracle knows where it
    is ignorant. This is the signal step 6 needs.
  - err_vs_std is positive and higher on test (0.52) than valid (0.39). Where the
    ensemble is uncertain it tends to be more wrong, and that link is stronger OOD.
  - the intervals are overconfident. cov68 is 0.31 to 0.40 against an ideal of 0.68,
    cov90 is 0.61 to 0.64 against 0.90. A K=3 ensemble underestimates the size of the
    uncertainty. The direction and ranking are right, the magnitude needs recalibration
    (temperature scaling or conformal) if we ever need honest intervals.

Takeaway: fine-tuning helped the headline and the OOD bias, and the ensemble
uncertainty rises out of distribution and tracks error. That rise is the lever step 6
will use. The one caveat is overconfident intervals, fixable later.

## Where this is going

v1 found that the oracle is confidently wrong out of distribution. v2 made it more
accurate and gave it an uncertainty that rises where it should. Step 4 puts the oracle
in a closed loop with a proposer. Step 6 is the centerpiece: show an optimizer
exploiting the oracle's blind spots, then fix it with a trust region or an uncertainty
penalty built on exactly the uncertainty v2 is measuring now.
