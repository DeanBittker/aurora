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

Two heads on purpose. Ridge as a linear baseline, a small MLP as the learned
head. If Ridge already ranks well, the signal is in the embeddings and the head barely matters.

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

## Step 3. Oracle v2: fine-tune plus uncertainty (in progress)

Two upgrades aimed at v1's weakness.

Fine-tune the backbone instead of freezing it. ESM-2 150M, full fine-tune, the size
that actually trains on the Mac. Different learning rates, small on the backbone
(2e-5), larger on the head (1e-3). This should lift the headline number.

Add uncertainty with a deep ensemble. Three fine-tunes from different seeds, and the
spread across them is the uncertainty. This is epistemic uncertainty (the model's own
ignorance) on purpose, not aleatoric (noise in the data). Epistemic is what flags the
out-of-distribution region and what the step 6 optimizer will try to game.

The question this step answers: does the uncertainty rise on the test set? If the
ensemble is more unsure out in the dark region it never trained on, the oracle knows
where it is ignorant, and we can use that later to keep the optimizer honest.

Results: pending the full run. Fill in test Spearman vs the v1 0.62 baseline, and the
mean ensemble std on valid vs test (the number that says whether uncertainty rises out
of distribution).

## Where this is going

v1 found that the oracle is confidently wrong out of distribution. v2 is about making
it know when it is out of its depth. Step 4 puts the oracle in a closed loop with a
proposer. Step 6 is the centerpiece: show an optimizer exploiting the oracle's blind
spots, then fix it with a trust region or an uncertainty penalty built on exactly the
uncertainty v2 is measuring now.
