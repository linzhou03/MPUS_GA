# R2 + latent subgroup selective contrast

This profile is selected explicitly with `--method r2_subgroup --experiment A` (A–F).
It derives from `A_R2`; the existing R2 and later A–F profiles are unchanged.
The pre-change Git checkpoint is `42dfe5b`; original R2 is
`c7cc997f7af00d04f451dda3da37c24230bfd4ae`.

## Model and loss

The 1s/2s/4s encoders, temporal MSAD, learned low-rank multiview fusion,
source classification, domain adversarial learning, and frozen source memory
retain their R2 settings. Separate subgroup banks operate on the three
**independent scale embeddings before fusion**. Their gradients train these
encoders jointly with the original fusion/classification objectives.

This profile **replaces the original whole-class centroid attraction** with
selective subgroup contrast. Keeping that attraction would continue pulling
unmatched subgroups together. In equations,

`L_new = L_R2 - lambda_proto * L_centroid + lambda_sub * ramp * L_sub_con`.

The existing centroid bank is retained for R2's reliability/fusion evidence;
its centroid alignment loss is disabled only in the new profile. Other R2
losses are unchanged. MPUS_GA R2 is not the original parent CAGA-SGA implementation;
this change does not introduce parent CAGA-SGA style/SGA losses into R2.

1. A frozen EMA teacher supplies features in both domains and coarse target
   pseudo-labels. Source grouping uses only true coarse source labels. Target
   batches containing `y` are rejected. Teacher inference runs in eval mode.
2. Target evidence requires confidence >= 0.90, multiscale JSD <= 0.15 and at
   least two agreeing scales. Only agreeing scales enter that trial's bank row.
   No requirement that all classes occur in a batch is imposed.
3. Memory is keyed by `(source-domain-index, subject, session, trial)`, with
   separate source/target dictionaries, at most 256 unique trials per class,
   and maximum age 200 iterations. Resampling a trial cannot inflate support.
   A new pseudo-label replaces the previous one; unreliable new evidence removes it.
4. After iteration 300, spherical clustering is recomputed every 50 iterations.
   K=4 is an upper bound, not a fine-emotion count. It is reduced when evidence
   is scarce; a cluster requires at least three distinct trials. Classes and
   domains can have different effective K, including zero.
5. Within each scale and coarse class, mutual nearest prototypes are matched
   only with cosine >= 0.80 and nearest/second-nearest margin >= 0.02 on both
   sides. Matching is partial; no OT or forced complete correspondence is used.
6. A student sample is assigned using its detached teacher feature. If its
   subgroup has a reliable cross-domain match and assignment cosine >= 0.50,
   the matched opposite-domain prototype is its positive. Only valid
   **different-coarse-class** opposite-domain prototypes enter the denominator.
   Unmatched same-class relations are ignored, never used as negatives.
7. Contrast is symmetric (source→target and target→source), weighted by teacher
   confidence, prototype quality and matching cosine, then averaged equally
   over available class/scale/direction terms. No positive or no different-class
   negative yields differentiable zero. Temperature=0.10, weight=0.05;
   cosine ramp is 0 at iteration 300 and 1 at 600.

Each step first uses the previously published prototype snapshot. Only after
the student optimizer step are the teacher EMA and memory updated; a newly
refreshed snapshot can first contribute on the next step. EMA momentum is 0.99
with startup bias correction. The subgroup bank is separate from R2's frozen
source memory. Every target subject and seed starts a fresh model/teacher/bank.

All numeric settings above are configurable using `--subgroup-...`; run
`python -m MPUS_GA.scripts.run_r2_subgroup_suite --help` for the full list.
Defaults are fixed in advance and must not be selected using target truth.
Latent subgroups may reflect subjects, sessions or noise rather than emotions;
an unmatched subgroup cannot be interpreted as anger without post-training
diagnostics. Gains, especially balanced accuracy gains, require experiments.

## Protocol and scheduling

| Direction | Source | Target | Target subjects | Folds for seeds 43, 42 |
|---|---|---|---:|---:|
| A | SEED-VII | SEED-V | 16 | 32 |
| B | SEED-V | SEED-VII | 20 | 40 |
| C | SEED-IV | SEED-V | 16 | 32 |
| D | SEED-V | SEED-IV | 15 | 30 |
| E | SEED-IV | SEED-VII | 20 | 40 |
| F | SEED-VII | SEED-IV | 15 | 30 |

All directions use Positive / Neutral / Negative. Each fold uses all source
subjects and one unlabeled target subject for exactly 1000 iterations, without
validation or early stopping; the student is evaluated at the end. Target truth
is not used in subgroup learning, matching, model selection or tuning.
Existing accuracy, macro F1 and balanced accuracy reporting remains available.

Two queues: GPU 0 runs A→C→E; GPU 1 runs B→D→F. Each direction completes seed 43
then seed 42. At most one training child runs on each GPU. There are 204 folds.
GPU UUID isolation maps each selected physical GPU to child `cuda:0`.

```bash
cd /home/gzw/projects
/home/gzw/miniforge3/envs/BCI/bin/python -u \
  -m MPUS_GA.scripts.run_r2_subgroup_suite \
  --run-name r2_subgroup_six_direction_s43_s42_20260910 \
  --gpus 0 1 --random-seeds 43 42 --target-subjects all
```

The launcher backgrounds the worker; no additional nohup is needed.
Add `--dry-run` to inspect all six commands without launching or creating outputs.
Use a new run name when changing parameters/code/data. A manifest and filesystem
lock prevent incompatible resumption and duplicate workers for the same run.
A failed direction stops that GPU's queue and is reported in suite.log.

- Results: `results_<run-name>/<A-F>/seed_<seed>_subject_<NN>.json`, plus `summary.csv`.
- Logs: `logs/<run-name>/<A-F>.log` and `logs/<run-name>/suite.log`.
- JSON `subgroup_alignment.refresh_history` records effective K, unique support,
  prototype cosine matrices, match edges, coverage and unmatched fraction by
  domain/scale/coarse class. Zero evidence is flagged separately.
- `training_trace[*].subgroup_alignment` records teacher acceptance by class,
  positive/negative sample-to-prototype pair counts and sample matching coverage.
  Cumulative pair counts cover all iterations. Pair counts count scale-specific
  relations, not unique samples. Unused padded prototypes have zero support and
  their cosine entries must be ignored when analyzing the matrices.

## Verification

Run `python -m MPUS_GA.tests.test_subgroup_alignment` for CPU tests of label
isolation, unique support, periodic/stale memory, partial matching, exclusion
of same-class negatives, gradient flow, R2 integration and six-direction CLI.
