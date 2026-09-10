# R2 class-conditional multiscale co-teaching

Pre-change checkpoint: `571022a`. First subgroup version: `33601c7`.

Select `--method r2_coteaching --experiment A` (A–F). This profile derives from
R2. It retains the original encoder, MSAD, learned multiscale fusion, source
classification, domain loss, source-only frozen R2 memory, and original 0.1
centroid alignment coefficient. The earlier `r2_subgroup` profile stays available.

## What changes

An EMA teacher supplies source/target independent scale features and probabilities.
Source coarse labels and unlabeled target trial identities are the only policy
inputs. All state starts fresh for every subject/seed. No fine emotion labels,
target truth, validation, early stopping or target-selected parameters are used.
Student evaluation remains at iteration 1000.

**Class-conditional reliability.** Teacher source predictions update a confusion
matrix EMA per scale (momentum 0.95). Per-class precision and recall yield a
bounded F1 reliability score (floor 0.10). This is training-data evidence, not a
calibrated estimate of target accuracy. These scores are published every 50 steps.

**Two peers teach the remaining scale.** For scale s, its own probability is
excluded from its soft supervision target. The other two scales' probabilities
are averaged with the source reliability scores for each class, then normalized.
They must agree on a coarse class, exceed the 0.60 confidence floor, and retain
stable predictions across two refreshes (JSD <= 0.10). The current peer predictions
must still agree with the published target when used. A weak third scale may
receive supervision without vetoing the two peers. Class/scale terms use weighted
KL, averaged equally over available class/scale terms. Coefficient: 0.05 with a
cosine ramp from steps 300–600. Features are not forced to be identical across scales.

**Classwise target selection.** A separate consensus uses only scales voting for
the majority coarse label (at least two). A target trial must exceed confidence
0.60 and remain stable across two refreshes. Within each predicted class, keep the
highest-confidence 50% of eligible unique trials; deterministic trial-key ordering
breaks ties. This selects a proportion within each class, not equal numbers or a
uniform target prior. Missing/unreliable classes remain empty. Repeated samples in
one interval do not count as repeated stability observations. Unseen stale evidence
cannot become stable merely by staying in memory. The target cache holds at most
512 trials with age <= 200 steps; decisions are updated only at 50-step boundaries.

Only selected trials whose own scale prediction agrees with the consensus enter
that scale's target subgroup bank. A weak scale can therefore receive peer KL
before its features become trustworthy enough for subgroup alignment. Source
subgroup features remain grouped by true coarse source labels.

**Partial matching with R2 fallback.** K <= 4, minimum three distinct trial IDs,
cosine >= 0.80, mutual-nearest margin >= 0.02, assignment cosine >= 0.50 and
contrast temperature 0.10 remain as in the first version. Matching is partial and
same-class unmatched prototypes are never contrastive negatives. Contrast weight
is 0.05. There is no OT.

For each scale/class, at least three target trial IDs must remain in matched
subgroups across consecutive refreshes. Both domains must also have reliable
other-class negative prototypes. The readiness increases 0 -> 0.5 -> 1 over two
successful overlap checks; losing evidence resets it to zero. R2's centroid
attraction is multiplied by `1 - readiness * ramp` for that scale/class; selective
contrast uses `readiness * ramp`. The original centroid denominator is retained
so removing a term does not amplify the remaining terms. At full readiness,
unmatched same-class subgroups receive no centroid or contrastive attraction.
During the transition the original centroid contribution tapers down.

With no reliable subgroup evidence, the original centroid loss is preserved.
Peer teaching can still operate independently if its own evidence is reliable.
With neither reliable peer evidence nor reliable subgroup evidence, tests verify
that a complete optimizer step matches R2. This structural fallback is not a
promise that classification accuracy cannot decrease. Disabling contrast with
`--subgroup-weight 0` preserves the original centroid term.

The current step uses the previous published evidence and prototype snapshot.
After its optimizer step, teacher parameters are EMA-updated, observations are
collected, and any scheduled refresh is published for subsequent steps.

## Run on csu

```bash
cd /home/gzw/projects
/home/gzw/miniforge3/envs/BCI/bin/python -u \
  -m MPUS_GA.scripts.run_r2_coteaching_suite \
  --run-name r2_coteaching_six_direction_s43_s42_20260910 \
  --gpus 0 \
  --random-seeds 43 42 \
  --target-subjects all
```

The worker detaches automatically. GPU 0: A -> B -> C -> D -> E -> F.
There is exactly one sequential queue, with at most one training child. Each direction finishes
seed 43 before seed 42. The launcher resolves physical GPU UUIDs and each child
uses its isolated `cuda:0`. Six directions, 204 folds, 1000 steps per fold.

| Direction | Source | Target | Target subjects | Folds |
|---|---|---|---:|---:|
| A | SEED-VII | SEED-V | 16 | 32 |
| B | SEED-V | SEED-VII | 20 | 40 |
| C | SEED-IV | SEED-V | 16 | 32 |
| D | SEED-V | SEED-IV | 15 | 30 |
| E | SEED-IV | SEED-VII | 20 | 40 |
| F | SEED-VII | SEED-IV | 15 | 30 |

All numeric policy settings are configurable through `--subgroup-*`, including
`--subgroup-confidence`, `--subgroup-keep-fraction`, `--subgroup-stable-refreshes`,
`--subgroup-temporal-jsd`, `--subgroup-teaching-weight` and
`--subgroup-replacement-refreshes`. See `--help`. Add `--dry-run` to inspect commands
without starting training. Use a new run name after any code/data/config change.

- Results: `results_<run-name>/<A-F>/seed_<seed>_subject_<NN>.json` and `summary.csv`.
- Logs: `logs/<run-name>/<A-F>.log` and `logs/<run-name>/suite.log`.
- Manifest includes full policy configuration, source hashes, data identities and
  GPU mapping. A lock prevents a duplicate worker for the same result root.
- Metrics remain ACC, BA, Macro-F1 and class recalls.
- JSON `subgroup_alignment.evidence_history`: per-class counts after majority,
  confidence, temporal stability and class-rank selection; source reliability.
- `transition_history`: matched trial support and per-scale/class readiness.
- `training_trace`: peer KL, selective contrast, active pair counts, replacement
  strength, snapshot step and the actual remaining centroid loss.
- Console progress includes `proto`, `sub`, `match`, `peer`.

## Tests

`python -m MPUS_GA.tests.test_multiscale_coteaching` checks peer independence,
classwise selection, temporal/stale evidence, source reliability, centroid value
and gradient fallback, complete-step R2 equivalence, active KL/contrast gradients,
label isolation, all six direction configurations and seed order. Existing
training and first-version subgroup regression tests must also pass.

The independent real-data smoke fold uses C, seed 43, target subject 1, all source
subjects and the normal 1000-step protocol. It writes to a separate temporary
result root on csu. Only runtime/participation diagnostics are inspected; target
accuracy is not used to select policy parameters. The formal six-direction suite
is launched by the user with the command above.

Validation completed on csu (2026-09-10): 60 existing regression tests and 10 new
behavior tests passed. The isolated C/seed43/subject01 GPU-0 fold completed all
1000 steps successfully, recording 7,468 active peer-teaching relations and
11,919 contrastive positives (50,799 negatives). Peak allocated CUDA memory was
5,604 MiB. This verifies execution and participation, not an accuracy gain. Its
artifacts are at `/tmp/mpus_coteaching_smoke_20260910_y3oqeq3c/` on csu.
A mocked worker test verifies exact A-B-C-D-E-F order on a single GPU and halts
subsequent directions after a failed child. The final command uses GPU 0 only.

## xju: GPU 1, reverse direction order

The same training method can run F -> E -> D -> C -> B -> A with `--reverse`.
Only the direction order changes; each direction still runs seed 43 then seed 42.
The detached worker receives this flag and the manifest records the reversed GPU
queue. Use an independent xju run name:

```bash
cd /home/gzw/projects
/home/gzw/anaconda3/envs/BCI/bin/python -u \
  -m MPUS_GA.scripts.run_r2_coteaching_suite \
  --run-name r2_coteaching_xju_reverse_s43_s42_20260910 \
  --gpus 1 --reverse \
  --random-seeds 43 42 --target-subjects all
```

The scheduler isolates physical GPU 1 by UUID and the child uses `cuda:0` inside
that isolated view. Logs and results use the same layout documented above under
this independent run name. No extra nohup is necessary.
