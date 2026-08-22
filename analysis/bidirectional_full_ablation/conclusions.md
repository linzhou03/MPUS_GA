# Statistical conclusions

This analysis covers all 864 fixed-final folds. The independent unit is the
target subject: seeds 42, 43, and 44 are averaged within each subject before
paired testing (16 subjects in direction A and 20 in direction B). Holm
correction is applied across the 16 planned comparisons separately for each
metric.

## Confirmatory findings after Holm correction

The clearest evidence favors multiscale fusion rather than any individual
specialized component:

- A3 versus A1 improves Accuracy by 6.57 pp (95% bootstrap CI 4.31 to 8.89,
  Holm p=0.0116).
- B3 versus B0 improves Accuracy by 4.75 pp (3.54 to 5.94, Holm p=0.00313)
  and Macro-F1 by 3.07 pp (1.97 to 4.19, Holm p=0.000427).
- B3 versus B2 improves Macro-F1 by 2.80 pp (1.36 to 4.25, Holm p=0.0215).

No primary-metric comparison of Main against the uniform fusion, no-pyramid,
no-boundary, or no-suppression ablations survives Holm correction. The current
data therefore support the general multiscale claim more strongly than the
claim that the complete specialized architecture is superior.

## Class-distribution changes

Several class-specific shifts remain significant after per-metric Holm
correction:

- A3 versus A0/A1 raises negative recall by 11.96/13.81 pp and raises the
  negative prediction rate by 9.07/11.20 pp.
- B3 versus B0 raises negative recall by 10.45 pp, lowers the positive
  prediction rate by 5.92 pp, and raises the negative prediction rate by
  8.40 pp.
- Source-excess suppression raises the negative prediction rate by 10.79 pp in
  A5 versus A6 and by 3.27 pp in B5 versus B6.
- B_main versus B4 raises the neutral prediction rate by 3.77 pp (Holm
  p=0.0310). This is direct evidence that the feature-pyramid path aggravates
  the reverse-direction neutral attraction even though its effects on the
  headline metrics are not significant.

## Exploratory effects worth retaining as hypotheses

- A_main versus A5 raises mean worst-class recall by 3.32 pp (bootstrap CI
  0.15 to 6.71), but the Wilcoxon and Holm tests are not significant. The
  boundary-attractor mechanism remains an A-direction hypothesis rather than
  a confirmed component.
- B_main versus B3 raises mean worst-class recall by 1.98 pp (0.52 to 3.51;
  raw Wilcoxon p=0.0289, Holm p=0.376). Class-conditional fusion may improve
  reverse-direction worst-class behavior, but another full confirmation is
  required.
- B4 provides the strongest practical reverse-direction trade-off. Adding the
  feature pyramid changes Accuracy by -0.85 pp, Macro-F1 by -0.50 pp, and
  recall-gap reduction by -3.75 pp, while significantly increasing neutral
  prediction frequency.

Bootstrap intervals test the paired mean change, whereas Wilcoxon tests use
paired ranks; they can disagree for heterogeneous subject effects. Claims of a
confirmed component should use the Holm-adjusted paired test rather than an
uncorrected interval alone.
