# CBST implementation validation, 2026-09-22

- Local core tests: 5 passed; previous uncertainty core regression tests also passed (9 combined).
- xju BCI: 7 CBST core/integration tests passed, including real train_step,
  prototype disablement, mask-selected CE, and passive evaluation RNG restoration.
- xju real-data B/subject 01/seed 43: three forced updates at 300,350,1000;
  fixed-final reporting passed, saved checkpoint iteration 3 in the isolated probe.
- csu real-data same subject/seed: three forced updates and three evaluations;
  selected update 2; selected prediction confusion matched the reported result.
  Final model saved separately. No probe score is a formal experiment score.
- Both probes produced matching CBST training scalars: first active round
  thresholds [0.4291602373,0.3915201128,0.3734878302], class support [59,19,2],
  selected class counts [11,3,1], selected CE 0.9717083573. Next round p=.25.
- Peak tensor allocation in each probe approximately 5181.50 MiB; checkpointing
  and window chunking disabled, no model/batch reduction.
- Old MPUS training/queue processes were absent on both hosts before deployment.
- Runtime code release on both: /home/gzw/projects/MPUS_GA_releases/cbst_20260922_s43_v1/MPUS_GA.
- Runs: cbst_xju_fixed_20260922_s43_v1 and cbst_csu_best_20260922_s43_v1.
- Each run has six direction/seed jobs, 102 subject folds, GPU 0 A/C/E,
  GPU 1 B/D/F, seed 43, 1-second relay interval. No ablations.
