# Processed data destination

The preprocessing pipeline writes SEED-IV, SEED-V, and SEED-VII 1 s, 2 s, and
4 s DE files here. Large NPZ artifacts are intentionally ignored by Git.
Dataset manifests describe every NPZ currently present, while immutable
per-invocation records are stored in `processing_runs/`.
