from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MPUS_GA.preprocessing.preprocess import (  # noqa: E402
    _rebuild_artifact_manifests,
    _write_processing_metadata,
)


def _write_artifact(path: Path, scale: int) -> None:
    count = 4 // scale
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.zeros((count, 62, 5), dtype=np.float32),
        label_3class=np.zeros(count, dtype=np.int8),
        subject_id=np.ones(count, dtype=np.int16),
        session_id=np.ones(count, dtype=np.int8),
        channel_names=np.asarray([f"CH{i:02d}" for i in range(62)]),
        source_file=np.asarray("1_20160518.mat"),
        window_seconds=np.asarray(scale, dtype=np.float32),
    )


def test_metadata_is_full_inventory_not_last_run_scope(tmp_path: Path) -> None:
    output = tmp_path / "data_processed"
    for scale in (1, 2, 4):
        _write_artifact(
            output
            / "seed_iv"
            / f"window_{scale}s"
            / "subject_01_session_1.npz",
            scale,
        )

    manifests = _rebuild_artifact_manifests(
        output,
        {"seed-iv": {"raw_dir": "/dataset/SEED_IV", "excluded_files": []}},
    )
    manifest = manifests["seed-iv"]
    assert manifest["inventory_scope"] == "all_existing_artifacts"
    assert manifest["artifact_count"] == 3
    assert manifest["subject_count"] == 1
    assert manifest["subject_session_count"] == 1
    assert manifest["sample_counts"] == {"1": 4, "2": 2, "4": 1}

    summary = _write_processing_metadata(
        output,
        tmp_path,
        "test-run",
        manifests,
        tmp_path / "Channel Order.xlsx",
    )
    assert summary["totals"] == {"artifacts": 3, "subject_sessions": 1}
    config = (output / "processing_config.json").read_text(encoding="utf-8")
    assert '"record_type": "stable_pipeline_configuration"' in config
    assert '"subjects"' not in config
