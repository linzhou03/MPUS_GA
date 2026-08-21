from pathlib import Path
import sys

import numpy as np
from openpyxl import Workbook
from scipy.io import savemat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from MPUS_GA.preprocessing.dataset_metadata import (
    EMOTION_TO_ORIGINAL_SEED_IV,
    EMOTION_TO_THREE_CLASS,
    SEED_IV_EMOTIONS,
    load_channel_names,
    parse_seed_iv_filename,
)
from MPUS_GA.preprocessing.preprocess import _seed_iv_trials, build_parser


def test_seed_iv_labels_are_balanced_and_map_to_three_classes() -> None:
    for emotions in SEED_IV_EMOTIONS.values():
        assert len(emotions) == 24
        assert {emotion: emotions.count(emotion) for emotion in set(emotions)} == {
            "neutral": 6,
            "sad": 6,
            "fear": 6,
            "happy": 6,
        }
    assert EMOTION_TO_ORIGINAL_SEED_IV == {
        "neutral": 0,
        "sad": 1,
        "fear": 2,
        "happy": 3,
    }
    assert EMOTION_TO_THREE_CLASS["happy"] == 0
    assert EMOTION_TO_THREE_CLASS["neutral"] == 1
    assert EMOTION_TO_THREE_CLASS["sad"] == 2
    assert EMOTION_TO_THREE_CLASS["fear"] == 2


def test_seed_iv_filename_uses_parent_as_session(tmp_path: Path) -> None:
    path = tmp_path / "2" / "15_20150514.mat"
    assert parse_seed_iv_filename(path) == (15, 2)


def test_seed_iv_loader_orders_trials_by_numeric_suffix(tmp_path: Path) -> None:
    session_dir = tmp_path / "3"
    session_dir.mkdir()
    path = session_dir / "2_20151012.mat"
    payload = {
        f"subject_prefix_eeg{trial}": np.full(
            (62, 801), trial, dtype=np.float64
        )
        for trial in reversed(range(1, 25))
    }
    savemat(path, payload)
    channels = tuple(f"CH{index:02d}" for index in range(62))

    trials, observed_channels = _seed_iv_trials(path, channels)

    assert observed_channels == channels
    assert [trial.session_trial for trial in trials] == list(range(1, 25))
    assert [trial.global_trial for trial in trials] == list(range(49, 73))
    assert [trial.emotion for trial in trials] == list(SEED_IV_EMOTIONS[3])
    assert [float(trial.data[0, 0]) for trial in trials] == list(
        map(float, range(1, 25))
    )


def test_seed_iv_channel_order_and_cli(tmp_path: Path) -> None:
    path = tmp_path / "Channel Order.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    for index in range(62):
        sheet.cell(index + 1, 1, f"CH{index:02d}")
    workbook.save(path)
    workbook.close()

    assert load_channel_names(path) == tuple(f"CH{index:02d}" for index in range(62))
    args = build_parser().parse_args(["--datasets", "seed-iv"])
    assert args.datasets == ["seed-iv"]
