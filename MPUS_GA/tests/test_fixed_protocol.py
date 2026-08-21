import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MPUS_GA.protocols import FIXED_UDA_PROTOCOL  # noqa: E402


def test_fixed_protocol_contract() -> None:
    assert FIXED_UDA_PROTOCOL.name == "cross_dataset_transductive_fixed1000"
    assert FIXED_UDA_PROTOCOL.training_iterations == 1000
    assert FIXED_UDA_PROTOCOL.checkpoint_selection == "none_final_iteration"
    assert FIXED_UDA_PROTOCOL.target_evaluations == 1
    FIXED_UDA_PROTOCOL.validate_fold(
        target_trials=45,
        expected_target_trials=45,
        training_iterations=1000,
    )
    FIXED_UDA_PROTOCOL.validate_fold(
        target_trials=80,
        expected_target_trials=80,
        training_iterations=1000,
    )


@pytest.mark.parametrize(
    ("target_trials", "expected_target_trials", "training_iterations"),
    ((44, 45, 1000), (81, 80, 1000), (45, 45, 999)),
)
def test_fixed_protocol_rejects_mismatch(
    target_trials: int,
    expected_target_trials: int,
    training_iterations: int,
) -> None:
    with pytest.raises(ValueError, match="protocol mismatch"):
        FIXED_UDA_PROTOCOL.validate_fold(
            target_trials=target_trials,
            expected_target_trials=expected_target_trials,
            training_iterations=training_iterations,
        )
