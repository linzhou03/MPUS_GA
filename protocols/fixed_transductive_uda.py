"""Fixed transductive-UDA protocol for fair model comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FixedTransductiveUDAProtocol:
    name: str = "cross_dataset_transductive_fixed1000"
    training_iterations: int = 1000
    checkpoint_selection: str = "none_final_iteration"
    target_evaluations: int = 1

    def validate_fold(
        self,
        *,
        target_trials: int,
        expected_target_trials: int,
        training_iterations: int,
    ) -> None:
        observed = (target_trials, training_iterations)
        expected = (expected_target_trials, self.training_iterations)
        if observed != expected:
            raise ValueError(
                "Fixed transductive-UDA protocol mismatch: "
                f"observed={observed}, expected={expected}"
            )

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


FIXED_UDA_PROTOCOL = FixedTransductiveUDAProtocol()
