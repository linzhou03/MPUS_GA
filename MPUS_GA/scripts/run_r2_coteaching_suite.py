"""Run class-conditional multiscale co-teaching A-F sequentially on GPU 0."""
from MPUS_GA.trial_temporal.multiscale_coteaching import CoTeachingConfig
from .run_r2_subgroup_suite import main as run_suite


def main():
    run_suite(method="r2_coteaching", config_type=CoTeachingConfig,
              module="MPUS_GA.scripts.run_r2_coteaching_suite", default_gpus=["0"])


if __name__ == "__main__":
    main()
