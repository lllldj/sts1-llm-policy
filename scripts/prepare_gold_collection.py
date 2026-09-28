"""Prepare an explicit GOLD collection config and source-selection record."""
import argparse
from pathlib import Path

from sts1_llm_policy.data.gold.preparation import prepare_collection
from sts1_llm_policy.data.gold.state_sampling import VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config", required=True)
    parser.add_argument("--route-count", type=int, default=80)
    parser.add_argument("--groups-per-route", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026090906)
    parser.add_argument("--weak-states", type=int, default=1)
    parser.add_argument("--strong-states", type=int, default=2)
    parser.add_argument("--elite-states", type=int, default=2)
    parser.add_argument("--boss-states", type=int, default=4)
    parser.add_argument("--sampling", choices=[VERSION, "uniform_decisions_v1"], default=VERSION)
    parser.add_argument("--turn-start-weight", type=float, default=.4)
    parser.add_argument("--event-weight", type=float, default=.3)
    parser.add_argument("--random-weight", type=float, default=.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--exclusions", help="Source-report-bound JSON containing excluded tuning routes")
    parser.add_argument("--training-candidates", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--hp-tolerance", type=float, default=5.0)
    parser.add_argument("--trace-observation", choices=["full", "reconstruct"], default="full")
    parser.add_argument("--compact-storage", action="store_true", help="Verify each finished state, then retain terminal rows, action sequences and sparse full traces")
    parser.add_argument("--route-progress", action="store_true")
    args = parser.parse_args()
    selection = prepare_collection(
        Path(__file__).resolve().parents[1], template_config=args.template_config,
        config_output=args.config_output, output=args.output,
        route_count=args.route_count, groups_per_route=args.groups_per_route, seed=args.seed,
        states_per_combat={"normal_weak": args.weak_states, "normal_strong": args.strong_states,
                           "elite": args.elite_states, "boss": args.boss_states},
        sampling=args.sampling, sampling_weights={"turn_start": args.turn_start_weight,
                                                 "event": args.event_weight, "random": args.random_weight},
        workers=args.workers, exclusions=args.exclusions,
        training_candidates=args.training_candidates, adaptive=args.adaptive,
        hp_tolerance=args.hp_tolerance, trace_observation=args.trace_observation,
        compact_storage=args.compact_storage, route_progress=args.route_progress,
    )
    print({k: selection[k] for k in ("status", "state_count", "states_by_family", "model_root_count", "planned_executions", "configuration")})


if __name__ == "__main__":
    main()
