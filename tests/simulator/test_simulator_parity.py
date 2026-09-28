from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from sts1_llm_policy.env.action_schema import ActionType, CanonicalAction
from sts1_llm_policy.env.simulator_env import SimulatorCombatEndedError
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    CombatState,
    MonsterState,
    PlayerState,
    PowerState,
)
from sts1_llm_policy.eval.simulator_parity import (
    SimulatorParityConfig,
    canonical_json_sha256,
    evaluate_simulator_parity,
    load_parity_config,
    load_parity_fixtures,
    load_supported_mechanics,
    project_canonical_state,
    seed_bit_pattern_u64,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _state(*, terminal: bool = False, seed: int = 41) -> CanonicalState:
    card = CardState(
        card_id="Strike_R",
        name="Strike",
        card_type="ATTACK",
        cost=1,
        upgrades=0,
        description="Deal 99 dynamic damage." if terminal else "Deal 6 damage.",
        exhausts=False,
        ethereal=False,
        has_target=True,
        is_playable=not terminal,
        uuid="source-private",
    )
    return CanonicalState(
        seed=seed,
        character="IRONCLAD",
        ascension_level=0,
        act=1,
        floor=1,
        combat=CombatState(
            turn=1,
            player=PlayerState(80, 80, 0, 3),
            monsters=(
                MonsterState(
                    monster_id="Cultist",
                    name="Cultist",
                    current_hp=0 if terminal else 6,
                    max_hp=48,
                    block=0,
                    intent="BUFF",
                    move_id=3,
                    move_hits=1,
                    move_base_damage=-1,
                    move_adjusted_damage=-1,
                    powers=(
                        PowerState("Strength", "Strength", 3),
                        PowerState("Ritual", "Ritual", 3),
                    ),
                    is_gone=terminal,
                ),
            ),
            hand=() if terminal else (card,),
            draw_pile=(),
            discard_pile=(card,) if terminal else (),
            exhaust_pile=(),
        ),
    )


class _TerminalEnv:
    def __init__(self) -> None:
        self.state = _state()
        self.action = CanonicalAction(
            "ACTION_0",
            ActionType.PLAY_CARD,
            hand_index=0,
            card_uuid="sim-private",
            card_name="Strike",
            target_index=0,
        )

    def reset(
        self,
        scenario_id: str,
        seed: int,
        *,
        ascension: int = 0,
        player_current_hp: int | None = None,
    ) -> CanonicalState:
        self.state = _state()
        return self.state

    def legal_actions(self) -> tuple[CanonicalAction, ...]:
        return (self.action,)

    def step(self, action: CanonicalAction) -> CanonicalState:
        raise SimulatorCombatEndedError(
            {"source": "simulator"},
            terminal_state=_state(terminal=True),
            reward=1.0,
            outcome="PLAYER_VICTORY",
        )

    def close(self) -> None:
        pass


class SimulatorParityContractTest(unittest.TestCase):
    def test_normalizes_signed_java_seed_to_uint64_bit_pattern(self) -> None:
        signed_seed = -4887420107870246919
        unsigned_seed = 13559323965839304697

        self.assertEqual(seed_bit_pattern_u64(signed_seed), unsigned_seed)
        self.assertEqual(seed_bit_pattern_u64(unsigned_seed), unsigned_seed)

        state = _state(seed=signed_seed)
        self.assertEqual(project_canonical_state(state)["seed"], unsigned_seed)

    def test_projector_ignores_private_identity_description_and_power_order(self) -> None:
        first = asdict(_state())
        second = asdict(_state())
        second["combat"]["hand"][0]["uuid"] = "different"
        second["combat"]["hand"][0]["description"] = "different display text"
        second["combat"]["monsters"][0]["powers"] = tuple(
            reversed(second["combat"]["monsters"][0]["powers"])
        )

        self.assertEqual(
            project_canonical_state(first),
            project_canonical_state(second),
        )

    def test_repository_fixtures_are_valid(self) -> None:
        config = load_parity_config(
            PROJECT_ROOT / "configs/runs/evaluation/sts_lightspeed_parity_v1.json",
            project_root=PROJECT_ROOT,
        )
        load_supported_mechanics(config.mechanics_manifest)
        self.assertTrue(load_parity_fixtures(config.fixture_path))

    def test_gate_passes_executable_fixture_but_blocks_missing_encounter_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads(
                (
                    PROJECT_ROOT
                    / "configs"
                    / "env"
                    / "supported_mechanics_v1.json"
                ).read_text(encoding="utf-8")
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            state = asdict(_state())
            action = asdict(_TerminalEnv().action)
            action["action_type"] = action["action_type"].value
            raw_snapshot = {"source": "real", "decision": 0}
            common = {
                "schema_version": "parity_fixture_v1",
                "step_index": 0,
                "source": {"episode_id": "test"},
                "real_raw_snapshot_sha256": canonical_json_sha256(raw_snapshot),
                "real_raw_snapshot": raw_snapshot,
                "canonical_input": state,
                "legal_actions": [action],
                "chosen_action": action,
                "expected_next": {
                    "done": True,
                    "terminal_outcome": "victory",
                    "player_current_hp": 80,
                    "state_fields": None,
                },
            }
            executable = {
                **common,
                "fixture_id": "cultist-test",
                "encounter": "cultist",
                "execution": {
                    "status": "executable",
                    "simulator_reset": {
                        "scenario_id": "cultist",
                        "seed": 41,
                        "ascension": 0,
                    },
                },
            }
            excluded = {
                **common,
                "fixture_id": "louse-test",
                "encounter": "two_louse",
                "execution": {
                    "status": "excluded",
                    "simulator_reset": None,
                    "blockers": ["arbitrary state reset unavailable"],
                },
            }
            fixture_path = root / "fixtures.jsonl"
            fixture_path.write_text(
                json.dumps(executable) + "\n" + json.dumps(excluded) + "\n",
                encoding="utf-8",
            )
            config = SimulatorParityConfig(
                gate_id="test",
                backend_revision=manifest["backend_revision"],
                mechanics_manifest=manifest_path,
                fixture_path=fixture_path,
                required_real_encounters=("cultist", "two_louse"),
                all_executable_transitions_match=True,
                all_required_real_encounters_executable=True,
                no_unresolved_supported_mechanic_mismatch=True,
            )

            report = evaluate_simulator_parity(config, env=_TerminalEnv())

        self.assertEqual(report["status"], "no_go")
        self.assertTrue(report["gates"]["executable_transition_parity"]["passed"])
        self.assertFalse(report["gates"]["required_real_encounter_coverage"]["passed"])
        self.assertEqual(
            report["gates"]["required_real_encounter_coverage"][
                "missing_executable_encounters"
            ],
            ["two_louse"],
        )


if __name__ == "__main__":
    unittest.main()
