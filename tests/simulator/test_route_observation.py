"""Route context must change policy behavior inputs without exposing hidden draws."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from sts1_llm_policy.env.route_context import public_route_context
from sts1_llm_policy.data.trajectory import (
    TrajectoryLogger,
    TerminalOutcome,
    iter_trajectory_records,
)
from sts1_llm_policy.env.route_context import CombatRouteContext, RouteOperation
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.simulator_adapter import from_sts_lightspeed_response
from sts1_llm_policy.workflows.continuous_plan import (
    build_route_lineage,
    load_scope,
)
from sts1_llm_policy.workflows.continuous_run import (
    _stateful_snapshot,
    _run_route,
)
from sts1_llm_policy.workflows.continuous_artifacts import (
    summarize_combat,
)
from sts1_llm_policy.policy.llm_policy import LLMPolicy

from tests.simulator.state_fixture import _response
from tests.simulator.native_support import NativeSelectionTestCase


def decision_fixture():
    response = _response()
    for monster in response["state"]["monsters"]:
        monster["move_history"] = []
    decision = from_sts_lightspeed_response(response)
    from sts1_llm_policy.env.monster_behavior import build_monster_behavior
    monsters = tuple(replace(m, behavior=build_monster_behavior(
        m, previous_move_id=None, living_monster_count=1, combat_turn=1,
    )) for m in decision.state.combat.monsters)
    return replace(decision, state=replace(decision.state, combat=replace(decision.state.combat, monsters=monsters)))


def route_steps():
    return [
        {"kind": "reset"},
        {"kind": "combat", "combat_index": 1, "encounter_family": "normal_weak",
         "scenario_id": "cultist", "combat_seed": 41, "policy_seed": 12345},
        {"kind": "reward", "choices": ["HIDDEN_CARD"]},
        {"kind": "heal", "amount": 24},
        {"kind": "upgrade", "evaluation_targets": [{"combat_seed": 23456}]},
        {"kind": "remove"},
        {"kind": "ordinary_relic", "relic_id": "HIDDEN_RELIC"},
        {"kind": "combat", "combat_index": 2, "encounter_family": "elite",
         "scenario_id": "HIDDEN_MONSTER", "combat_seed": 34567},
        {"kind": "combat", "combat_index": 3, "encounter_family": "boss",
         "scenario_id": "the_guardian", "combat_seed": 45678},
    ]


class RouteObservationTests(unittest.TestCase):
    def test_v7_public_memory_only_adds_known_draw_information(self):
        decision = decision_fixture()
        original = replace(decision.state, route_context=public_route_context(route_steps(), 1))
        original_text = serialize_observation(original, decision.actions, version="observation_v7")
        state = replace(original, known_draw_top=decision.state.combat.hand[:1])
        text = serialize_observation(state, decision.actions, version="observation_v7")
        self.assertIn("KNOWN_DRAW_TOP (next draw first): " + state.known_draw_top[0].name, text)
        prefix, suffix = text.split("\n\nKNOWN_DRAW_TOP (next draw first):", 1)
        _, original_actions = suffix.split("\nOnly this observed prefix is ordered; the remaining draw pile is unknown.", 1)
        self.assertEqual(prefix + original_actions, original_text)
        empty = serialize_observation(replace(state, known_draw_top=()), decision.actions, version="observation_v7")
        self.assertIn("KNOWN_DRAW_TOP (next draw first): none", empty)
        with self.assertRaisesRegex(ValueError, "requires explicit route context"):
            serialize_observation(replace(state, route_context=None), decision.actions, version="observation_v7")

    def test_public_projection_order_and_substitution(self):
        steps = route_steps()
        decision = decision_fixture()
        context = public_route_context(steps, 1)
        state = replace(decision.state, route_context=context)
        text = serialize_observation(state, decision.actions, version="observation_v7")
        self.assertIn("REMAINING_COMBATS_AFTER_CURRENT: 2", text)
        self.assertIn("VISIBLE_BOSS: the_guardian", text)
        for hidden in ("HIDDEN_CARD", "HIDDEN_RELIC", "HIDDEN_MONSTER", "12345", "23456", "34567", "45678"):
            self.assertNotIn(hidden, text)
        self.assertLess(text.index("1. CARD_PICK"), text.index("2. HEAL"))
        self.assertIn("3. UPGRADE", text)
        self.assertIn("4. REMOVE", text)
        self.assertIn("5. RANDOM_RELIC", text)
        self.assertNotIn("PRIMARY: Maximize the probability of winning this combat.", text)
        changed = deepcopy(steps)
        changed[3]["amount"] = 9
        changed[7]["encounter_family"] = "normal_strong"
        other = serialize_observation(
            replace(state, route_context=public_route_context(changed, 1)),
            decision.actions, version="observation_v7",
        )
        self.assertIn("HP +9", other)
        self.assertIn("TYPE NORMAL_STRONG", other)
        self.assertNotEqual(text, other)
        # Hidden future draws do not change the model-facing representation.
        changed = deepcopy(steps)
        changed[7]["scenario_id"] = "OTHER_HIDDEN_MONSTER"
        changed[7]["combat_seed"] += 1
        self.assertEqual(context, public_route_context(changed, 1))

    def test_boss_context_and_version_boundary(self):
        decision = decision_fixture()
        state = replace(decision.state, route_context=public_route_context(route_steps(), 8))
        text = serialize_observation(state, decision.actions, version="observation_v7")
        self.assertIn("REMAINING_COMBATS_AFTER_CURRENT: 0", text)
        self.assertIn("IS_FINAL_BOSS_COMBAT: TRUE", text)
        with self.assertRaisesRegex(ValueError, "requires explicit route context"):
            serialize_observation(decision.state, decision.actions, version="observation_v7")
        with self.assertRaisesRegex(ValueError, "requires observation_v7"):
            serialize_observation(state, decision.actions, version="observation_v6")
        with self.assertRaises(ValueError):
            replace(state.route_context, remaining_operations=(RouteOperation("heal", heal_amount=24),))

    def test_route_lineage_groups_formal_seeds_and_arms(self):
        config = {"run_id": "first", "seeds": {"reward": 81}}
        first = {"route_index": 0, "combat_seed_group_index": 0, "steps": route_steps()}
        other = deepcopy(first)
        other["combat_seed_group_index"] = 3
        other["steps"][1]["combat_seed"] = 999
        other["steps"][1]["policy_seed"] = 998
        self.assertEqual(build_route_lineage(config, first)["source_route_id"],
                         build_route_lineage({**config, "run_id": "v7"}, other)["source_route_id"])
        other["steps"][7]["scenario_id"] = "another_elite"
        self.assertNotEqual(build_route_lineage(config, first)["source_route_id"],
                            build_route_lineage(config, other)["source_route_id"])

    def test_combat_terminal_is_not_route_terminal_and_retry_matches_log(self):
        decision = decision_fixture()
        for outcome, final_boss, expected_done in (
            (TerminalOutcome.VICTORY, False, False),
            (TerminalOutcome.DEFEAT, False, True),
            (TerminalOutcome.VICTORY, True, True),
            (TerminalOutcome.ABORTED, False, True),
        ):
            with self.subTest(outcome=outcome, boss=final_boss), TemporaryDirectory() as temporary:
                state = replace(decision.state, route_context=public_route_context(route_steps(), 8 if final_boss else 1))
                backend = RecordingBackend(retry=True)
                policy = LLMPolicy(backend, 0, observation_version="observation_v7")
                result = policy.select_action(state, decision.actions)
                logger = TrajectoryLogger(
                    Path(temporary) / "combat.jsonl", episode_id="route-combat", game_seed=41,
                    policy_seed=0, policy_name="llm", observation_serializer_version="observation_v7",
                    route_lineage={"source_route_id": "group", "combat_index": state.route_context.combat_index},
                )
                row = logger.log_transition(state=state, legal_actions=decision.actions,
                                            policy_result=result, next_state=None, done=True,
                                            terminal_outcome=outcome)
                self.assertEqual(row["serialized_state"], backend.requests[0].user_prompt)
                self.assertIn(row["serialized_state"], backend.requests[1].user_prompt)
                self.assertEqual(row["route_done"], expected_done)
                self.assertEqual(row["route_lineage"]["source_route_id"], "group")
                self.assertNotIn("source_route_id", row["serialized_state"])


class RecordingBackend:
    def __init__(self, *, retry=False):
        self.requests = []
        self.retry = retry

    def generate(self, request):
        self.requests.append(request)
        if self.retry and len(self.requests) == 1:
            return "INVALID"
        for kind in ("PLAY", "SELECT", "END_TURN"):
            match = re.search(r"^(ACTION_\d+): " + kind, request.user_prompt, re.MULTILINE)
            if match:
                return match[1]
        raise AssertionError("No legal action in prompt")


class NativeRouteObservationTests(NativeSelectionTestCase):
    """Two short combats/selection steps using the existing native bridge."""

    def setUp(self):
        self.client = self.execution.create_client()
        self.env = self.execution.create_environment()
        self.addCleanup(self.client.close)
        self.addCleanup(self.env.close)

    def snapshot(self, cards, relics):
        from sts1_llm_policy.env.combat_snapshot import with_snapshot_relics
        self.client.reward_reset(712, ascension=0, act=1, relics=[])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        snapshot["deck"] = [{"string_id": name, "upgraded": False} for name in cards]
        snapshot["current_hp"] = 30
        return with_snapshot_relics(snapshot, relics)

    def test_continuous_executor_logs_actual_context_and_intercombat_heal(self):
        from sts1_llm_policy.eval.generation_strategies import resolve_generation_strategies
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/panels/continuous_act1_development.json").read_text(encoding="utf-8"))
        config["run_id"] = "route-observation-test"
        config["route"]["operations"] = ["combat:normal_weak", "heal:24", "combat:boss"]
        config["max_decisions"] = 100
        scope = load_scope(root / config["scope"], project_root=root)
        route = {"route_index": 0, "combat_seed_group_index": 0, "steps": [
            {"kind": "reset", "act": 1, "relics": [{"id": "BURNING_BLOOD"}]},
            {"kind": "combat", "combat_index": 1, "floor": 1, "encounter_family": "normal_weak",
             "scenario_id": "small_slimes", "combat_seed": 712, "policy_seed": 0},
            {"kind": "heal", "amount": 24},
            {"kind": "combat", "combat_index": 2, "floor": 3, "encounter_family": "boss",
             "scenario_id": "the_guardian", "combat_seed": 713, "policy_seed": 1},
        ]}
        backend = RecordingBackend()
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            validated = {
                "config": config, "execution": self.execution,
                # This bounded fixture has no rewards; the historical picker is
                # intentionally restricted to its seven/fifteen-reward panels.
                "strategies": replace(resolve_generation_strategies(config["strategies"]),
                                      card_pick=lambda **kwargs: lambda *args: self.fail("No reward node declared")),
                "card_ids": frozenset(scope["reward_cards"]["included_enum_ids"]),
                "output": output, "identity": {},
            }
            result = _run_route(validated, {"config": {"id": "test", "policy": "llm"}}, route,
                                backend=backend, picker=None, rate_table=None, route_root=output)
            self.assertEqual(len(result["combats"]), 2)
            first, second = result["combats"]
            self.assertEqual(second["input_snapshot"]["current_hp"],
                             min(first["summary"]["max_hp"], first["summary"]["ending_hp"] + 24))
            rows = [row for combat in result["combats"]
                    for row in iter_trajectory_records(output / combat["trajectory"])]
            self.assertEqual([r["serialized_state"] for r in rows], [r.user_prompt for r in backend.requests])
            self.assertEqual(len({r["route_lineage"]["source_route_id"] for r in rows}), 1)
            self.assertTrue(rows[-1]["route_done"])
            self.assertIn("REMAINING_COMBATS_AFTER_CURRENT: 1", rows[0]["serialized_state"])
            self.assertIn("REMAINING_COMBATS_AFTER_CURRENT: 0", rows[-1]["serialized_state"])

    def test_decision_limit_stops_route_without_heal_and_is_resumable(self):
        from sts1_llm_policy.eval.generation_strategies import resolve_generation_strategies
        from sts1_llm_policy.workflows.continuous_artifacts import (
            load_completed_route,
            summarize_arm,
            IDENTITY_SCHEMA_VERSION,
        )
        identity = {"schema_version": IDENTITY_SCHEMA_VERSION}
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / "configs/panels/continuous_act1_development.json").read_text(encoding="utf-8"))
        config["run_id"] = "route-observation-test"
        config["route"]["operations"] = ["combat:normal_weak", "heal:24", "combat:boss"]
        config["max_decisions"] = 1
        route = {"route_index": 0, "combat_seed_group_index": 0, "steps": [
            {"kind": "reset", "act": 1, "relics": [{"id": "BURNING_BLOOD"}]},
            {"kind": "combat", "combat_index": 1, "floor": 1, "encounter_family": "normal_weak",
             "scenario_id": "jaw_worm", "combat_seed": 712, "policy_seed": 0},
            {"kind": "heal", "amount": 24},
            {"kind": "combat", "combat_index": 2, "floor": 3, "encounter_family": "boss",
             "scenario_id": "the_guardian", "combat_seed": 713, "policy_seed": 1},
        ]}
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            validated = {
                "config": config, "execution": self.execution,
                "strategies": replace(resolve_generation_strategies(config["strategies"]),
                                      card_pick=lambda **kwargs: lambda *args: self.fail("Unexpected reward")),
                "card_ids": frozenset(), "output": output, "identity": identity,
            }
            result = _run_route(validated, {"config": {"id": "test", "policy": "llm"}}, route,
                                backend=RecordingBackend(), picker=None, rate_table=None, route_root=output)
            self.assertEqual(result["status"], "truncated")
            self.assertEqual(result["truncation_reason"], "decision_limit")
            self.assertEqual(result["truncation_combat_index"], 1)
            self.assertIsNone(result["death_combat_index"])
            self.assertEqual(result["combats_started"], 1)
            self.assertEqual(result["topology_events"], [])
            combat = result["combats"][0]
            rows = list(iter_trajectory_records(output / combat["trajectory"]))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[-1]["route_done"])
            self.assertEqual(rows[-1]["route_terminal_outcome"], "aborted")
            metrics = summarize_arm([result])
            self.assertEqual((metrics["route_execution_count"], metrics["boss_victories"], metrics["truncated_routes"]), (1, 0, 1))
            self.assertEqual(metrics["death_combat_index_counts"], {})
            path = output / "route.json"
            path.write_text(json.dumps(result))
            self.assertEqual(load_completed_route(path, identity=identity, arm="test", route=route, output_root=output), result)
            result["death_combat_index"] = 1
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "outcome is inconsistent"):
                load_completed_route(path, identity=identity, arm="test", route=route, output_root=output)

    def test_victory_heal_and_relic_counter_feed_next_combat_behavior(self):
        from sts1_llm_policy.eval.episode_runner import run_combat_episode
        snapshot = self.snapshot(["Carnage"] * 12, [
            {"id": "BURNING_BLOOD", "counter": 0}, {"id": "NUNCHAKU", "counter": 6},
        ])
        context = CombatRouteContext(1, 2, "normal_weak", "the_guardian",
                                     (RouteOperation("combat", "boss"),))
        self.env.reset("small_slimes", 712, combat_snapshot=snapshot, route_context=context)
        backend = RecordingBackend()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "combat.jsonl"
            logger = TrajectoryLogger(path, episode_id="first", game_seed=712, policy_seed=0,
                                      policy_name="test", observation_serializer_version="observation_v7")
            result = run_combat_episode(self.env, LLMPolicy(backend, 0, observation_version="observation_v7"),
                                        logger, max_steps=30)
            self.assertEqual(result.outcome, TerminalOutcome.VICTORY)
            records = tuple(iter_trajectory_records(path))
            summary = summarize_combat(records, result.outcome.value)
        before_victory = records[-1]["canonical_state"]["combat"]["player"]["current_hp"]
        self.assertEqual(summary["ending_hp"], min(summary["max_hp"], before_victory + 6))
        self.assertEqual(summary["total_hp_loss"], 30 - before_victory)
        counters = {r["id"]: r["counter"] for r in summary["relics"]}
        self.assertEqual(counters["NUNCHAKU"], 9)
        carried = _stateful_snapshot(snapshot, {
            "current_hp": summary["ending_hp"], "max_hp": summary["max_hp"], "relics": summary["relics"],
        }, floor=3)
        state = self.env.reset("the_guardian", 713, combat_snapshot=carried,
                               route_context=CombatRouteContext(2, 2, "boss", "the_guardian", ()))
        self.assertEqual(state.combat.player.current_hp, summary["ending_hp"])
        self.assertEqual(state.combat.player.block, 0)
        text = serialize_observation(state, self.env.legal_actions(), version="observation_v7")
        self.assertIn("Nunchaku | COUNTER 9", text)
        result = LLMPolicy(RecordingBackend(), 0, observation_version="observation_v7").select_action(state, self.env.legal_actions())
        after = self.env.step(result.action)
        self.assertEqual(after.combat.player.energy, state.combat.player.energy - 2 + 1)
        self.assertEqual(next(r.counter for r in after.relics if r.relic_id == "NUNCHAKU"), 0)

    def test_selection_retry_keeps_route_and_reset_clears_it(self):
        from sts1_llm_policy.env.card_selection import CardSelectionState
        snapshot = self.snapshot(["Armaments", "Defend_R", "Strike_R", "Bash", "Defend_R"], [])
        context = CombatRouteContext(1, 2, "normal_weak", "the_guardian",
                                     (RouteOperation("upgrade"), RouteOperation("combat", "boss")))
        self.env.reset("jaw_worm", 712, combat_snapshot=snapshot, route_context=context)
        action = next(a for a in self.env.legal_actions() if a.card_name == "Armaments")
        selected = self.env.step(action)
        self.assertIsInstance(selected, CardSelectionState)
        self.assertEqual(selected.route_context, context)
        backend = RecordingBackend(retry=True)
        result = LLMPolicy(backend, 0, observation_version="observation_v7").select_action(selected, self.env.legal_actions())
        self.assertTrue(result.retry_used)
        self.assertIn("CARD_SELECTION:", backend.requests[0].user_prompt)
        self.assertIn(context.serialize(), backend.requests[1].user_prompt)
        after = self.env.step(result.action)
        self.assertEqual(after.route_context, context)
        reset = self.env.reset("jaw_worm", 713, combat_snapshot=snapshot)
        self.assertIsNone(reset.route_context)
        self.assertNotIn("ROUTE_CONTEXT:", serialize_observation(reset, self.env.legal_actions(), version="observation_v6"))
