from pathlib import Path
import tempfile
import unittest

from sts1_llm_policy.env.simulator_client import (
    SimulatorProtocolError,
    SimulatorRemoteError,
    StsLightspeedClient,
)
from sts1_llm_policy.env.feature_loadout import build_feature_loadout
from sts1_llm_policy.env.simulator_installation import StsLightspeedInstallation

from tests.simulator.client_fixture import _FakeProcess


def _installation(root: Path) -> StsLightspeedInstallation:
    executable = root / "bridge.exe"
    executable.write_bytes(b"fake")
    return StsLightspeedInstallation(
        project_root=root,
        isolation_root=root,
        source_dir=root,
        build_dir=root,
        manifest_path=root / "manifest.json",
        bridge_executable=executable,
        revision="a" * 40,
    )


class SimulatorClientTest(unittest.TestCase):
    def test_handshake_and_reset_use_matching_jsonl_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            response = client.reset("cultist", 12345)

            self.assertEqual(client.hello_info["backend"], "sts_lightspeed")
            self.assertEqual(response["state"]["scenario_id"], "cultist")
            self.assertEqual(response["legal_actions"][0]["action_id"], "END")
            client.close()
            self.assertFalse(client.is_running)

    def test_reset_sends_optional_player_current_hp_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            client.reset(
                "cultist",
                12345,
                player_current_hp=56,
                deck_preset="boss_ready",
            )

            reset_request = next(
                request for request in process.requests if request["op"] == "reset"
            )
            self.assertEqual(reset_request["player_current_hp"], 56)
            self.assertEqual(reset_request["deck_preset"], "boss_ready")
            client.close()

    def test_reset_sends_validated_dynamic_deck_spec_without_preset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )
            loadout = build_feature_loadout(
                "feature_boss", "status_exhaust", 44
            )

            client.reset("three_sentries", 12345, deck_spec=loadout.as_deck_spec())

            reset_request = next(
                request for request in process.requests if request["op"] == "reset"
            )
            self.assertNotIn("deck_preset", reset_request)
            self.assertEqual(
                reset_request["deck_spec"]["loadout_id"],
                loadout.loadout_id,
            )
            client.close()

    def test_reset_rejects_preset_and_deck_spec_together_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: _FakeProcess(),
            )
            loadout = build_feature_loadout(
                "feature_basic", "low_cost_order", 4
            )
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                client.reset(
                    "cultist",
                    1,
                    deck_preset="starter",
                    deck_spec=loadout.as_deck_spec(),
                )

    def test_search_sends_bounded_budget_seed_and_current_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            response = client.search(7, simulations=64, search_seed=123)

            request = next(
                request for request in process.requests if request["op"] == "search"
            )
            self.assertEqual(request["decision_id"], 7)
            self.assertEqual(request["simulations"], 64)
            self.assertEqual(request["search_seed"], 123)
            self.assertNotIn("hidden_order_seed", request)
            self.assertEqual(response["suggested_action_id"], "END")
            client.close()

    def test_search_sends_optional_hidden_order_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            client.search(
                7,
                simulations=64,
                search_seed=123,
                hidden_order_seed=701,
            )

            request = next(
                request for request in process.requests if request["op"] == "search"
            )
            self.assertEqual(request["hidden_order_seed"], 701)
            client.close()

    def test_search_sends_optional_root_action_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            client.search(
                7,
                simulations=64,
                search_seed=123,
                ensure_root_action_coverage=True,
            )

            request = next(
                request for request in process.requests if request["op"] == "search"
            )
            self.assertIs(request["ensure_root_action_coverage"], True)
            client.close()

    def test_search_sends_optional_minimum_root_action_visits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            response = client.search(
                7,
                simulations=8192,
                search_seed=123,
                minimum_root_action_visits=128,
            )

            request = next(
                request for request in process.requests if request["op"] == "search"
            )
            self.assertEqual(request["minimum_root_action_visits"], 128)
            self.assertEqual(response["minimum_root_action_visits"], 128)
            client.close()


    def test_reward_helpers_preserve_floor_and_choice_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            client.reward_reset(
                91,
                ascension=10,
                relics=[{"id": "QUESTION_CARD", "counter": 0}],
            )
            client.sample_card_reward(room="MONSTER", floor=3)
            client.apply_card_reward_choice(1, "CARD_0")
            client.advance_reward_act(2)

            operations = [request["op"] for request in process.requests]
            self.assertEqual(
                operations[-4:],
                [
                    "reward_reset",
                    "sample_card_reward",
                    "apply_card_reward_choice",
                    "advance_reward_act",
                ],
            )
            sample = process.requests[-3]
            self.assertEqual(sample["room"], "MONSTER")
            self.assertEqual(sample["floor"], 3)
            self.assertEqual(process.requests[-2]["choice"], "CARD_0")
            client.close()

    def test_combat_snapshot_is_forwarded_without_conflicting_reset_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )
            snapshot = {
                "schema_version": "combat_snapshot_v1",
                "fingerprint_fnv1a64": "0123456789abcdef",
            }

            client.export_combat_snapshot()
            client.reset("small_slimes", 123, combat_snapshot=snapshot)

            self.assertEqual(process.requests[-2]["op"], "export_combat_snapshot")
            reset = process.requests[-1]
            self.assertEqual(reset["combat_snapshot"], snapshot)
            self.assertNotIn("ascension", reset)
            self.assertNotIn("deck_preset", reset)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                client.reset(
                    "small_slimes",
                    123,
                    deck_preset="starter",
                    combat_snapshot=snapshot,
                )
            client.close()

    def test_search_rejects_invalid_budget_and_seed_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: _FakeProcess(),
            )

            for simulations in (True, 0, 1_000_001):
                with self.subTest(simulations=simulations):
                    with self.assertRaisesRegex(ValueError, "simulations"):
                        client.search(0, simulations=simulations, search_seed=1)
            for search_seed in (True, -1, 2**32):
                with self.subTest(search_seed=search_seed):
                    with self.assertRaisesRegex(ValueError, "search_seed"):
                        client.search(0, simulations=1, search_seed=search_seed)
            for hidden_order_seed in (True, -1, 2**32):
                with self.subTest(hidden_order_seed=hidden_order_seed):
                    with self.assertRaisesRegex(ValueError, "hidden_order_seed"):
                        client.search(
                            0,
                            simulations=1,
                            search_seed=1,
                            hidden_order_seed=hidden_order_seed,
                        )

    def test_remote_error_does_not_kill_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess()
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )
            client.start()

            with self.assertRaisesRegex(SimulatorRemoteError, "illegal_action") as caught:
                client.step(0, "NOT_LEGAL")

            self.assertEqual(caught.exception.code, "illegal_action")
            self.assertEqual(client.legal_actions()["decision_id"], 0)
            client.close()

    def test_rejects_response_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = _FakeProcess(mismatch_hello_id=True)
            client = StsLightspeedClient(
                _installation(Path(directory)),
                popen_factory=lambda *args, **kwargs: process,
            )

            with self.assertRaisesRegex(SimulatorProtocolError, "id mismatch"):
                client.start()

            self.assertFalse(client.is_running)


if __name__ == "__main__":
    unittest.main()
