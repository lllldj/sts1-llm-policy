from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, TextIO

from .feature_loadout import validate_deck_spec
from .simulator_installation import (
    StsLightspeedInstallation,
    discover_sts_lightspeed,
)


PROTOCOL_VERSION = 1
MAX_SEARCH_SIMULATIONS = 1_000_000
MAX_SEARCH_SEED = 2**32 - 1


class SimulatorClientError(RuntimeError):
    """Base error for the native simulator decision bridge."""


class SimulatorProcessError(SimulatorClientError):
    """Raised when the native bridge cannot be started or exits unexpectedly."""


class SimulatorProtocolError(SimulatorClientError):
    """Raised when the bridge emits a malformed or mismatched response."""


class SimulatorRemoteError(SimulatorClientError):
    """A structured request error returned by a healthy bridge process."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


PopenFactory = Callable[..., subprocess.Popen[str]]


class StsLightspeedClient:
    """Sequential JSONL client for one persistent native battle process."""

    def __init__(
        self,
        installation: StsLightspeedInstallation | None = None,
        *,
        project_root: str | Path | None = None,
        verify_hashes: bool = True,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self._installation = installation
        self._project_root = project_root
        self._verify_hashes = verify_hashes
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1
        self._hello: dict[str, Any] | None = None

    @property
    def hello_info(self) -> dict[str, Any]:
        self.start()
        assert self._hello is not None
        return self._hello

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.is_running:
            return
        if self._process is not None:
            raise SimulatorProcessError("Native simulator bridge has already exited")

        if self._installation is None:
            self._installation = discover_sts_lightspeed(
                project_root=self._project_root,
                verify_hashes=self._verify_hashes,
            )
        executable = self._installation.bridge_executable
        try:
            self._process = self._popen_factory(
                [str(executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
            )
        except OSError as error:
            raise SimulatorProcessError(
                f"Unable to start native simulator bridge: {executable}"
            ) from error

        try:
            hello = self._exchange("hello")
            if hello.get("backend") != "sts_lightspeed":
                raise SimulatorProtocolError("Native bridge backend mismatch")
            if hello.get("protocol_version") != PROTOCOL_VERSION:
                raise SimulatorProtocolError("Native bridge protocol mismatch")
            self._hello = hello
        except Exception:
            self._stop_process()
            raise

    def reset(
        self,
        scenario_id: str,
        seed: int,
        *,
        ascension: int = 0,
        player_current_hp: int | None = None,
        deck_preset: str | None = None,
        deck_spec: Mapping[str, object] | None = None,
        combat_snapshot: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {"scenario_id": scenario_id, "seed": seed}
        if combat_snapshot is not None:
            if deck_preset is not None or deck_spec is not None or player_current_hp is not None:
                raise ValueError(
                    "combat_snapshot is mutually exclusive with deck_preset, "
                    "deck_spec, and player_current_hp"
                )
            if ascension != 0:
                raise ValueError("combat_snapshot supplies ascension")
            payload["combat_snapshot"] = dict(combat_snapshot)
            return self.request("reset", **payload)
        payload["ascension"] = ascension
        if deck_spec is not None:
            if deck_preset is not None:
                raise ValueError("deck_preset and deck_spec are mutually exclusive")
            payload["deck_spec"] = validate_deck_spec(deck_spec).as_deck_spec()
        else:
            payload["deck_preset"] = deck_preset or "starter"
        if player_current_hp is not None:
            payload["player_current_hp"] = player_current_hp
        return self.request("reset", **payload)

    def legal_actions(self) -> dict[str, Any]:
        return self.request("legal_actions")

    def search(
        self,
        decision_id: int,
        *,
        simulations: int,
        search_seed: int,
        hidden_order_seed: int | None = None,
        public_state_seed: int | None = None,
        ensure_root_action_coverage: bool = False,
        minimum_root_action_visits: int = 0,
    ) -> dict[str, Any]:
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations <= 0
            or simulations > MAX_SEARCH_SIMULATIONS
        ):
            raise ValueError(
                "simulations must be an integer between 1 and 1000000"
            )
        if (
            isinstance(search_seed, bool)
            or not isinstance(search_seed, int)
            or search_seed < 0
            or search_seed > MAX_SEARCH_SEED
        ):
            raise ValueError("search_seed must be an unsigned 32-bit integer")
        if hidden_order_seed is not None and (
            isinstance(hidden_order_seed, bool)
            or not isinstance(hidden_order_seed, int)
            or hidden_order_seed < 0
            or hidden_order_seed > MAX_SEARCH_SEED
        ):
            raise ValueError("hidden_order_seed must be an unsigned 32-bit integer")
        if not isinstance(ensure_root_action_coverage, bool):
            raise ValueError("ensure_root_action_coverage must be boolean")
        if (
            isinstance(minimum_root_action_visits, bool)
            or not isinstance(minimum_root_action_visits, int)
            or minimum_root_action_visits < 0
            or minimum_root_action_visits > simulations
        ):
            raise ValueError(
                "minimum_root_action_visits must be an integer between 0 and simulations"
            )
        self.start()
        assert self._hello is not None
        capabilities = self._hello.get("capabilities")
        if (
            not isinstance(capabilities, dict)
            or capabilities.get("battle_scum_searcher2") is not True
            or capabilities.get("privileged_root_search") is not True
            or (
                ensure_root_action_coverage
                and capabilities.get("root_action_coverage_v1") is not True
            )
            or (
                minimum_root_action_visits > 0
                and capabilities.get("root_action_minimum_visits_v1") is not True
            )
        ):
            raise SimulatorProtocolError(
                "Native bridge does not declare privileged search support"
            )
        payload: dict[str, object] = {
            "decision_id": decision_id,
            "simulations": simulations,
            "search_seed": search_seed,
        }
        if hidden_order_seed is not None:
            payload["hidden_order_seed"] = hidden_order_seed
        if ensure_root_action_coverage:
            payload["ensure_root_action_coverage"] = True
        if minimum_root_action_visits > 0:
            payload["minimum_root_action_visits"] = minimum_root_action_visits
        if public_state_seed is not None:
            if isinstance(public_state_seed, bool) or not isinstance(public_state_seed, int) or not 0 <= public_state_seed <= MAX_SEARCH_SEED:
                raise ValueError("public_state_seed must be uint32")
            if hidden_order_seed is not None:
                raise ValueError("Do not mix hidden-state sampling protocols")
            payload["public_state_seed"] = public_state_seed
        return self._exchange("search", **payload)

    def step(self, decision_id: int, action_id: str) -> dict[str, Any]:
        return self.request(
            "step",
            decision_id=decision_id,
            action_id=action_id,
        )


    def reward_reset(
        self,
        seed: int,
        *,
        ascension: int = 0,
        act: int = 1,
        floor: int | None = None,
        relics: list[Mapping[str, object]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {
            "seed": seed,
            "ascension": ascension,
            "act": act,
        }
        if floor is not None:
            payload["floor"] = floor
        if relics is not None:
            payload["relics"] = [dict(relic) for relic in relics]
        return self.request("reward_reset", **payload)

    def sample_card_reward(self, *, room: str, floor: int) -> dict[str, Any]:
        return self.request("sample_card_reward", room=room, floor=floor)

    def apply_card_reward_choice(
        self,
        reward_id: int,
        choice: str,
    ) -> dict[str, Any]:
        return self.request(
            "apply_card_reward_choice",
            reward_id=reward_id,
            choice=choice,
        )

    def advance_reward_act(self, act: int) -> dict[str, Any]:
        return self.request("advance_reward_act", act=act)

    def obtain_reward_relic(self, relic_id: str) -> dict[str, Any]:
        return self.request("obtain_reward_relic", relic_id=relic_id)

    def remove_reward_card(self, card_id: str) -> dict[str, Any]:
        return self.request("remove_reward_card", card_id=card_id)

    def upgrade_reward_card(self, deck_index: int) -> dict[str, Any]:
        return self.request("upgrade_reward_card", deck_index=deck_index)

    def export_combat_snapshot(self) -> dict[str, Any]:
        return self.request("export_combat_snapshot")

    def request(self, operation: str, **payload: object) -> dict[str, Any]:
        self.start()
        return self._exchange(operation, **payload)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                self._exchange("close")
            except SimulatorClientError:
                pass
        self._stop_process()

    def _exchange(self, operation: str, **payload: object) -> dict[str, Any]:
        process = self._process
        if process is None or process.poll() is not None:
            raise SimulatorProcessError("Native simulator bridge is not running")
        if process.stdin is None or process.stdout is None:
            raise SimulatorProcessError("Native simulator bridge pipes are unavailable")

        request_id = f"py-{self._next_request_id}"
        self._next_request_id += 1
        request = {
            "v": PROTOCOL_VERSION,
            "id": request_id,
            "op": operation,
            **payload,
        }
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        try:
            process.stdin.write(encoded + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError, UnicodeError) as error:
            raise SimulatorProcessError(
                "Native simulator bridge failed during request I/O"
            ) from error

        if line == "":
            return_code = process.poll()
            detail = self._read_finished_stderr(process.stderr, return_code)
            raise SimulatorProcessError(
                f"Native simulator bridge closed stdout (exit={return_code}){detail}"
            )
        try:
            response = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SimulatorProtocolError(
                "Native simulator bridge emitted invalid JSON"
            ) from error
        if not isinstance(response, dict):
            raise SimulatorProtocolError("Native bridge response must be an object")
        if response.get("v") != PROTOCOL_VERSION:
            raise SimulatorProtocolError("Native bridge response version mismatch")
        if response.get("id") != request_id:
            raise SimulatorProtocolError("Native bridge response id mismatch")
        if response.get("op") != operation:
            raise SimulatorProtocolError("Native bridge response operation mismatch")
        if response.get("ok") is False:
            error = response.get("error")
            if not isinstance(error, dict):
                raise SimulatorProtocolError("Native bridge error payload is malformed")
            code = error.get("code")
            message = error.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise SimulatorProtocolError("Native bridge error fields are malformed")
            raise SimulatorRemoteError(code, message)
        if response.get("ok") is not True:
            raise SimulatorProtocolError("Native bridge response ok must be boolean")
        return response

    @staticmethod
    def _read_finished_stderr(
        stream: TextIO | None,
        return_code: int | None,
    ) -> str:
        if stream is None or return_code is None:
            return ""
        try:
            content = stream.read().strip()
        except (OSError, UnicodeError):
            return ""
        return f": {content}" if content else ""

    def _stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self._process = None

    def __enter__(self) -> StsLightspeedClient:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
