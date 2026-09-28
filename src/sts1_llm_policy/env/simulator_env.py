from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Mapping, Protocol

from .action_schema import CanonicalAction
from .action_equivalence import build_model_action_space
from .errors import CombatEndedError
from .simulator_adapter import (
    SimulatorCanonicalDecision,
    from_sts_lightspeed_response,
)
from .simulator_client import StsLightspeedClient
from .simulator_search import (
    BattleScumSearchResult,
    ProjectedBattleScumSearchResult,
    parse_battle_scum_search_result,
    project_battle_scum_search_result,
)
from .state_schema import CanonicalState
from .route_context import CombatRouteContext


class SimulatorBridgeClient(Protocol):
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
    ) -> dict:
        ...

    def step(self, decision_id: int, action_id: str) -> dict:
        ...

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
    ) -> dict:
        ...


    def close(self) -> None:
        ...


class SimulatorCombatEndedError(CombatEndedError):
    """Terminal simulator transition with explicit reward and outcome."""

    def __init__(
        self,
        raw_state: dict,
        *,
        terminal_state: CanonicalState,
        reward: float,
        outcome: str,
    ) -> None:
        super().__init__(raw_state)
        self.terminal_state = terminal_state
        self.reward = reward
        self.outcome = outcome


class StsLightspeedEnv:
    """Canonical combat environment backed by one persistent bridge client."""

    def __init__(self, client: SimulatorBridgeClient | None = None, *, allow_card_selection: bool = False) -> None:
        self._allow_card_selection = allow_card_selection
        self._client = client if client is not None else StsLightspeedClient()
        self._decision: SimulatorCanonicalDecision | None = None
        self._raw_state: dict | None = None
        self._route_context: CombatRouteContext | None = None

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
        route_context: CombatRouteContext | None = None,
    ) -> CanonicalState:
        if route_context is not None and not isinstance(route_context, CombatRouteContext):
            raise ValueError("route_context must be a public CombatRouteContext")
        self._route_context = route_context
        reset_options: dict[str, object] = {"ascension": ascension}
        if deck_preset is not None:
            reset_options["deck_preset"] = deck_preset
        if deck_spec is not None:
            reset_options["deck_spec"] = deck_spec
        if combat_snapshot is not None:
            if deck_preset is not None or deck_spec is not None:
                raise ValueError(
                    "combat_snapshot is mutually exclusive with deck_preset and deck_spec"
                )
            if ascension != 0 or player_current_hp is not None:
                raise ValueError(
                    "combat_snapshot supplies ascension and player_current_hp"
                )
            reset_options["combat_snapshot"] = combat_snapshot
        if player_current_hp is None:
            response = self._client.reset(
                scenario_id,
                seed,
                **reset_options,
            )
        else:
            reset_options["player_current_hp"] = player_current_hp
            response = self._client.reset(
                scenario_id,
                seed,
                **reset_options,
            )
        decision = from_sts_lightspeed_response(response, allow_card_selection=self._allow_card_selection)
        decision = self._with_route_context(decision)
        if decision.terminal:
            raise RuntimeError("Simulator reset unexpectedly returned terminal state")
        self._set_decision(decision, response)
        return decision.state

    def get_state(self) -> CanonicalState:
        if self._decision is None:
            raise RuntimeError("Simulator environment has not been reset")
        return self._decision.state

    def get_raw_state(self) -> dict:
        if self._raw_state is None:
            raise RuntimeError("Simulator environment has not been reset")
        return deepcopy(self._raw_state)

    def legal_actions(self) -> tuple[CanonicalAction, ...]:
        if self._decision is None:
            raise RuntimeError("Simulator environment has not been reset")
        return self._decision.actions

    def search(
        self,
        *,
        simulations: int,
        search_seed: int,
        hidden_order_seed: int | None = None,
        public_state_seed: int | None = None,
        ensure_root_action_coverage: bool = False,
        minimum_root_action_visits: int = 0,
    ) -> BattleScumSearchResult:
        decision = self._decision
        if decision is None:
            raise RuntimeError("Simulator environment has not been reset")
        extra = {} if public_state_seed is None else {"public_state_seed": public_state_seed}
        response = self._client.search(
            decision.decision_id,
            simulations=simulations,
            search_seed=search_seed,
            hidden_order_seed=hidden_order_seed,
            ensure_root_action_coverage=ensure_root_action_coverage,
            minimum_root_action_visits=minimum_root_action_visits,
            **extra,
        )
        return parse_battle_scum_search_result(
            response,
            current_decision_id=decision.decision_id,
            current_native_action_ids=decision.native_action_ids,
            requested_hidden_order_seed=hidden_order_seed,
            requested_public_state_seed=public_state_seed,
            requested_root_action_coverage=ensure_root_action_coverage,
            requested_minimum_root_action_visits=minimum_root_action_visits,
        )


    def public_state(self, *, seed: int | None = None) -> CanonicalState:
        """Expose executed draw memory, optionally sample a fresh evaluation world.

        Route context is preserved; sampling does not change the policy objective.
        """
        if self._decision is None:
            raise RuntimeError("Simulator environment has not been reset")
        payload = {"decision_id": self._decision.decision_id}
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError("public state seed must be uint32")
            payload["public_state_seed"] = seed
        response = self._client.request("public_state", **payload)
        decision = from_sts_lightspeed_response(response, allow_card_selection=self._allow_card_selection)
        decision = self._with_route_context(decision)
        self._set_decision(decision, response)
        return decision.state

    def search_model_actions(
        self,
        *,
        simulations: int,
        search_seed: int,
        hidden_order_seed: int | None = None,
        public_state_seed: int | None = None,
        ensure_root_action_coverage: bool = False,
        minimum_root_action_visits: int = 0,
    ) -> ProjectedBattleScumSearchResult:
        decision = self._decision
        if decision is None:
            raise RuntimeError("Simulator environment has not been reset")
        result = self.search(
            public_state_seed=public_state_seed,
            simulations=simulations,
            search_seed=search_seed,
            hidden_order_seed=hidden_order_seed,
            ensure_root_action_coverage=ensure_root_action_coverage,
            minimum_root_action_visits=minimum_root_action_visits,
        )
        return project_battle_scum_search_result(
            result,
            native_action_ids=decision.native_action_ids,
            canonical_actions=decision.actions,
            action_space=build_model_action_space(
                decision.state,
                decision.actions,
            ),
        )

    def step(self, action: CanonicalAction) -> CanonicalState:
        decision = self._decision
        if decision is None:
            raise RuntimeError("Simulator environment has not been reset")

        native_action_id = decision.native_action_id(action)
        response = self._client.step(decision.decision_id, native_action_id)
        next_decision = from_sts_lightspeed_response(response, allow_card_selection=self._allow_card_selection)
        next_decision = self._with_route_context(next_decision)
        raw_state = self._normalize_raw_response(response)
        if next_decision.terminal:
            raise SimulatorCombatEndedError(
                raw_state,
                terminal_state=next_decision.state,
                reward=next_decision.reward,
                outcome=next_decision.outcome,
            )

        self._decision = next_decision
        self._raw_state = raw_state
        return next_decision.state

    def close(self) -> None:
        self._client.close()

    def _with_route_context(self, decision: SimulatorCanonicalDecision) -> SimulatorCanonicalDecision:
        if self._route_context is None:
            return decision
        return replace(decision, state=replace(decision.state, route_context=self._route_context))

    def _set_decision(
        self,
        decision: SimulatorCanonicalDecision,
        response: dict,
    ) -> None:
        self._decision = decision
        self._raw_state = self._normalize_raw_response(response)

    @staticmethod
    def _normalize_raw_response(response: dict) -> dict:
        """Remove transport request IDs while retaining simulator evidence."""

        return {
            "source": "simulator",
            "backend": "sts_lightspeed",
            "state": deepcopy(response["state"]),
            "legal_actions": deepcopy(response["legal_actions"]),
        }

    def __enter__(self) -> StsLightspeedEnv:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
