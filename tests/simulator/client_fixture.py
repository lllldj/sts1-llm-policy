"""In-memory JSONL process double for simulator client tests."""
import io


class _FakeStdout:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def readline(self) -> str:
        return self.lines.pop(0) if self.lines else ""

    def close(self) -> None:
        pass

class _FakeStdin:
    def __init__(self, process: "_FakeProcess") -> None:
        self.process = process
        self.pending = ""

    def write(self, value: str) -> int:
        self.pending += value
        return len(value)

    def flush(self) -> None:
        lines = self.pending.splitlines()
        self.pending = ""
        for line in lines:
            self.process.handle(line)

    def close(self) -> None:
        pass

class _FakeProcess:
    def __init__(self, *, mismatch_hello_id: bool = False) -> None:
        self.stdin = _FakeStdin(self)
        self.stdout = _FakeStdout()
        self.stderr = io.StringIO()
        self.returncode: int | None = None
        self.mismatch_hello_id = mismatch_hello_id
        self.requests: list[dict] = []

    def handle(self, line: str) -> None:
        import json

        request = json.loads(line)
        self.requests.append(request)
        operation = request["op"]
        response = {
            "v": 1,
            "id": request["id"],
            "op": operation,
            "ok": True,
        }
        if operation == "hello":
            response.update(
                backend="sts_lightspeed",
                protocol_version=1,
                transport="jsonl_stdio",
                capabilities={
                    "battle_scum_searcher2": True,
                    "privileged_root_search": True,
                    "root_action_coverage_v1": True,
                    "root_action_minimum_visits_v1": True,
                },
            )
            if self.mismatch_hello_id:
                response["id"] = "wrong"
        elif operation == "reset":
            response.update(
                state={
                    "decision_id": 0,
                    "terminal": False,
                    "scenario_id": request["scenario_id"],
                },
                legal_actions=[{"action_id": "END", "kind": "END_TURN"}],
            )
        elif operation == "legal_actions":
            response.update(
                decision_id=0,
                legal_actions=[{"action_id": "END", "kind": "END_TURN"}],
            )
        elif operation == "search":
            response.update(
                source="BattleScumSearcher2",
                decision_id=request["decision_id"],
                simulations=request["simulations"],
                search_seed=request["search_seed"],
                search_quality_normalization="min_max_guarded_v1",
                root_action_coverage_policy=(
                    "minimum_root_edge_visits_before_ucb_v1"
                    if request.get("minimum_root_action_visits", 0) > 1
                    else (
                        "visit_every_root_edge_before_ucb_v1"
                        if request.get("ensure_root_action_coverage") is True
                        or request.get("minimum_root_action_visits", 0) == 1
                        else "legacy_ucb_no_root_coverage_guarantee"
                    )
                ),
                minimum_root_action_visits=max(
                    request.get("minimum_root_action_visits", 0),
                    1 if request.get("ensure_root_action_coverage") is True else 0,
                ),
                root_simulation_count=request["simulations"],
                root_actions=[
                    {
                        "action_id": "END",
                        "kind": "END_TURN",
                        "visits": request["simulations"],
                        "evaluation_sum": 10.0,
                        "evaluation_square_sum": 100.0,
                        "mean_evaluation": 10.0 / request["simulations"],
                        "terminal_wins": request["simulations"],
                        "terminal_losses": 0,
                        "win_rate": 1.0,
                        "ending_hp_sum": 10 * request["simulations"],
                        "ending_hp_mean": 10.0,
                        "victory_ending_hp_sum": 10 * request["simulations"],
                        "victory_ending_hp_mean": 10.0,
                    }
                ],
                suggested_action_id="END",
                selection_rule=(
                    "max_win_rate_then_victory_hp_then_visits_then_mean_evaluation_then_action_id"
                ),
                best_sequence_first_action_id="END",
                best_sequence_length=1,
                best_action_value=10.0,
                min_action_value=1.0,
                best_outcome_player_hp=10,
                elapsed_ms=0.5,
                objective="upstream_battle_scum_evaluate_end_state_v1",
                privileged_state={
                    "battle_context_copy": True,
                    "ordered_draw_pile": True,
                    "future_rng_state": True,
                },
                transition_semantics={
                    "engine": "shared_corrected_action_execute_v1",
                    "bridge_corrections_applied_inside_rollouts": True,
                    "lagavulin_rejected": False,
                    "upgraded_disarm_decks_rejected": False,
                    "corrections_applied": {
                        "lagavulin_natural_wake": 0,
                        "upgraded_disarm": 0,
                        "red_slaver_entangle_once": 0,
                        "philosopher_bronze_orb_strength": 0,
                        "burning_blood_victory_heal": 0,
                    },
                },
            )
        elif operation == "step":
            response["ok"] = False
            response["error"] = {
                "code": "illegal_action",
                "message": "action_id is not legal in the current state",
            }
        elif operation == "close":
            self.returncode = 0

        self.stdout.lines.append(json.dumps(response) + "\n")

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -1

    def kill(self) -> None:
        self.returncode = -9
