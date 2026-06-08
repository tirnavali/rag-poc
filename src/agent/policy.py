"""PolicyEnforcer — session-selection intersection over planner-suggested collections."""
from __future__ import annotations

from src.agent.schemas import OrchestratorState, PolicyResult
from src.config.pipeline_loader import PolicyConfig


class PolicyEnforcer:
    """Enforces collection-access policy.

    Current mode: session_intersection. Planner-suggested collections are
    intersected with the user's session-selected collections; denied entries
    carry a reason string for trace and UI.
    """

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    def run(
        self,
        state: OrchestratorState,
        session_collections: list[str],
    ) -> OrchestratorState:
        suggested: list[str] = []
        if state.planner_output is not None:
            suggested = [r.collection for r in state.planner_output.resources]

        session_set = set(session_collections)
        allowed = [c for c in suggested if c in session_set]
        denied = [c for c in suggested if c not in session_set]

        state.policy_result = PolicyResult(
            allowed_collections=allowed,
            denied_collections=denied,
            reason_by_collection={c: "not_in_session_selection" for c in denied},
        )
        if not allowed:
            state.errors.append("policy_no_allowed_collections")
        return state
