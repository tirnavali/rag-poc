"""AllocationPlanner — builds per-collection execution plans with YAML-driven budgets."""
from __future__ import annotations

from src.agent.schemas import CollectionExecutionPlan, OrchestratorState, SearchPlan
from src.config.pipeline_loader import AllocationConfig


class AllocationPlanner:
    """Maps allowed collections to CollectionExecutionPlan entries.

    Budgets are looked up by `state.planner_output.query_type`. When the
    planner provides a draft with filters for a collection, the first draft's
    filters are propagated to the execution plan.
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._config = config

    def run(self, state: OrchestratorState) -> OrchestratorState:
        if not state.policy_result or not state.policy_result.allowed_collections:
            state.errors.append("allocation_no_allowed_collections")
            return state

        if state.planner_output is None:
            state.errors.append("allocation_no_planner_output")
            return state

        budget = self._config.budget_for(state.planner_output.query_type)
        filters_by_collection = self._collect_first_filters(state.planner_output)

        plans = []
        for idx, name in enumerate(state.policy_result.allowed_collections):
            plans.append(
                CollectionExecutionPlan(
                    collection_name=name,
                    priority=idx + 1,
                    retrieval_budget=budget.primary,
                    reserve_budget=budget.reserve,
                    fetch_k=budget.fetch_k,
                    filters=filters_by_collection.get(name, {}),
                    route_reason="planner_suggested_and_session_allowed",
                )
            )
        state.collection_plans = plans
        return state

    @staticmethod
    def _collect_first_filters(plan: SearchPlan) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for resource in plan.resources:
            if not resource.query_drafts:
                continue
            first = resource.query_drafts[0]
            if first.filters is None:
                continue
            data = first.filters.model_dump(exclude_none=True)
            if data:
                out[resource.collection] = data
        return out
