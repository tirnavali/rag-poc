"""AllocationPlanner — builds per-collection execution plans with YAML-driven budgets."""
from __future__ import annotations

from src.agent.schemas import CollectionExecutionPlan, OrchestratorState, SearchPlan
from src.common.filter_translators import build_chroma_where
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
        drafts_by_collection = self._collect_draft_texts(state.planner_output)

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
                    query_drafts=drafts_by_collection.get(name, []),
                    route_reason="planner_suggested_and_session_allowed",
                )
            )
        state.collection_plans = plans
        return state

    @staticmethod
    def _collect_first_filters(plan: SearchPlan) -> dict[str, dict]:
        """Per-collection Chroma where-filters from each resource's first draft.

        Filters are translated to ChromaDB `where` syntax here (the orchestrator
        path's single choke point), so SearchTool receives the same already-
        translated dict as the PlanningAgent path (planner.py `_execute_single`).
        A raw model_dump (e.g. {"year_lte": 2000}) is NOT a valid Chroma filter:
        `year_lte`/`year_gte` are not real metadata fields and multi-field dicts
        need a `$and` wrapper — ChromaFilterTranslator handles both.
        """
        out: dict[str, dict] = {}
        for resource in plan.resources:
            if not resource.query_drafts:
                continue
            first = resource.query_drafts[0]
            if first.filters is None:
                continue
            # build_chroma_where resolves `author` to the collection's actual
            # labels ($in) per-collection, like the legacy _execute_single path.
            where = build_chroma_where(first.filters, resource.collection)
            if where:
                out[resource.collection] = where
        return out

    @staticmethod
    def _collect_draft_texts(plan: SearchPlan) -> dict[str, list[str]]:
        """Per-collection planner query rewrites, in draft order, deduplicated.

        These are the alternative phrasings the planner generated for a
        collection. The orchestrator runs each as a parallel search and RRF-fuses
        the ranked lists, so the planner's query expansion actually contributes
        recall instead of being discarded. Blank drafts are dropped; a collection
        with no usable drafts is omitted (retrieval falls back to the raw query).
        """
        out: dict[str, list[str]] = {}
        for resource in plan.resources:
            seen: set[str] = set()
            texts: list[str] = []
            for draft in resource.query_drafts:
                text = (draft.text or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
            if texts:
                out[resource.collection] = texts
        return out
