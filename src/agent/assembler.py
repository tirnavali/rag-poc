"""BalancedContextAssembler — cross-collection doc-deduped primary slot fill."""
from __future__ import annotations

from src.agent.schemas import Chunk, ContextAssemblyItem, OrchestratorState
from src.config.pipeline_loader import AllocationConfig


class BalancedContextAssembler:
    """Iterates collections in priority order, deduplicates by document_id
    across collections, honors max_per_document and max_total_primary.
    """

    def __init__(self, config: AllocationConfig) -> None:
        self._config = config

    def run(self, state: OrchestratorState) -> OrchestratorState:
        per_doc_count: dict[str, int] = {}
        assembled: list[Chunk] = []
        items: list[ContextAssemblyItem] = []
        total = 0

        for plan in sorted(state.collection_plans, key=lambda p: p.priority):
            rr = state.retrieval_results.get(plan.collection_name)
            if not rr:
                continue

            taken = 0
            for chunk in rr.chunks:
                if total >= self._config.max_total_primary:
                    break
                if per_doc_count.get(chunk.document_id, 0) >= self._config.max_per_document:
                    continue
                if taken >= plan.retrieval_budget:
                    break

                assembled.append(chunk)
                items.append(ContextAssemblyItem(
                    chunk_id=chunk.chunk_id,
                    collection_name=chunk.collection_name,
                    document_id=chunk.document_id,
                    slot_type="primary",
                    assembly_reason="collection_budget_fill",
                    order_index=len(items),
                ))
                per_doc_count[chunk.document_id] = per_doc_count.get(chunk.document_id, 0) + 1
                taken += 1
                total += 1

            if total >= self._config.max_total_primary:
                break

        state.assembled_chunks = assembled
        state.balanced_context = items
        return state
