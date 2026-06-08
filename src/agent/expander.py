"""ExpansionPlanner — consumes per-collection reserves to extend assembled context."""
from __future__ import annotations

from src.agent.schemas import ContextAssemblyItem, OrchestratorState


class ExpansionPlanner:
    """Pulls held-back reserve_chunks into assembled_chunks when the judge
    asks for expansion. No new vector calls; reserves are the next-best
    candidates from the original fetch.
    """

    def run(self, state: OrchestratorState) -> OrchestratorState:
        decision = state.evidence_decision
        if decision is None or decision.action != "expand":
            return state

        current_doc_ids = {c.document_id for c in state.assembled_chunks}
        added_total = 0

        for plan in sorted(state.collection_plans, key=lambda p: p.priority):
            rr = state.retrieval_results.get(plan.collection_name)
            if not rr:
                continue
            added = 0
            for chunk in rr.reserve_chunks:
                if chunk.document_id in current_doc_ids:
                    continue
                if added >= plan.reserve_budget:
                    break
                state.assembled_chunks.append(chunk)
                state.balanced_context.append(ContextAssemblyItem(
                    chunk_id=chunk.chunk_id,
                    collection_name=chunk.collection_name,
                    document_id=chunk.document_id,
                    slot_type="reserve",
                    assembly_reason="evidence_expansion",
                    order_index=len(state.balanced_context),
                ))
                current_doc_ids.add(chunk.document_id)
                added += 1
                added_total += 1

        state.expanded = added_total > 0
        return state
