# Quality-Based Re-Retrieval Design

Date: 2026-05-21

## Context

The agent pipeline re-retrieval currently triggers only when the number of retrieved
documents is below `re_retrieval_min_results`. When enough documents are found but none
contain the requested specific information (names, dates, events), the model produces
a vague "not found" answer and stops. There is no mechanism to detect this quality
failure and try again with different queries.

Reported example: query "mecliste kim kime merdikıptı dedi 23 dönem?" returns 5 documents
(above threshold) but the answer omits the names because no retrieved chunk covered
that exact exchange.

## Goal

After the sanitizer retry loop, if the answer quality is still insufficient (cannot
answer the user's actual question), trigger one additional round of targeted search
designed specifically to fill the information gap — then re-answer with the augmented
context.

## Approach: B + C

**B — Gap-fill LLM plan:** When quality failure detected, call the planner LLM with a
new prompt that explicitly states what information is missing. This produces targeted
queries (e.g. specific person names, event terms) rather than just "broader" queries.

**C — Keyword heuristic guard:** Before calling the LLM, check the answer for Turkish
"nothing-found" patterns (`bulunamadı`, `kaynaklarda yer almıyor`, `tespit edilemedi`,
etc.). This provides a fast, zero-LLM-cost signal that also catches cases where the
sanitizer's `addresses_query` check passes leniently.

Both gates can independently trigger quality re-retrieval; either is sufficient.

## Architecture

### Config change — `src/config/pipeline_loader.py`

Add one field to `PlannerConfig`:
```python
re_retrieval_on_quality_failure: bool = True
```

### New prompt — `src/agent/planner.py`

```
GAP_FILL_PROMPT: str
```

Instructs the planner LLM to generate queries targeting the specific missing information,
using the original query, the insufficient answer, and validation issues as context.

### New methods — `PlanningAgent` in `src/agent/planner.py`

- `_needs_quality_reretrieval(answer: str, validation: ValidationResult) -> bool`
  Returns True if `addresses_query` check is False **OR** answer matches known
  "nothing found" Turkish keyword patterns.

- `_generate_gap_fill_plan(query, answer, validation, current_plan, tracer) -> SearchPlan | None`
  Calls planner LLM with `GAP_FILL_PROMPT`; same JSON schema as regular plan.
  Returns None on LLM error (fail-open).

### Hook in `PlanningAgent.run()` — after sanitizer loop

```python
if (not quality_re_retrieved
    and self._config.planner.re_retrieval_on_quality_failure
    and self._needs_quality_reretrieval(answer, validation)):
    gap_plan = self._generate_gap_fill_plan(query, answer, validation, current_plan, tracer)
    if gap_plan:
        gap_results = self._execute_plan(gap_plan, tracer, phase="quality_reretrieval")
        all_results = self._merge_results(all_results, gap_results)
        context, sources = self._context_tool.build(all_results)
        thinking, answer = self._call_answering(query, context, tracer)
        validation = self._validate_output(query, answer, sources, tracer)
        quality_re_retrieved = True
```

`quality_re_retrieved` flag prevents infinite loops (single extra round only).

### Trace display — `src/agent/tracer.py`

`quality_reretrieval` phase rendered under PHASE 2: Retrieval with a distinct label
(`[bold magenta]Quality Re-retrieval[/bold magenta]`).

### `AgentOutput` — `src/agent/schemas.py`

Add `quality_re_retrieved: bool = False` alongside existing `re_retrieved`.

### `pipeline.yaml`

```yaml
planner:
  re_retrieval_on_quality_failure: true
```

### `Readme.md`

Update the Re-retrieval/Fallback/Sanitizer section to document quality-triggered
re-retrieval.

## Data Flow

```
validation.passes=False
  OR answer contains "bulunamadı/..."
        │
        ▼
_generate_gap_fill_plan(query, answer, issues, current_plan)
        │
        ▼
_execute_plan(gap_plan, phase="quality_reretrieval")
        │
        ▼
_merge_results(original, gap_results)
        │
        ▼
_call_answering(query, enriched_context)
        │
        ▼
_validate_output → AgentOutput(quality_re_retrieved=True)
```

## Error Handling

- LLM fails in gap-fill → return None → skip quality re-retrieval, return original answer
- Gap-fill finds 0 new documents → merge is no-op → re-answer with same context
  (may still improve if answering LLM is nondeterministic, but generally same result)
- Flag `quality_re_retrieved=True` appears in trace either way once triggered

## Testing

`tests/test_quality_reretrieval.py`:

1. `test_keyword_heuristic_triggers` — answer with "bulunamadı" sets
   `_needs_quality_reretrieval=True` even when `addresses_query=True`
2. `test_addresses_query_false_triggers` — `addresses_query=False` triggers regardless
   of answer text
3. `test_quality_reretrieval_disabled` — `re_retrieval_on_quality_failure=False` in
   config prevents triggering
4. `test_quality_reretrieval_skipped_when_passes` — clean validation → no trigger
5. `test_gap_fill_plan_generated` — mock LLM, verify gap-fill prompt sent with
   issues/answer context

## Files Modified

| File | Change |
|---|---|
| `src/config/pipeline_loader.py` | Add `re_retrieval_on_quality_failure: bool = True` |
| `src/agent/planner.py` | `GAP_FILL_PROMPT`, `_needs_quality_reretrieval()`, `_generate_gap_fill_plan()`, hook in `run()` |
| `src/agent/schemas.py` | Add `quality_re_retrieved: bool = False` to `AgentOutput` |
| `src/agent/tracer.py` | Render `quality_reretrieval` phase |
| `src/ui/chat.py` | Show `quality_re_retrieved` flag in debug info |
| `pipeline.yaml` | Add `re_retrieval_on_quality_failure: true` |
| `Readme.md` | Update Re-retrieval section |
| `tests/test_quality_reretrieval.py` | New unit tests (5 cases) |
