# RAG Guardrails — Design Spec

**Date:** 2026-05-24
**Status:** Design (awaiting implementation plan)
**Companion:** `docs/superpowers/specs/2026-05-24-agentic-orchestrator-design.md`
**Goal:** Bolt safety/quality guards onto the orchestrator pipeline at four insertion points. Cover input safety (prompt injection / jailbreak), off-topic refusal, hard citation grounding, and output toxicity.

---

## 1. Motivation

The orchestrator design leaves four gaps:

1. **No input filter.** Planner LLM receives the raw user query. Injection attempts (`Ignore previous instructions, do X`) can manipulate the plan or final answer.
2. **`EvidenceDecision.action == "refuse"` underspecified.** Judge can emit refuse but no concrete trigger criteria.
3. **Citation grounding is soft.** Existing `SanitizerAgent` flags issues and can rewrite, but does not enforce a hard rule that every factual claim maps to a cited chunk.
4. **No output content filter.** Toxic / hateful content from public-doc corpus passes through unchecked.

This spec adds explicit guard components at the four insertion points already reserved in `agentic-orchestrator-design.md §14`.

## 2. Scope

**In scope:**
- `InputGuard`: regex/blocklist for prompt injection + jailbreak patterns (Turkish + English)
- `OffTopicGuard`: heuristic feeding `EvidenceJudge`'s `refuse` action; LLM fallback for borderline
- `CitationGroundingGuard`: hard rule on answer ↔ cited chunk overlap; rejects ungrounded sentences
- `OutputGuard`: toxicity classifier on final answer text
- `RefuseReason` taxonomy + structured refuse messages
- Per-category fail mode (refuse / redact / uncertain)
- Trace events for every guard decision

**Out of scope (deferred):**
- PII redaction (corpus is public docs; revisit if private corpora added)
- Rate-limiting (single-user local chat; add when MCP/API surface goes multi-user)
- Audit log redaction
- ML-based injection classifier (start with patterns; upgrade later if false-negative rate high)

## 3. Threat Model and Failure Mode

| Category | Threat | Insertion point | Tier | Fail mode |
|---|---|---|---|---|
| Injection / jailbreak | Query manipulates Planner or Answer prompts; coerces off-script output | Pre-Planner | Pattern blocklist + small classifier fallback | **Hard refuse**: skip planner, emit refuse w/ reason `injection_detected`; log full query |
| Off-topic / out-of-corpus | Query has no chunks to ground on (e.g. coding question to parliamentary RAG) | Pre-Answer (Judge integration) | Heuristic: chunk count + score floor; LLM judge borderline | **Soft refuse**: refuse w/ reason `out_of_corpus`, suggest topic areas |
| Citation grounding | Answer contains claims not backed by assembled chunks | Post-Sanitizer | Token/n-gram overlap check; sentence-level mapping | **Mark uncertain or rewrite**: rewrite to drop ungrounded sentences; if all sentences fail, refuse w/ reason `ungrounded` |
| Toxicity | Generated answer contains hateful/violent content | Post-Answer (pre-citation) | Heuristic blocklist + optional external classifier | **Hard refuse**: drop answer, refuse w/ reason `toxic_output` |

Rate-limit deferred. Spec notes the insertion point at orchestrator entry; component stub may land empty.

## 4. Architecture (Insertion into Orchestrator)

```
[User Query]
   │
   ▼
[InputGuard] ────► refuse(injection_detected)  ◄── hard refuse, skip Planner
   │
   ▼
[Planner LLM]
   │
   ▼
[Policy] [Allocator] [Retrieve] [Assembler]
   │
   ▼
[EvidenceJudge] ──► consults OffTopicGuard for chunk-floor + score-floor
   │                refuse(out_of_corpus) is one of its actions
   │
   ▼  (action = answer)
[AnswerTool.stream]
   │
   ▼
[OutputGuard] ────► refuse(toxic_output)  ◄── hard refuse, drop streamed text
   │
   ▼
[SanitizerAgent] (existing, text-level rewrite)
   │
   ▼
[CitationGroundingGuard]
   │   ├─ all sentences grounded ─► proceed
   │   ├─ some ungrounded ──────► rewrite to drop them
   │   └─ all ungrounded ───────► refuse(ungrounded)
   ▼
[CitationBuilder] ─► final_answer + citations
```

OutputGuard runs on completed stream text, not per-token. Stream tokens render to UI optimistically; on toxic detection, replace rendered text with refuse message (acceptable UX trade vs token-level scanning). For paranoid mode, OutputGuard runs first chunk after stream end before sanitizer.

## 5. Schema Additions (`src/agent/schemas.py`)

```python
class RefuseReason(BaseModel):
    code: Literal[
        "injection_detected",
        "out_of_corpus",
        "ungrounded",
        "toxic_output",
        "no_allowed_collections",
        "judge_refuse",
    ]
    message: str            # user-facing Turkish message
    details: dict = Field(default_factory=dict)   # internal: matched pattern, score, etc.


class GuardDecision(BaseModel):
    guard_name: Literal["input", "off_topic", "grounding", "output"]
    passed: bool
    reason: Optional[RefuseReason] = None
    latency_ms: float = 0.0
    judge_type: Literal["heuristic", "classifier", "llm"] = "heuristic"


class GroundingReport(BaseModel):
    total_sentences: int
    grounded_sentences: int
    ungrounded_sentences: list[str] = Field(default_factory=list)
    coverage_ratio: float       # grounded / total
    rewrite_applied: bool = False
```

Augment `OrchestratorState`:

```python
class OrchestratorState(BaseModel):
    # ... existing fields ...
    guard_decisions: list[GuardDecision] = Field(default_factory=list)
    grounding_report: Optional[GroundingReport] = None
    refuse_reason: Optional[RefuseReason] = None
```

Augment `AgentOutput`:

```python
class AgentOutput(BaseModel):
    # ... existing fields ...
    refuse_reason: Optional[RefuseReason] = None
    grounding_report: Optional[GroundingReport] = None
    guard_trace: list[GuardDecision] = Field(default_factory=list)
```

## 6. Module Layout

| Path | Action | Responsibility |
|---|---|---|
| `src/agent/guards/__init__.py` | Create | Package marker |
| `src/agent/guards/input_guard.py` | Create | `InputGuard.check(query) -> GuardDecision` |
| `src/agent/guards/off_topic.py` | Create | `OffTopicGuard.evaluate(state) -> GuardDecision`; consumed by `EvidenceJudge` |
| `src/agent/guards/grounding.py` | Create | `CitationGroundingGuard.check(answer, chunks) -> tuple[str, GroundingReport]` |
| `src/agent/guards/output_guard.py` | Create | `OutputGuard.check(answer) -> GuardDecision` |
| `src/agent/guards/patterns.py` | Create | Injection / toxicity regex sets (Turkish + English) |
| `src/agent/guards/refuse_messages.py` | Create | Map of `RefuseReason.code` → user-facing Turkish text |
| `src/agent/schemas.py` | Modify | Add `RefuseReason`, `GuardDecision`, `GroundingReport`; augment State + Output |
| `src/agent/orchestrator.py` | Modify | Wire guards into 4 insertion points |
| `src/agent/judge.py` | Modify | Consume `OffTopicGuard` in heuristic stage |
| `src/config/pipeline_loader.py` | Modify | Add `GuardsConfig` (sub-configs per guard) |
| `pipeline.yaml` | Modify | Add `guards:` block |

## 7. Component Specifications

### 7.1 `InputGuard`

Pattern-first; small classifier fallback optional.

```python
class InputGuard:
    def __init__(self, config: InputGuardConfig) -> None:
        self._patterns = compile_patterns(config.pattern_set)   # from patterns.py
        self._classifier = _load_classifier(config) if config.classifier_enabled else None

    def check(self, query: str) -> GuardDecision:
        t0 = time.perf_counter()
        # Stage 1: pattern match
        for pattern, label in self._patterns:
            if pattern.search(query):
                return GuardDecision(
                    guard_name="input",
                    passed=False,
                    reason=RefuseReason(
                        code="injection_detected",
                        message=REFUSE_MESSAGES["injection_detected"],
                        details={"matched_pattern": label},
                    ),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    judge_type="heuristic",
                )
        # Stage 2: classifier (optional)
        if self._classifier and self._classifier.predict(query) > self._config.classifier_threshold:
            return GuardDecision(
                guard_name="input", passed=False,
                reason=RefuseReason(code="injection_detected", message=..., details={"classifier_score": ...}),
                judge_type="classifier",
            )
        return GuardDecision(guard_name="input", passed=True, latency_ms=...)
```

Pattern set seeds (Turkish + English) in `patterns.py`:

```python
INJECTION_PATTERNS = [
    (r"(?i)\bignore\s+(previous|all)\s+instructions?\b", "ignore_instructions_en"),
    (r"(?i)\bsystem\s+prompt\b", "system_prompt_en"),
    (r"\bönceki\s+talimatları\s+(unut|yoksay)\b", "ignore_instructions_tr"),
    (r"\bsen\s+(artık|şimdi)\s+\w+\s+olarak\b", "role_override_tr"),
    (r"(?i)\bjailbreak\b|\bDAN\s+mode\b", "jailbreak_token"),
    # ... extended set committed in patterns.py
]
```

Classifier (optional) — start without; revisit if pattern false-negative rate > 5% in eval.

### 7.2 `OffTopicGuard` (consumed by `EvidenceJudge`)

Not a separate stage; provides decision input to Judge's heuristic.

```python
class OffTopicGuard:
    def __init__(self, config: OffTopicConfig) -> None: ...

    def evaluate(self, state: OrchestratorState) -> GuardDecision:
        chunks = state.assembled_chunks
        if not chunks:
            return GuardDecision(
                guard_name="off_topic", passed=False,
                reason=RefuseReason(code="out_of_corpus", message=..., details={"chunk_count": 0}),
            )
        # Score floor: max rerank_score across chunks
        max_score = max((c.rerank_score for c in chunks), default=0.0)
        if max_score < self._config.min_max_rerank_score:
            return GuardDecision(
                guard_name="off_topic", passed=False,
                reason=RefuseReason(code="out_of_corpus", message=..., details={"max_rerank_score": max_score}),
            )
        return GuardDecision(guard_name="off_topic", passed=True)
```

`EvidenceJudge.run()` calls `OffTopicGuard.evaluate(state)` before the chunk-count check. If it returns `passed=False`, judge emits `action="refuse"` with the `out_of_corpus` reason instead of `clarify`/`expand`.

### 7.3 `CitationGroundingGuard`

Hard rule: each sentence in the answer maps to at least one assembled chunk by n-gram overlap. Below threshold → drop sentence.

```python
class CitationGroundingGuard:
    def __init__(self, config: GroundingConfig) -> None: ...

    def check(self, answer: str, chunks: list[Chunk]) -> tuple[str, GroundingReport]:
        sentences = split_sentences_tr(answer)
        chunk_corpus = " ".join(c.text for c in chunks)
        chunk_ngrams = build_ngram_set(chunk_corpus, n=self._config.ngram_size)

        grounded, ungrounded = [], []
        for sent in sentences:
            overlap = ngram_overlap_ratio(sent, chunk_ngrams, n=self._config.ngram_size)
            if overlap >= self._config.min_overlap_ratio:
                grounded.append(sent)
            else:
                ungrounded.append(sent)

        coverage_ratio = len(grounded) / max(len(sentences), 1)

        if not grounded:
            # All ungrounded → caller refuses
            return "", GroundingReport(
                total_sentences=len(sentences),
                grounded_sentences=0,
                ungrounded_sentences=ungrounded,
                coverage_ratio=0.0,
                rewrite_applied=False,
            )

        if ungrounded and self._config.rewrite_on_partial:
            rewritten = " ".join(grounded)
            return rewritten, GroundingReport(
                total_sentences=len(sentences),
                grounded_sentences=len(grounded),
                ungrounded_sentences=ungrounded,
                coverage_ratio=coverage_ratio,
                rewrite_applied=True,
            )

        return answer, GroundingReport(
            total_sentences=len(sentences),
            grounded_sentences=len(grounded),
            ungrounded_sentences=ungrounded,
            coverage_ratio=coverage_ratio,
            rewrite_applied=False,
        )
```

Tunables (from `pipeline.yaml`):
- `ngram_size`: 3 (trigrams; balances precision and Turkish morphology)
- `min_overlap_ratio`: 0.3 (sentence must share ≥30% trigrams with chunk corpus)
- `rewrite_on_partial`: true

Sentence splitter: Turkish-aware (handle `.` in `Sn.`, `Av.`, numbers; reuse existing if present in `src/common/text.py`, else add minimal splitter).

### 7.4 `OutputGuard`

Pattern blocklist primary; external classifier (e.g., Detoxify multilingual or local model) optional.

```python
class OutputGuard:
    def __init__(self, config: OutputGuardConfig) -> None: ...

    def check(self, answer: str) -> GuardDecision:
        for pattern, label in self._toxicity_patterns:
            if pattern.search(answer):
                return GuardDecision(
                    guard_name="output", passed=False,
                    reason=RefuseReason(
                        code="toxic_output",
                        message=REFUSE_MESSAGES["toxic_output"],
                        details={"matched_pattern": label},
                    ),
                )
        if self._classifier:
            score = self._classifier.score(answer)
            if score > self._config.classifier_threshold:
                return GuardDecision(
                    guard_name="output", passed=False,
                    reason=RefuseReason(code="toxic_output", message=..., details={"classifier_score": score}),
                    judge_type="classifier",
                )
        return GuardDecision(guard_name="output", passed=True)
```

Pattern set is conservative (slurs / explicit hate terms). Public parliamentary corpus is unlikely to surface these but historical quotes (e.g. press clips) may.

## 8. Refuse Message Map (`refuse_messages.py`)

```python
REFUSE_MESSAGES = {
    "injection_detected": (
        "Sorgunuzda sistem talimatlarına müdahale etme girişimi tespit edildi. "
        "Lütfen sorunuzu konuyla ilgili olarak yeniden sorun."
    ),
    "out_of_corpus": (
        "Bu soru için arşivde yeterli kaynak bulunamadı. "
        "Gazete arşivi ve meclis tutanaklarıyla ilgili sorular sorabilirsiniz."
    ),
    "ungrounded": (
        "Mevcut kaynaklarla doğrulanabilir bir yanıt üretilemedi. "
        "Lütfen sorunuzu daha özelleştirin veya başka bir kaynak seçin."
    ),
    "toxic_output": (
        "Yanıt politikalara uygun değil, gösterilemiyor. "
        "Lütfen sorunuzu yeniden yapılandırın."
    ),
    "no_allowed_collections": (
        "Seçili koleksiyonlarda bu konu için arama yapılamaz. "
        "Başlangıçta farklı koleksiyonlar seçin."
    ),
    "judge_refuse": (
        "Yetkili kaynaklarla yanıt veremiyorum."
    ),
}
```

## 9. `pipeline.yaml` Additions

```yaml
guards:
  input:
    enabled: true
    pattern_set: default       # patterns.py:INJECTION_PATTERNS
    classifier:
      enabled: false           # upgrade path
      model: ""
      threshold: 0.7
  off_topic:
    enabled: true
    min_max_rerank_score: 0.15   # below this, refuse out_of_corpus
  grounding:
    enabled: true
    ngram_size: 3
    min_overlap_ratio: 0.3
    rewrite_on_partial: true
    refuse_on_all_ungrounded: true
  output:
    enabled: true
    pattern_set: default       # patterns.py:TOXICITY_PATTERNS
    classifier:
      enabled: false
      model: ""
      threshold: 0.6
  rate_limit:
    enabled: false             # deferred; insertion point reserved
```

## 10. Orchestrator Wiring (changes in `src/agent/orchestrator.py`)

```python
class OrchestratorAgent:
    def __init__(self, config: PipelineConfig, client_pool: LLMClientPool) -> None:
        # ... existing components ...
        self._input_guard = InputGuard(config.guards.input)
        self._off_topic_guard = OffTopicGuard(config.guards.off_topic)
        self._output_guard = OutputGuard(config.guards.output)
        self._grounding_guard = CitationGroundingGuard(config.guards.grounding)

    def run(self, query, session_collections, stream_callback=None) -> AgentOutput:
        state = OrchestratorState(request_id=_uuid(), user_query=query)

        # Guard #1: input
        decision = self._input_guard.check(query)
        state.guard_decisions.append(decision)
        if not decision.passed:
            return self._make_refuse_from_guard(state, decision.reason)

        # ... Planner / Policy / Allocator / Retrieve / Assembler ...

        # Judge with OffTopicGuard input
        off_topic_decision = self._off_topic_guard.evaluate(state)
        state.guard_decisions.append(off_topic_decision)
        if not off_topic_decision.passed:
            return self._make_refuse_from_guard(state, off_topic_decision.reason)

        self._judge.run(state)
        # ... expand path ...

        if state.evidence_decision.action in ("clarify", "refuse"):
            return self._dispatch_non_answer(state)

        # Answer stream
        state.final_answer, validation = self._generate_and_validate(state, tracer, stream_callback)

        # Guard #2: output
        output_decision = self._output_guard.check(state.final_answer)
        state.guard_decisions.append(output_decision)
        if not output_decision.passed:
            return self._make_refuse_from_guard(state, output_decision.reason)

        # Guard #3: citation grounding (after sanitizer rewrite)
        rewritten, report = self._grounding_guard.check(state.final_answer, state.assembled_chunks)
        state.grounding_report = report
        state.guard_decisions.append(GuardDecision(
            guard_name="grounding",
            passed=report.grounded_sentences > 0,
            reason=None if report.grounded_sentences > 0 else RefuseReason(code="ungrounded", message=...),
        ))
        if report.grounded_sentences == 0:
            return self._make_refuse_from_guard(state, RefuseReason(code="ungrounded", message=REFUSE_MESSAGES["ungrounded"]))
        state.final_answer = rewritten

        # Citation
        state.citations = CitationBuilder.build(state.assembled_chunks)
        return self._build_output(state)
```

## 11. Streaming UX Considerations

- **InputGuard fail**: print refuse message immediately. No spinner, no stream.
- **OutputGuard fail post-stream**: tokens may have been rendered. Clear UI buffer, replace with refuse message. Chat library (Rich) supports this via panel update. Add `service.ask_stream()` contract: yields tokens then yields a terminal "FINAL" frame; if guard fails, terminal frame is the refuse message instead.
- **GroundingGuard partial rewrite**: stream finished; replace final rendered text with rewritten version + small notice `"(bazı kanıtlanamayan ifadeler kaldırıldı)"`.
- **GroundingGuard refuse (all ungrounded)**: clear stream, show refuse message.

Document the post-stream replacement contract in `src/ui/chat.py` and `src/generator/service.py`.

## 12. Testing Strategy

| Test file | Coverage |
|---|---|
| `tests/test_guard_input.py` | Pattern matches each in INJECTION_PATTERNS, clean queries pass, latency under 10ms, classifier-disabled path |
| `tests/test_guard_off_topic.py` | Zero chunks → refuse, low score → refuse, normal results pass, threshold tunable |
| `tests/test_guard_grounding.py` | All grounded passes, partial → rewrite, none → refuse, Turkish sentence splitting, ngram overlap math |
| `tests/test_guard_output.py` | Pattern matches, clean answers pass, classifier-disabled path |
| `tests/test_refuse_messages.py` | All RefuseReason codes have entries; Turkish text non-empty |
| `tests/test_orchestrator_with_guards.py` | E2E: injection blocks before planner, off-topic blocks after assembly, output toxic blocks after stream, ungrounded refuses, partial-grounded rewrites |
| Fixtures | `tests/fixtures/guards/` — sample queries (clean, injection-tr, injection-en, jailbreak); sample answers (grounded, ungrounded, toxic) |

## 13. Migration / Rollout

1. **Schemas + patterns + module skeletons** behind `guards.*.enabled: false`. Add `GuardDecision`, `RefuseReason`, `GroundingReport`. Pattern sets seeded.
2. **InputGuard** first (cheapest, simplest, highest value). Flip enabled; observe trace.
3. **OffTopicGuard** wired into `EvidenceJudge`. Tune `min_max_rerank_score` from eval golden distribution.
4. **CitationGroundingGuard** with `rewrite_on_partial: true` and `refuse_on_all_ungrounded: false` initially (observe-only). After eval, flip refuse on.
5. **OutputGuard** with pattern set only. Classifier upgrade path documented but not blocking.
6. **Eval pass**: run `src/evaluator/` golden suite with all guards on. Verify refuse-rate, false-positive rate per guard.
7. **Default flip**: all four guards enabled in shipped `pipeline.yaml`.
8. **Future**: rate-limit when MCP/API surface goes external; PII redaction if private corpora ingested.

## 14. Error Handling and Observability

- Any guard exception → log full traceback, fall back to "guard passed" (fail-open) to avoid blocking on guard bugs. Counter `guard_exceptions_total` tracked in trace.
- Each `GuardDecision` carries `latency_ms` and `judge_type` for cost/perf analysis.
- Trace event `phase: "guard"` per guard invocation, `details: GuardDecision.model_dump()`.
- `AgentOutput.refuse_reason.details` contains internal info (matched pattern, score) — useful for ops, not user-facing. Front-end displays only `RefuseReason.message`.

## 15. Open Questions (resolve during implementation)

- **Turkish sentence splitting**: build minimal in-repo splitter or pull `pysbd` / `spacy`? Default: minimal in-repo (handles `Sn.`, `Av.`, numeric decimals); upgrade if false-split rate hurts grounding accuracy.
- **OutputGuard classifier model**: which? Detoxify multilingual covers Turkish weakly. Defer choice; ship pattern-only first.
- **Grounding ngram_size**: 3 (trigrams) is the default; eval may push to 2 for Turkish morphology (suffix-heavy). Tune from data.
- **Sanitizer overlap with GroundingGuard**: sanitizer already validates groundedness softly. Decide if sanitizer's groundedness check should be dropped once GroundingGuard lands (it's a hard version of the same thing). Default: keep sanitizer for tone/style checks, narrow its groundedness criterion.
- **Refuse copy review**: have product/legal review Turkish refuse messages before flipping defaults on.
