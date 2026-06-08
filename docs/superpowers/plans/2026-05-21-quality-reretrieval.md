# Quality-Based Re-Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trigger a targeted second-round search when the agent's answer fails to address the user's question (detected via `addresses_query=False` OR Turkish "nothing-found" keyword patterns).

**Architecture:** Add `_needs_quality_reretrieval()` and `_generate_gap_fill_plan()` to `PlanningAgent`. After the sanitizer retry loop in `run()`, if quality fails, call the planner LLM with a gap-fill prompt that includes the insufficient answer and validation issues, execute the new plan, merge results, and re-answer. Single extra round, fail-open on LLM error.

**Tech Stack:** Python, Pydantic, Ollama (via existing `LLMClientPool`), pytest + unittest.mock

---

## File Map

| File | Change |
|---|---|
| `src/config/pipeline_loader.py` | Add `re_retrieval_on_quality_failure` to `PlannerConfig` |
| `pipeline.yaml` | Add `on_quality_failure: true` under `agent.planner.re_retrieval` |
| `src/agent/schemas.py` | Add `quality_re_retrieved: bool = False` to `AgentOutput` |
| `src/agent/planner.py` | Add `NOTHING_FOUND_PATTERNS`, `GAP_FILL_PROMPT`, two new methods, hook in `run()` |
| `src/agent/tracer.py` | Render `quality_reretrieval` phase in retrieval section |
| `src/ui/chat.py` | Show `quality_re_retrieved` flag in console + debug_info |
| `Readme.md` | Update Re-retrieval section |
| `tests/test_quality_reretrieval.py` | New — 5 test cases |

---

## Task 1: Config — add `on_quality_failure` flag

**Files:**
- Modify: `src/config/pipeline_loader.py` (PlannerConfig.__init__, ~line 67)
- Modify: `pipeline.yaml` (agent.planner.re_retrieval section, ~line 89)

- [ ] **Step 1: Add field to PlannerConfig**

In `src/config/pipeline_loader.py`, inside `PlannerConfig.__init__` after the existing `re_retrieval` block reads (~line 72):

```python
        self.re_retrieval_on_quality_failure = rr.get("on_quality_failure", True)
```

The full `rr` block after the change:
```python
        rr = config.get("re_retrieval", {})
        self.re_retrieval_enabled = rr.get("enabled", True)
        self.re_retrieval_max_retries = rr.get("max_retries", 1)
        self.re_retrieval_min_results = rr.get("trigger_min_results", 3)
        self.re_retrieval_strategy = rr.get("strategy", "broaden_filters")
        self.re_retrieval_prompt = rr.get("prompt", "")
        self.re_retrieval_on_quality_failure = rr.get("on_quality_failure", True)
```

- [ ] **Step 2: Add to pipeline.yaml**

In `pipeline.yaml` under `agent.planner.re_retrieval`, add after `prompt: |...` block:

```yaml
      on_quality_failure: true  # re-search when answer doesn't address the query
```

- [ ] **Step 3: Verify config loads**

```bash
python -c "
from src.config.pipeline_loader import load_pipeline_config
cfg = load_pipeline_config()
print(cfg.planner.re_retrieval_on_quality_failure)
"
```

Expected output: `True`

- [ ] **Step 4: Commit**

```bash
git add src/config/pipeline_loader.py pipeline.yaml
git commit -m "feat(config): add re_retrieval.on_quality_failure flag to PlannerConfig"
```

---

## Task 2: Schema — add `quality_re_retrieved` to `AgentOutput`

**Files:**
- Modify: `src/agent/schemas.py`

- [ ] **Step 1: Add field to AgentOutput**

Open `src/agent/schemas.py`. Find the `AgentOutput` class. Add `quality_re_retrieved` alongside `re_retrieved`:

```python
class AgentOutput(BaseModel):
    answer: str
    thinking: str = ""
    plan: Optional[SearchPlan] = None
    validation: Optional[ValidationResult] = None
    trace: list[AgentTraceEvent] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    re_retrieved: bool = False
    quality_re_retrieved: bool = False   # ← add this line
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.agent.schemas import AgentOutput; o = AgentOutput(answer='x'); print(o.quality_re_retrieved)"
```

Expected: `False`

- [ ] **Step 3: Commit**

```bash
git add src/agent/schemas.py
git commit -m "feat(schemas): add quality_re_retrieved field to AgentOutput"
```

---

## Task 3: Write tests (TDD — before implementation)

**Files:**
- Create: `tests/test_quality_reretrieval.py`

- [ ] **Step 1: Write the test file**

```python
"""Unit tests for quality-based re-retrieval logic in PlanningAgent."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.planner import NOTHING_FOUND_PATTERNS, PlanningAgent
from src.agent.schemas import SearchPlan, ValidationResult
from src.config.pipeline_loader import PlannerConfig


def _make_validation(
    addresses_query: bool = True,
    passes: bool = True,
    issues: list[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        passes=passes,
        checks={"addresses_query": addresses_query, "backed_by_sources": True},
        issues=issues or ([] if addresses_query else ["Yanıt soruyu karşılamıyor"]),
    )


def _make_agent(quality_enabled: bool = True) -> PlanningAgent:
    """Build a PlanningAgent with mocked tools for unit testing."""
    from src.config.pipeline_loader import PipelineConfig

    config_dict = {
        "deployment_blocks": {
            "fast-01": {
                "host": "http://localhost:11434",
                "models": {"planner": "test-model", "sanitizer": "test-model"},
            },
            "gpu-01": {
                "host": "http://localhost:11434",
                "models": {"answer": "test-model"},
            },
        },
        "agent": {
            "planner": {
                "block": "fast-01",
                "model_key": "planner",
                "re_retrieval": {
                    "enabled": True,
                    "max_retries": 1,
                    "trigger_min_results": 3,
                    "on_quality_failure": quality_enabled,
                },
                "fallback": {"default_collections": [], "default_queries": []},
            },
            "answering": {"block": "gpu-01", "model_key": "answer"},
            "sanitizer": {
                "block": "fast-01",
                "model_key": "sanitizer",
                "validation_criteria": [],
            },
        },
        "retrieval": {},
    }
    config = PipelineConfig(config_dict)
    client_pool = MagicMock()
    with (
        patch("src.agent.planner.SearchTool"),
        patch("src.agent.planner.ContextBuilderTool"),
        patch("src.agent.planner.AnswerTool"),
        patch("src.agent.planner.SanitizerAgent"),
    ):
        return PlanningAgent(config=config, client_pool=client_pool)


# --- NOTHING_FOUND_PATTERNS ---

class TestNothingFoundPatterns:
    def test_bulunamadi_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Bu bilgi kaynaklarda bulunamadı.")

    def test_yer_almamaktadir_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Bu konu kaynaklarda yer almamaktadır.")

    def test_bilgi_yok_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Hakkında bilgi yok.")

    def test_tespit_edilemedi_matches(self):
        assert NOTHING_FOUND_PATTERNS.search("Kim söyledi tespit edilemedi.")

    def test_positive_answer_no_match(self):
        assert not NOTHING_FOUND_PATTERNS.search(
            "Deniz Baykal mecliste 'merdikıptı' dedi."
        )

    def test_case_insensitive(self):
        assert NOTHING_FOUND_PATTERNS.search("BULUNAMADI")


# --- _needs_quality_reretrieval ---

class TestNeedsQualityReretrieval:
    def test_addresses_query_false_triggers(self):
        agent = _make_agent()
        validation = _make_validation(addresses_query=False, passes=False)
        assert agent._needs_quality_reretrieval("Herhangi bir yanıt.", validation)

    def test_keyword_match_triggers_even_when_passes(self):
        agent = _make_agent()
        validation = _make_validation(addresses_query=True, passes=True)
        assert agent._needs_quality_reretrieval(
            "Bu bilgi kaynaklarda bulunamadı.", validation
        )

    def test_clean_answer_no_trigger(self):
        agent = _make_agent()
        validation = _make_validation(addresses_query=True, passes=True)
        assert not agent._needs_quality_reretrieval(
            "Deniz Baykal 1997'de şöyle dedi: ...", validation
        )

    def test_disabled_config_no_trigger(self):
        agent = _make_agent(quality_enabled=False)
        validation = _make_validation(addresses_query=False, passes=False)
        assert not agent._needs_quality_reretrieval("bulunamadı", validation)


# --- _generate_gap_fill_plan ---

class TestGenerateGapFillPlan:
    def test_returns_plan_on_valid_llm_response(self):
        agent = _make_agent()
        tracer = MagicMock()
        tracer.phase.return_value.__enter__ = MagicMock(return_value=MagicMock())
        tracer.phase.return_value.__exit__ = MagicMock(return_value=False)

        plan_json = json.dumps({
            "intent": "factual",
            "resources": [
                {
                    "collection": "tutanaklar_jina_v3_4k",
                    "mode": "parallel",
                    "priority": 1,
                    "query_drafts": [
                        {"text": "merdikıptı meclis", "filters": {"period": 23}, "top_k": 8}
                    ],
                }
            ],
            "reasoning": "gap fill",
        })

        mock_client = MagicMock()
        mock_client.chat.return_value.message.content = plan_json
        agent._pool.get_client.return_value = mock_client
        agent._pool.get_model_for_block.return_value = "test-model"

        with patch.object(agent._config, "get_collection_catalog", return_value="catalog"):
            validation = _make_validation(addresses_query=False, passes=False)
            result = agent._generate_gap_fill_plan(
                query="kim kime merdikıptı dedi",
                answer="bulunamadı",
                validation=validation,
                tracer=MagicMock(),
            )

        assert result is not None
        assert isinstance(result, SearchPlan)
        assert result.resources[0].collection == "tutanaklar_jina_v3_4k"
        assert result.resources[0].query_drafts[0].top_k == 8

    def test_returns_none_on_llm_error(self):
        agent = _make_agent()
        mock_client = MagicMock()
        mock_client.chat.side_effect = RuntimeError("LLM unavailable")
        agent._pool.get_client.return_value = mock_client
        agent._pool.get_model_for_block.return_value = "test-model"

        with patch.object(agent._config, "get_collection_catalog", return_value="catalog"):
            validation = _make_validation(addresses_query=False)
            result = agent._generate_gap_fill_plan(
                query="test",
                answer="bulunamadı",
                validation=validation,
                tracer=MagicMock(),
            )

        assert result is None
```

- [ ] **Step 2: Run tests — expect failures (NOTHING_FOUND_PATTERNS and methods not yet defined)**

```bash
python -m pytest tests/test_quality_reretrieval.py -v 2>&1 | head -40
```

Expected: `ImportError` or `AttributeError` — `NOTHING_FOUND_PATTERNS` not defined yet.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_quality_reretrieval.py
git commit -m "test(planner): add failing tests for quality-based re-retrieval"
```

---

## Task 4: Implement quality re-retrieval in `planner.py`

**Files:**
- Modify: `src/agent/planner.py`

- [ ] **Step 1: Add imports and constants at top of file**

After the existing imports (after `from src.config.pipeline_loader import PipelineConfig`), add:

```python
import re as _re
```

After the `RE_RETRIEVAL_PROMPT` constant, add:

```python
NOTHING_FOUND_PATTERNS = _re.compile(
    r"bulunamadı|bilgi\s+yok|kaynaklarda\s+yer\s+almıyor|tespit\s+edilemedi|"
    r"bilgiye\s+ulaşılamadı|mevcut\s+değil|yer\s+almamaktadır|"
    r"bilgi\s+bulunmamaktadır|yanıt\s+veremiyorum",
    _re.IGNORECASE,
)

GAP_FILL_PROMPT = """Önceki arama soruyu yanıtlayacak bilgiyi bulamadı.
Kullanıcının sorusunda aradığı spesifik bilgiyi bulmak için yeni hedefli arama sorguları üret.

Mevcut koleksiyonlar:
{catalog}

Orijinal soru: {query}
Yetersiz yanıt: {answer}
Sorunlar: {issues}

Özellikle şunlara odaklan:
- Soruda geçen spesifik kavramları farklı kelimelerle ifade et
- Eş anlamlı terimler, alternatif yazımlar dene
- Daha geniş veya daha dar kapsam alternatifleri ekle
- top_k değerini artır (en az 8)

JSON çıktısı (aynı format):
{{
  "intent": "...",
  "resources": [...],
  "reasoning": "..."
}}
"""
```

- [ ] **Step 2: Add `_needs_quality_reretrieval()` method to `PlanningAgent`**

Add after `_needs_reretrieval()` (~line 358):

```python
    def _needs_quality_reretrieval(
        self,
        answer: str,
        validation: ValidationResult,
    ) -> bool:
        """Check if quality-based re-retrieval should be triggered."""
        if not self._config.planner.re_retrieval_on_quality_failure:
            return False
        if not validation.checks.get("addresses_query", True):
            return True
        return bool(NOTHING_FOUND_PATTERNS.search(answer))
```

- [ ] **Step 3: Add `_generate_gap_fill_plan()` method**

Add after `_generate_broader_plan()`:

```python
    def _generate_gap_fill_plan(
        self,
        query: str,
        answer: str,
        validation: ValidationResult,
        tracer: PipelineTracer,
    ) -> SearchPlan | None:
        """Generate a targeted plan to fill the information gap in a failing answer."""
        planner_cfg = self._config.planner
        block_name = planner_cfg.block
        model_key = planner_cfg.model_key

        client = self._pool.get_client(block_name)
        model = self._pool.get_model_for_block(block_name, model_key)
        catalog = self._config.get_collection_catalog()

        issues_text = "; ".join(validation.issues) if validation.issues else "Yanıt soruyu karşılamıyor"
        prompt = GAP_FILL_PROMPT.format(
            catalog=catalog,
            query=query,
            answer=answer[:500],
            issues=issues_text,
        )

        try:
            res = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Sorgu: {query}"},
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": self._config.get_block(block_name).max_num_predict,
                },
                format="json",
                think=False,
            )
            raw = res.message.content.strip()
            plan_data = json.loads(raw)

            return SearchPlan(
                intent=plan_data.get("intent", "unknown"),
                resources=[
                    CollectionSearchPlan(
                        collection=r["collection"],
                        mode=r.get("mode", "parallel"),
                        priority=r.get("priority", 1),
                        query_drafts=[
                            SearchQueryDraft(
                                text=d["text"],
                                filters=d.get("filters"),
                                top_k=d.get("top_k", 8),
                            )
                            for d in r.get("query_drafts", [])
                        ],
                    )
                    for r in plan_data.get("resources", [])
                ],
                reasoning=plan_data.get("reasoning", ""),
            )
        except Exception:
            return None
```

- [ ] **Step 4: Hook into `run()` after sanitizer retry loop**

In `PlanningAgent.run()`, after the sanitizer for loop (after line `if validation and validation.passes: break`), before the `# Collect source metadata` comment, add:

```python
        # Phase 4b: Quality-based re-retrieval
        quality_re_retrieved = False
        if self._needs_quality_reretrieval(answer, validation):
            gap_plan = self._generate_gap_fill_plan(query, answer, validation, tracer)
            if gap_plan:
                gap_results = self._execute_plan(
                    gap_plan, tracer, phase="quality_reretrieval"
                )
                all_results = self._merge_results(all_results, gap_results)
                context, sources = self._context_tool.build(all_results)
                thinking, answer = self._call_answering(query, context, tracer)
                validation = self._validate_output(query, answer, sources, tracer)
                quality_re_retrieved = True
```

- [ ] **Step 5: Pass `quality_re_retrieved` to `AgentOutput` return**

In `run()`, update the return statement:

```python
        return AgentOutput(
            answer=answer,
            thinking=thinking,
            plan=plan,
            validation=validation,
            trace=tracer.events,
            sources=source_metas,
            re_retrieved=re_retrieved,
            quality_re_retrieved=quality_re_retrieved,
        )
```

- [ ] **Step 6: Run tests — expect pass**

```bash
python -m pytest tests/test_quality_reretrieval.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_filter_extractor_golden.py 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/agent/planner.py
git commit -m "feat(planner): add quality-based re-retrieval (gap-fill on addresses_query failure)"
```

---

## Task 5: Update tracer to display quality_reretrieval phase

**Files:**
- Modify: `src/agent/tracer.py`

- [ ] **Step 1: Include `quality_reretrieval` in retrieval_events filter**

In `print_trace()`, find the line:
```python
retrieval_events = [e for e in self.events if e.phase in ("retrieval", "re_retrieval")]
```

Change to:
```python
retrieval_events = [e for e in self.events if e.phase in ("retrieval", "re_retrieval", "quality_reretrieval")]
```

- [ ] **Step 2: Add rendering for the new phase**

In the retrieval_events rendering loop, after the `re_retrieval` branch:

```python
            if ev.phase == "re_retrieval":
                reason = ev.details.get("reason", "insufficient sources")
                lines.append(f"  [bold yellow]Re-retrieval[/bold yellow] ({reason})")
            elif ev.phase == "quality_reretrieval":
                lines.append(f"  [bold magenta]Quality Re-retrieval[/bold magenta] (answer quality)")
```

- [ ] **Step 3: Verify tracer renders without error**

```bash
python -c "
from src.agent.tracer import PipelineTracer
from src.agent.schemas import AgentTraceEvent
t = PipelineTracer()
t.events = [
    AgentTraceEvent(phase='retrieval', latency_ms=100, details={'collection': 'test', 'result_count': 5}),
    AgentTraceEvent(phase='quality_reretrieval', latency_ms=200, details={'collection': 'test', 'result_count': 3}),
]
from rich.console import Console
t.print_trace(Console())
"
```

Expected: prints trace with `Quality Re-retrieval` label, no errors.

- [ ] **Step 4: Commit**

```bash
git add src/agent/tracer.py
git commit -m "feat(tracer): render quality_reretrieval phase in pipeline trace"
```

---

## Task 6: Update chat.py UI

**Files:**
- Modify: `src/ui/chat.py`

- [ ] **Step 1: Show quality_re_retrieved flag in console**

In `_run_agent_query()`, after the existing `output.re_retrieved` check (~line 153):

```python
            if output.re_retrieved:
                console.print("  [bold cyan]↻ Re-retrieval tetiklendi (yetersiz sonuç)[/bold cyan]")
            if output.quality_re_retrieved:
                console.print("  [bold magenta]↻ Quality re-retrieval tetiklendi (yanıt yetersiz)[/bold magenta]")
```

- [ ] **Step 2: Add to debug_info**

In `_run_agent_query()`, add to the `debug_info` dict:

```python
            debug_info = {
                "agent_mode": True,
                "intent": output.plan.intent if output.plan else "unknown",
                "plan_reasoning": output.plan.reasoning if output.plan else "",
                "re_retrieved": output.re_retrieved,
                "quality_re_retrieved": output.quality_re_retrieved,   # ← add
                "validation_passed": output.validation.passes if output.validation else None,
            }
```

- [ ] **Step 3: Commit**

```bash
git add src/ui/chat.py
git commit -m "feat(chat): show quality_re_retrieved flag in agent mode console output"
```

---

## Task 7: Update Readme.md

**Files:**
- Modify: `Readme.md`

- [ ] **Step 1: Update the Re-retrieval/Fallback/Sanitizer section**

Find the section starting `### Re-retrieval, Fallback ve Sanitizer` (~line 599) and replace its content:

```markdown
### Re-retrieval, Fallback ve Sanitizer

- **Re-retrieval (miktar)**: Toplam sonuç `re_retrieval.trigger_min_results` altındaysa, planner
  filtreleri gevşeterek (yazar düşür → yıl düşür → semantik) yeni bir plan üretir.
- **Re-retrieval (kalite)**: Yanıt `addresses_query` kontrolünü geçemezse veya "bulunamadı /
  kaynaklarda yer almıyor" gibi Türkçe kalıplar içeriyorsa, `on_quality_failure: true` ile
  hedefli bir gap-fill araması tetiklenir. Validation sorunları ve yetersiz yanıt planner'a
  iletilir; model eksik bilgiyi spesifik sorgularla arar.
- **Fallback**: Planner LLM tamamen başarısız olursa, `agent.planner.fallback` altındaki
  varsayılan koleksiyon/sorgu kullanılır.
- **Sanitizer**: Yanıtı `validation_criteria` kriterlerine göre kontrol eder; başarısızsa
  `max_retries` kadar düzeltme dener. LLM hatasında fail-open davranır (yanıt geçer).
```

- [ ] **Step 2: Update the Akış (4 Faz) section**

Find the flow diagram (~line 529) and update to show the new Phase 4b:

```markdown
```
Sorgu
  ↓ FAZ 1: Planlama       (planner LLM → SearchPlan: intent, koleksiyonlar, query_drafts, filtreler)
  ↓ FAZ 2: Retrieval      (her draft için VectorSearch + reranker, koleksiyon başına)
  ↓ FAZ 2b: Re-retrieval  (sonuç < eşik ise filtreleri gevşeterek tekrar ara)
  ↓ FAZ 3: Answering      (answering LLM → bağlamdan yanıt üretir)
  ↓ FAZ 4: Validation     (sanitizer LLM → kriter kontrolü; gerekirse düzeltme denemesi)
  ↓ FAZ 4b: Quality Re-retrieval  (yanıt yetersizse gap-fill arama → tekrar yanıt)
AgentOutput (answer, thinking, plan, validation, trace, sources, re_retrieved, quality_re_retrieved)
```
```

- [ ] **Step 3: Commit**

```bash
git add Readme.md
git commit -m "docs(readme): document quality-based re-retrieval in agent pipeline section"
```

---

## Verification

```bash
# 1. Unit tests
python -m pytest tests/test_quality_reretrieval.py -v

# 2. Full suite (no regressions)
python -m pytest tests/ -v --ignore=tests/test_filter_extractor_golden.py

# 3. Config sanity
python -c "
from src.config.pipeline_loader import load_pipeline_config
cfg = load_pipeline_config()
print('on_quality_failure:', cfg.planner.re_retrieval_on_quality_failure)
"

# 4. Manual end-to-end in agent mode
python chat.py --agent
# Query: "mecliste kim kime merdikıptı dedi 23 dönem?"
# Expected trace shows Quality Re-retrieval phase if answer was insufficient
```
