# `src/agent` — Planlama Ajanı Pipeline'ı

Çok adımlı RAG sorgularını orkestre eder: plan → retrieve → answer → validate.

## Modüller

| Dosya | Görev |
|---|---|
| `planner.py` | `PlanningAgent` — ana giriş noktası; tüm pipeline döngüsünü yönetir |
| `tools.py` | `SearchTool`, `ContextBuilderTool`, `AnswerTool` — retrieval ve generation sarmalayıcıları |
| `sanitizer.py` | `SanitizerAgent` — yanıt kalitesini doğrular; gerekirse düzeltir |
| `schemas.py` | Pydantic sözleşmeleri: `SearchPlan`, `ValidationResult`, `AgentOutput` |
| `tracer.py` | `PipelineTracer` — aşama bazlı zamanlama ve yapısal trace olayları |

## Pipeline Akışı

```
PlanningAgent.run(query)
  │
  ├─ Aşama 1: _generate_plan()
  │     └─ LLM (fast-01/planner) → SearchPlan (intent + koleksiyon/query taslakları)
  │
  ├─ Aşama 2: _execute_plan()
  │     └─ SearchTool.search() × N draft (paralel veya sıralı, resource başına)
  │     └─ [min_results altındaysa re-retrieval döngüsü]
  │
  ├─ Aşama 3: ContextBuilderTool.build() → AnswerTool.generate()
  │     └─ LLM (gpu-01/answer) → (thinking, answer)
  │
  └─ Aşama 4: SanitizerAgent.validate()
        └─ LLM (fast-01/sanitizer) → ValidationResult
        └─ [yanıt başarısız + NOTHING_FOUND pattern → quality re-retrieval]
```

## Yapılandırma

Tüm ayarlar `pipeline.yaml` dosyasında `agent:` ve `retrieval:` bölümlerinde tanımlanır.
`src/config/pipeline_loader.PipelineConfig` üzerinden yüklenir.

Önemli parametreler:
- `planner.re_retrieval.enabled` / `trigger_min_results` — miktar bazlı re-retrieval
- `planner.re_retrieval.on_quality_failure` — kalite bazlı re-retrieval
- `sanitizer.max_retries` — vazgeçmeden önce kaç düzeltme denemesi yapılacağı
- `retrieval.reranker.enabled` — cross-encoder reranking (`SearchTool` örneği başına bir kez yüklenir)

## Kullanım

```python
from src.config.pipeline_loader import load_pipeline_config
from src.common.llm_client_pool import LLMClientPool
from src.agent.planner import PlanningAgent

config = load_pipeline_config()
pool   = LLMClientPool.from_config(config)
agent  = PlanningAgent(config, pool)

output = agent.run("Sorgunuz burada")
print(output.answer)
```

`AgentOutput` şu alanları taşır: `.answer`, `.plan`, `.validation`, `.trace`, `.sources`,
`.re_retrieved`, `.quality_re_retrieved`.
