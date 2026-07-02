# Faz B: Araştırma Stratejisi Playbook'u (research_strategies.md)

> [!NOTE]
> Bu doküman **yeni bir sohbette sıfırdan** yürütülmek üzere kendi kendine yeter.
> Aşağıdaki "Mevcut Durum" bölümü, playbook'un üstüne inşa edeceği hazır altyapıyı
> özetler. Todo maddeleri sırayla, kutucuklar işaretlenerek uygulanmalıdır.

## Amaç

Agent'ın, sorgu arketipine göre (özetle / hepsini getir / karşılaştır / analiz / olgusal)
bir **araştırma stratejisi** seçmesi; strateji **hem retrieval genişliğini/iterasyonunu
hem de yanıt sentez tarzını** yönlendirsin. Stratejiler **düzenlenebilir bir markdown
dosyasında** (`research_strategies.md`) tanımlanır; yeni strateji eklemek **kod değişikliği
gerektirmez**. Strateji seçimi **planner'ın mevcut LLM çağrısına katlanır** (ek gecikme yok);
deterministik keyword tespiti güvence/override olarak kalır.

## Mevcut Durum (Faz A — TAMAMLANDI, `main`'de)

Bu altyapı hazır ve playbook bunun üstüne oturacak:

- **`comprehensive` query_type** var: `SearchPlan.query_type` Literal'inde
  (`src/agent/schemas.py`), planner `_VALID_QUERY_TYPES`'ında (`src/agent/planner.py`).
- **Deterministik tespit**: `COMPREHENSIVE_KEYWORDS` (`src/config/settings.py`) +
  `OrchestratorAgent._is_comprehensive()` planlama sonrası query_type'ı yükseltiyor
  (`src/agent/orchestrator.py`, "Stage 2" planning bloğu).
- **Per-query-type bağlam tavanları**: `_AllocationBudget.max_total/max_per_document` +
  `AllocationConfig.max_total_for()/max_per_document_for()` (`src/config/pipeline_loader.py`);
  `BalancedContextAssembler.run()` bunları query_type'a göre okuyor (`src/agent/assembler.py`).
  `pipeline.yaml` → `allocation.by_query_type.comprehensive`.
- **İteratif toplama döngüsü**: `orchestrator.py` "Stage 5.1" — doygunluk (`_requery_expand`
  eklenen chunk sayısını döndürür) + ceiling + `judge.comprehensive_max_rounds`.
- **Sentez yanıtı (empty-answer fix)**: `SYNTHESIS_SYS_PROMPT` (`src/generator/prompts.py`) +
  `AnswerTool._select_system_prompt(mufettis_mode, query_type)` + `SYNTHESIS_QUERY_TYPES`
  (`src/agent/tools.py`); orchestrator Stage 6 `query_type`'ı `generate()`'e iletiyor.
  **Faz B bu sabit kümeyi playbook-güdümlü `answer_directive` ile değiştirecek.**

## Tasarım Kararları (onaylı)

1. Strateji seçimi **planner'ın tek çağrısında** yapılır (ek LLM turu yok).
2. Playbook **yapılandırılmış markdown** — hem retrieval (query_type preset) hem yanıt
   (answer_directive) kontrol eder.
3. `COMPREHENSIVE_KEYWORDS` deterministik override olarak korunur.

## Yeniden Kullanılacak Mevcut Kod (referans)

- **Katalog enjeksiyon deseni**: `PipelineConfig.get_collection_catalog()`
  (`src/config/pipeline_loader.py`) — string üretip planner prompt'una `{catalog}` ile
  basılıyor (`Planner._generate_plan`, `src/agent/planner.py`). `get_strategy_catalog()`
  bunu birebir taklit etmeli.
- **Planner prompt + parse**: `PLAN_SYSTEM_PROMPT` (`{catalog}` placeholder) ve
  `Planner._parse_plan()` (`src/agent/planner.py`) — `query_type` zaten burada okunuyor.
- **Yanıt prompt seçimi**: `AnswerTool._select_system_prompt` + `generate(query_type=...)`
  (`src/agent/tools.py`).

---

## Todo

### 1. Playbook dosyası
- [ ] Proje köküne `research_strategies.md` oluştur. Her strateji = parse edilebilir bölüm.
      Önerilen biçim (başlık + basit `key: value` alanları):
      ```markdown
      ## enumerate
      triggers: tüm, hepsi, listele, kaç tane, hangileri
      query_type: comprehensive
      answer_directive: Eldeki TÜM ilgili kayıtları madde madde topla; kısmi kanıttan
        bile sentezle, eksikleri dürüstçe belirt, asla boş dönme.

      ## summarize
      triggers: özetle, özet, kısaca
      query_type: summary
      answer_directive: 3–5 ana bulguyu öz ve yapılandırılmış biçimde ver; en yetkili
        kaynakları önceliklendir.

      ## comparative
      triggers: karşılaştır, farkı, kıyasla
      query_type: comparison
      answer_directive: 2+ pozisyonu/dönemi karşıt köşelerde ele al; benzerlik ve
        farkları belirginleştir.

      ## analytical
      triggers: neden, nasıl, analiz, değerlendir
      query_type: reasoning
      answer_directive: Olayları/alıntıları nedensellikle ilişkilendir; derin, gerekçeli
        sentez yap.

      ## factual
      triggers:
      query_type: fact
      answer_directive: (boş → strict SYS_PROMPT; kanıt yoksa dürüstçe reddet)
      ```

### 2. Loader — `StrategyPlaybook` (`src/config/pipeline_loader.py`)
- [ ] `research_strategies.md`'yi parse eden bir sınıf/fonksiyon ekle (yalnızca stdlib;
      `## ad` bölümleri + `key: value` satırları). Üretecekleri:
      - `catalog_text`: planner prompt'una enjekte edilecek kısa string
        (`- <ad> (<triggers>): <kısa açıklama>` satırları).
      - `by_name: dict[str, {query_type, answer_directive, triggers}]`.
- [ ] **Fail-open**: dosya yoksa boş katalog + boş harita → sistem Faz A davranışına döner.
- [ ] `PipelineConfig.__init__`'te yükle; `get_strategy_catalog()` ve
      `get_strategy(name) -> {...} | None` metotları ekle (mevcut `get_collection_catalog`
      yanında).

### 3. Planner — tek çağrıda strateji seçimi (`src/agent/planner.py`)
- [ ] `PLAN_SYSTEM_PROMPT`'a `{strategy_catalog}` placeholder + "uygun `strategy` adını seç"
      talimatı ekle; JSON çıktısına `"strategy": "<ad>"` alanı ekle.
- [ ] `_generate_plan`'de `get_strategy_catalog()`'u `get_collection_catalog()` yanında
      enjekte et (ek LLM turu YOK).
- [ ] `SearchPlan`'e `strategy: Optional[str]` alanı (`src/agent/schemas.py`);
      `_parse_plan` `plan_data.get("strategy")`'ı okusun.

### 4. Orchestrator — stratejiyi çöz ve uygula (`src/agent/orchestrator.py`)
- [ ] Planlama sonrası: `plan.strategy` playbook'ta varsa → `query_type`'ını ondan al
      (caps + Stage 5.1 döngüsünü zaten query_type sürüyor). `COMPREHENSIVE_KEYWORDS`
      override'ı korunur (keyword eşleşirse strateji=enumerate/comprehensive'e zorla).
- [ ] Seçili stratejinin `answer_directive`'ini state'e taşı (örn. yeni
      `state.answer_directive`) ve Stage 6'da `AnswerTool.generate(...)`'a geçir.
- [ ] Trace: `planning` fazına `strategy` alanını yaz (debug panelinde görünsün).

### 5. AnswerTool — direktif enjeksiyonu (`src/agent/tools.py`)
- [ ] `generate(..., answer_directive: str | None = None)` ekle. Direktif verilmişse
      seçilen sistem promptuna ("\n\nEK YÖNERGE:\n{directive}") eklensin.
- [ ] `_select_system_prompt` mantığını koru; playbook `answer_directive`'i bunun ÜSTÜNE
      biner (sabit `SYNTHESIS_QUERY_TYPES` kümesi, playbook varken direktifle yumuşar).
      Playbook yoksa/`answer_directive` boşsa Faz A davranışı aynen sürer.

### 6. Testler
- [ ] `StrategyPlaybook` parse: md → catalog + by_name; dosya-yok → fail-open (boş).
- [ ] Planner: mock LLM `strategy` emit ederse `SearchPlan.strategy` dolar.
- [ ] Orchestrator: strateji → query_type çözümü; keyword override; `answer_directive`
      `generate()`'e iletiliyor (mock ile doğrula).
- [ ] Regresyon: playbook dosyası yokken tüm mevcut testler yeşil (Faz A davranışı).
- [ ] `pytest -m "not integration and not slow"` tam yeşil (bilinen 5 ingest/OCR hatası hariç).

### 7. Doğrulama + commit
- [ ] Canlı: `research_strategies.md`'ye yeni bir strateji ekle → kod değişmeden planner
      seçsin; web Trace'te `planning.strategy` görünsün; `answer_directive` yanıt tonunu
      değiştirsin. Kapsamlı sorgu hâlâ 15+ chunk toplayıp dolu/sentez yanıt versin.
- [ ] Feature branch → `main` (ff-merge) → push (repo'nun mevcut iş akışı).

## Notlar / Sınırlar
- Playbook yalnızca **prose değil**, `query_type` üzerinden mevcut caps/döngü preset'ine
  bağlanır — yoksa config'te olanın üstüne dolaylama olur.
- Deterministik keyword override, küçük planner modelinin (gemma4:e2b) strateji seçimini
  kaçırdığı durumlar için güvence sağlar.
- Bu iş **retrieval veya answer davranışını yeniden yazmaz**; Faz A'daki altyapıyı
  (comprehensive caps/döngü + SYNTHESIS prompt) düzenlenebilir bir katmana genelleştirir.
