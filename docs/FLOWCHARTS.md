<!-- Preview Shortcut: Cmd + Shift + V (or Cmd + K, V for side-by-side) -->
# Flowcharts

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph UI["UI Layer - src/ui/"]
        Chat["chat.py<br/>Rich Terminal Chat"]
        Commands["commands.py<br/>/müfettiş, /kaynak, /rapor, /debug"]
    end

    subgraph MCP["MCP Layer - src/mcp/"]
        RouterServer["router_server.py<br/>FastAPI + MCP"]
        PressServer["press_server.py"]
        MinutesServer["minutes_server.py"]
    end

    subgraph Generator["Generator Layer - src/generator/"]
        RAGService["RAGService<br/>Main Facade"]
        OllamaGen["OllamaGenerator<br/>Streaming LLM"]
        DeepPipeline["DeepPipeline<br/>Müfettiş/Rapor Mode"]
    end

    subgraph Retriever["Retriever Layer - src/retriever/"]
        VectorRetriever["VectorRetriever<br/>Date filter + post-process"]
        VectorSearch["VectorSearch<br/>ChromaDB + rerank"]
        Reranker["CrossEncoderReranker<br/>mmarco-mMiniLMv2"]
        Context["build_context()"]
    end

    subgraph Trainer["Trainer Layer - src/trainer/"]
        IngestPress["press_clips/<br/>CSV → SQLite → FTS → Chroma"]
        IngestUnified["ingestion/<br/>IngestionPipeline"]
    end

    subgraph Evaluator["Evaluator Layer - src/evaluator/"]
        Benchmark["run_benchmark.py"]
        IndexHealth["index_health.py"]
        Judge["LLM Judge"]
    end

    subgraph Config["Config - src/config/"]
        Settings["settings.py"]
        Collections["collections.py<br/>CollectionSpec registry"]
    end

    subgraph Storage["Storage - data_lake/"]
        PressDB[("press_clips.db<br/>kupurler + FTS5")]
        MinutesDB[("parliament_digital_born_minutes.db")]
        PressVec[("press_clips_vectors/<br/>gazete_arsivi")]
        MinutesVec[("parliament_digital_born_minutes_vectors/<br/>tbmm_minutes")]
        OnergeVec[("onerge_vectors/<br/>tbmm_onerge")]
        Manifest[("document_manifest.db")]
    end

    subgraph LLM["External / Local Services"]
        Ollama["Ollama Server"]
        EmbedModel["nomic-embed-text-v2-moe<br/>(Ollama) or Jina v3/v4 (local)"]
        LLMModel["gemma4:latest"]
    end

    Chat --> Commands
    Commands --> RAGService
    RouterServer --> RAGService
    RouterServer --> DeepPipeline
    PressServer --> RAGService
    MinutesServer --> RAGService

    RAGService --> VectorRetriever
    RAGService --> OllamaGen
    RAGService --> DeepPipeline
    DeepPipeline --> OllamaGen
    DeepPipeline --> VectorRetriever

    VectorRetriever --> VectorSearch
    VectorSearch --> Reranker
    VectorRetriever --> Context

    VectorSearch --> PressVec
    VectorSearch --> MinutesVec
    VectorSearch --> OnergeVec
    VectorSearch --> EmbedModel
    OllamaGen --> Ollama

    Collections --> VectorRetriever
    Settings --> Collections

    IngestPress --> PressDB
    IngestPress --> PressVec
    IngestUnified --> MinutesVec
    IngestUnified --> OnergeVec
    IngestUnified --> Manifest

    Benchmark --> VectorRetriever
    IndexHealth --> MinutesVec
```

---

## 2. Query Flow (Normal Mode)

```mermaid
flowchart TD
    Start(["User types question"]) --> Parse["parse_command(raw)"]
    Parse --> ActionCheck{Action?}

    ActionCheck -->|NORMAL_QUERY| Retrieve["RAGService.retrieve(query)"]
    ActionCheck -->|Other| OtherActions["Handle /müfettiş, /kaynak, etc."]

    Retrieve --> ExtractDates["extract_dates(query)<br/>Parse Turkish/ISO dates"]
    ExtractDates --> YearFilter["Build ChromaDB year filter<br/>where_year_filter()"]

    YearFilter --> CheckReranker{USE_RERANKER?<br/>default: on}
    CheckReranker -->|Yes| InitReranker["Init CrossEncoderReranker"]
    CheckReranker -->|No| SkipReranker["Skip reranking"]
    InitReranker --> Embed
    SkipReranker --> Embed

    Embed["embed_query(query)<br/>via CollectionSpec embedder"] --> ChromaQuery["ChromaDB.query()<br/>fetch_k=100 candidates"]
    ChromaQuery --> RerankCheck{Reranker?}

    RerankCheck -->|Yes| Rerank["CrossEncoder.rerank()<br/>Score & sort candidates"]
    RerankCheck -->|No| RawResults["Use raw cosine distances"]
    Rerank --> PostProcess
    RawResults --> PostProcess

    PostProcess["For each result:<br/>extract_relevant_windows()<br/>Prepend metadata header"] --> BuildRetrieval["RetrievalResult<br/>docs, metadatas, distances"]

    BuildRetrieval --> BuildContext["build_context()<br/>Filter dist > 1.8<br/>4K per chunk, 12K total"]
    BuildContext --> Stream["OllamaGenerator.stream()<br/>context + query"]
    Stream --> OllamaChat["ollama.Client.chat(stream=True)<br/>gemma4:latest"]
    OllamaChat --> Display["Rich Live Display<br/>thinking + content panels"]
    Display --> Sources["Print source cards<br/>Update chat_history"]
```

---

## 3. Query Flow (Müfettiş / Deep Research Mode)

```mermaid
flowchart TD
    Start(["/müfettiş <query>"]) --> Parse["parse_command() -> MUFETTIS"]
    Parse --> AskStream["RAGService.ask_stream(mufettis_mode=True)"]

    AskStream --> DeepPipeline["DeepPipeline(service).run(query)"]

    DeepPipeline --> Expand["expand_query(query)<br/>LLM generates related keywords"]
    Expand --> Combined["combined = query + expanded"]

    Combined --> Retrieve["VectorRetriever.retrieve()<br/>top_k=40, fetch_k=150"]
    Retrieve --> ExtractDates["extract_dates() + year filter"]
    ExtractDates --> Embed["embed_query(combined)<br/>via CollectionSpec embedder"]
    Embed --> ChromaQuery["ChromaDB.query(n=150)"]
    ChromaQuery --> Rerank["CrossEncoderReranker.rerank()"]
    Rerank --> PostProcess["extract_relevant_windows()<br/>metadata headers"]
    PostProcess --> BuildRetrieval["RetrievalResult"]

    BuildRetrieval --> BuildContext["build_context()<br/>8K per chunk, 25K total"]
    BuildContext --> Stream["OllamaGenerator.stream()<br/>MUFETTIS_SYS_PROMPT<br/>temp=0.2, tokens=16384"]

    Stream --> Display["Rich Live Display"]
    Display --> ReportCheck{Action = RAPOR?}

    ReportCheck -->|Yes| SaveReport["save_report()<br/>artifacts/reports/<br/>Markdown + frontmatter"]
    ReportCheck -->|No| Done(["Done"])
    SaveReport --> Done
```

---

## 4. Data Ingestion Pipeline (Press Clips)

```mermaid
flowchart TD
    Start(["src.trainer.ingestion.ingest"]) --> StepCheck{--step?}

    StepCheck -->|csv| CSV["Step 1: CSV -> SQLite"]
    StepCheck -->|fts| FTS["Step 2: Build FTS5"]
    StepCheck -->|embed| Embed["Step 3: Embed & Index"]
    StepCheck -->|None| All["Run all steps"]

    CSV --> ReadCSV["pandas.read_csv()<br/>gazete-rag-001.csv"]
    ReadCSV --> Normalize["Normalize TARIH<br/>-> YYYY-MM-DD"]
    Normalize --> ToSQL["df.to_sql('kupurler')<br/>data_lake/press_clips.db"]

    ToSQL --> FTS
    All --> CSV

    FTS --> CreateFTS["CREATE VIRTUAL TABLE<br/>kupurler_fts USING fts5()"]
    CreateFTS --> PopulateFTS["INSERT INTO kupurler_fts<br/>SELECT FROM kupurler"]
    PopulateFTS --> Triggers["Create sync triggers<br/>INSERT/UPDATE/DELETE"]

    Triggers --> Embed
    All --> FTS

    Embed --> BatchLoop["For each row (batches of 20)"]
    BatchLoop --> BuildPrefix["Build prefix:<br/>Gazete | Tarih | Yazar | Baslik"]
    BuildPrefix --> Chunk["Split text<br/>1500 chars, 150 overlap"]
    Chunk --> GenIDs["Generate IDs:<br/>{KAYIT_NO}_{chunk_index}"]
    GenIDs --> EmbedChunks["OllamaEmbeddings<br/>nomic-embed-text-v2-moe"]
    EmbedChunks --> Upsert["ChromaDB.upsert()<br/>press_clips_vectors/gazete_arsivi"]
    Upsert --> NextBatch{More rows?}
    NextBatch -->|Yes| BatchLoop
    NextBatch -->|No| Done(["Indexing complete"])
```

---

## 5. Unified Ingestion Pipeline

```mermaid
flowchart TD
    Start(["IngestionPipeline.run_document()"]) --> ManifestCheck["Query DocumentManifest<br/>data_lake/document_manifest.db"]

    ManifestCheck --> HashCheck{content_hash<br/>matches?}
    HashCheck -->|Yes + done| SKIP(["SKIP - already indexed"])
    HashCheck -->|No or changed| URLCheck{URL source?}

    URLCheck -->|Yes| HEAD["HTTP HEAD check<br/>ETag / Last-Modified"]
    URLCheck -->|No| Parse
    HEAD --> Changed{Content<br/>changed?}
    Changed -->|No| SKIP
    Changed -->|Yes| DeleteOld["Delete old chunks<br/>from ChromaDB"]
    DeleteOld --> Parse

    Parse["Adapter.parse()"] --> AdapterCheck{source_type?}

    AdapterCheck -->|tutanak| Tutanak["TutanakPdfAdapter<br/>Docling OCR"]
    AdapterCheck -->|press_clip| Press["PressClipAdapter<br/>Inline text"]
    AdapterCheck -->|pdf_report| PDF["PdfReportAdapter<br/>Docling parser"]
    AdapterCheck -->|kanun_teklifi| Kanun["KanunTeklifiAdapter"]
    AdapterCheck -->|onerge| Onerge["OnergeAdapter<br/>Docling parser"]

    Tutanak --> Chunks["full_text + chunks[]<br/>{text, span, metadata}"]
    Press --> Chunks
    PDF --> Chunks
    Kanun --> Chunks
    Onerge --> Chunks

    Chunks --> ChunkIDs["Deterministic IDs:<br/>{document_id}_{chunk_index}"]

    ChunkIDs --> LateCheck{CollectionSpec<br/>supports_late_chunking?}

    LateCheck -->|Yes - Jina v3/v4| LateChunk["LocalLateChunkingEmbedder<br/>Full doc -> mean-pool per span"]
    LateCheck -->|No - Nomic| StdEmbed["OllamaEmbeddings<br/>Per-chunk embedding"]

    LateChunk --> Upsert["ChromaDB.upsert()<br/>ids, embeddings, docs, metadatas"]
    StdEmbed --> Upsert

    Upsert --> ManifestUpdate["DocumentManifest.update()<br/>status=done, chunk_count, hash"]
    ManifestUpdate --> DONE(["Complete"])
```

---

## 6. Chat UI Command Flow

```mermaid
flowchart TD
    Start(["main() loop"]) --> Input["Prompt: 'Soru:' "]
    Input --> Parse["parse_command(raw)"]

    Parse --> Check{Action?}

    Check -->|EMPTY| Start
    Check -->|EXIT| Goodbye["Print goodbye<br/>break"]
    Check -->|CLEAR| Clear["console.clear()<br/>print_banner()"]
    Check -->|TOGGLE_DEBUG| Debug["Toggle state.debug_mode"]
    Check -->|SHOW_HELP| Help["Print help panel"]
    Check -->|UNKNOWN| Unknown["Print 'Bilinmeyen komut'"]

    Check -->|SHOW_SOURCE| Source["inspect_record()<br/>Display full DB record"]
    Check -->|NORMAL_QUERY| Normal["Normal RAG flow<br/>top_k=5, 4K/12K context"]
    Check -->|MUFETTIS| Mufettis["Deep research<br/>top_k=40, 8K/25K context"]
    Check -->|RAPOR| Rapor["Deep research +<br/>save Markdown report"]

    Clear --> Start
    Debug --> Start
    Help --> Start
    Unknown --> Start
    Source --> Start

    Normal --> Retrieve["service.retrieve()"]
    Mufettis --> DeepRetrieve["expand_query() + retrieve(top_k=40)"]
    Rapor --> DeepRetrieve

    Retrieve --> Context["build_context()"]
    DeepRetrieve --> Context

    Context --> Stream["service.ask_stream()"]
    Stream --> Display["Rich Live Display"]
    Display --> PostProcess

    PostProcess --> DebugCheck{debug_mode?}
    DebugCheck -->|Yes| DebugInfo["Print chunk scores<br/>+ context info"]
    DebugCheck -->|No| SkipDebug

    DebugInfo --> PrintSources["Print source cards"]
    SkipDebug --> PrintSources

    PrintSources --> RaporCheck{RAPOR?}
    RaporCheck -->|Yes| SaveReport["save_report() -> artifacts/reports/"]
    RaporCheck -->|No| UpdateState

    SaveReport --> UpdateState["Update state:<br/>chat_history + last_sources"]
    UpdateState --> Start
```

---

## 7. Component Relationship Diagram

```mermaid
classDiagram
    class RAGService {
        +retrieve(query, mufettis_mode) RetrievalResult
        +ask_stream(query, mufettis_mode) StreamChunk[]
        +build_context(results) str
        +inspect_record(db, chunk_id) dict
    }

    class VectorRetriever {
        +retrieve(query, top_k, fetch_k) RetrievalResult
        -spec: CollectionSpec
        -search: VectorSearch
    }

    class VectorSearch {
        +search(query, top_k, fetch_k) list[dict]
        -embedder: Embedder
        -collection: Collection
    }

    class CrossEncoderReranker {
        +rerank(pairs) list[tuple]
        -model: CrossEncoder
    }

    class OllamaGenerator {
        +stream(query, context, mufettis_mode) StreamChunk[]
        +expand_query(query) str
        +answer(query, context) str
        -client: ollama.Client
    }

    class DeepPipeline {
        +run(query) StreamChunk[]
        +run_blocking(query) ReportResult
        +retrieve_only(query) ReportResult
        -service: RAGService
    }

    class build_context {
        +build_context(results, max_chars, total_max_chars) str
    }

    class ChatState {
        +debug_mode: bool
        +chat_history: list
        +last_sources: list
    }

    class CollectionSpec {
        +name: str
        +db_path: Path
        +embed_model: str
        +supports_late_chunking: bool
        +source_type: str
    }

    class IngestionPipeline {
        +run_document(input) IngestResult
        +run_batch(inputs) list[IngestResult]
        -spec: CollectionSpec
    }

    class DocumentManifest {
        +check(hash) Record
        +update(hash, status, count)
    }

    class RouterServer {
        +api_search(req) SearchResponse
        +api_report(req) ReportResponse
    }

    RAGService --> VectorRetriever
    RAGService --> OllamaGenerator
    RAGService --> DeepPipeline
    DeepPipeline --> RAGService
    DeepPipeline --> OllamaGenerator
    DeepPipeline --> VectorRetriever
    VectorRetriever --> VectorSearch
    VectorRetriever --> build_context
    VectorRetriever --> CollectionSpec
    VectorSearch --> CrossEncoderReranker
    IngestionPipeline --> DocumentManifest
    IngestionPipeline --> CollectionSpec
    RouterServer --> RAGService
    RouterServer --> DeepPipeline
```

---

## 8. Data Storage Layout

```mermaid
graph LR
    subgraph DataLake["data_lake/"]
        PressDB[("press_clips.db<br/>kupurler table<br/>kupurler_fts")]
        MinutesDB[("parliament_digital_born_minutes.db<br/>parliament_minutes table")]
        PressVec[("press_clips_vectors/<br/>ChromaDB<br/>collection: gazete_arsivi")]
        MinutesVec[("parliament_digital_born_minutes_vectors/<br/>ChromaDB<br/>collection: tbmm_minutes")]
        OnergeVec[("onerge_vectors/<br/>ChromaDB<br/>collection: tbmm_onerge")]
        Manifest[("document_manifest.db<br/>Dedup & status tracking")]
        Downloads["downloads/<br/>PDF cache"]
        ParseCache["parse_cache/<br/>Docling parse cache"]
    end

    subgraph Artifacts["artifacts/"]
        Reports["reports/<br/>Markdown reports<br/>from /rapor"]
    end

    PressDB --- PressVec
    MinutesDB --- MinutesVec
    Manifest --- MinutesVec
    Manifest --- OnergeVec
    Downloads --- ParseCache
```
