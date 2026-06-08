from src.config.collections import get_spec
from src.retriever.vector_retriever import VectorRetriever

spec = get_spec('tbmm_tutanaklar_docling_jina_v3_4k')
retriever = VectorRetriever(spec)

sessions = [
    "tutanak-20-01-03-19960123",
    "tutanak-20-01-04-19960124",
    "tutanak-20-01-02-19960118"
]

for sess in sessions:
    print(f"\n=== Analyzing Session: {sess} ===")
    for i in range(1, 20, 5):
        chunk_id = f"{sess}_{i}"
        rec = retriever.inspect_record("tutanak", chunk_id)
        if rec:
            content = rec.get("content", "")
            print(f"[{chunk_id}] {content[:250]}...")
