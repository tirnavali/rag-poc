"""Manual smoke-test harness with hardcoded queries."""
import argparse
import sys


def run_standard():
    from src.generator.service import RAGService
    rag = RAGService()

    test_queries = [
        "1997 yılındaki Türkleşmek ve İslamlaşmak hakkındaki köşe yazısı ne anlatıyor?",
        "Ahmet Yılmaz hangi konularda yazılar yazmış?",
        "2025 yılındaki seçim sonuçları nelerdir?"
    ]

    DEBUG = True

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}")
        print(f"SORU: {q}")
        try:
            response = rag.ask(q, debug=DEBUG)
            print("\nYANIT:")
            print(response)
        except TimeoutError as e:
            print("\nZAMAN AŞIMI:")
            print(str(e))
        except Exception as e:
            print("\nHATA:")
            print(type(e).__name__, "-", str(e))
        print(f"{'=' * 80}")


def run_agent(pipeline_path: str | None = None):
    from src.generator.service import RAGService
    rag = RAGService(pipeline_config_path=pipeline_path)

    test_queries = [
        "1997 yılındaki Türkleşmek ve İslamlaşmak hakkındaki köşe yazısı ne anlatıyor?",
        "Ahmet Yılmaz hangi konularda yazılar yazmış?",
        "2025 yılındaki seçim sonuçları nelerdir?"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i} (AGENT MODE)")
        print(f"SORU: {q}")
        try:
            output = rag.run_agent(q)
            print(f"\nPLAN:")
            if output.plan:
                print(f"  intent: {output.plan.intent}")
                print(f"  resources: {', '.join(r.collection for r in output.plan.resources)}")
                print(f"  reasoning: {output.plan.reasoning}")
            print(f"\nYANIT:")
            print(output.answer)
            if output.thinking:
                print(f"\nDÜŞÜNCE:")
                print(output.thinking[:500])
            if output.validation:
                v = output.validation
                print(f"\nDOĞRULAMA: {'GEÇTI' if v.passes else 'BAŞARISIZ'}")
                if v.issues:
                    print(f"  sorunlar: {v.issues}")
            print(f"\nTRACE:")
            for ev in output.trace:
                print(f"  [{ev.phase}] {ev.latency_ms:.0f}ms | {ev.details}")
            print(f"\nKAYNAKLAR: {len(output.sources)} belge")
        except Exception as e:
            print("\nHATA:")
            print(type(e).__name__, "-", str(e))
            import traceback
            traceback.print_exc()
        print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Smoke Test")
    parser.add_argument("--agent", action="store_true", help="Enable Planning Agent mode")
    parser.add_argument("--pipeline", type=str, default=None, help="Path to pipeline.yaml")
    args = parser.parse_args()

    if args.agent:
        run_agent(pipeline_path=args.pipeline)
    else:
        run_standard()
