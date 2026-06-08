"""Root-level entry-point shim — real logic lives in src/ui/chat.py."""
import argparse
from src.ui.chat import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Archive Chat Interface")
    parser.add_argument("--agent", action="store_true", help="Enable Planning Agent mode")
    parser.add_argument("--pipeline", type=str, default=None, help="Path to pipeline.yaml")
    args = parser.parse_args()
    main(agent_mode=args.agent, pipeline_path=args.pipeline)
