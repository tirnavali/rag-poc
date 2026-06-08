"""Scrape TBMM HTML minutes from URLs and save as JSON.

Usage:
  python -m scripts.parse_minutes_urls <url1> [<url2> ...] [--output DIR]
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TBMM Tutanak Ayrıştırıcı")
    parser.add_argument("urls", nargs="+", help="URLs of TBMM Tutanak HTML files")
    parser.add_argument("--output", default="tutanak/extracted", help="Output directory")
    args = parser.parse_args()

    from src.trainer.minutes.parse_html import process_url
    output_dir = Path(args.output)
    for url in args.urls:
        process_url(url, output_dir)


if __name__ == "__main__":
    main()
