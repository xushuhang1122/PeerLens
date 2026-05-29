from __future__ import annotations

import argparse
import sys

from ..config import settings
from ..crawl.pipeline import CrawlPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="peerlens crawl",
        description="Crawl and index papers from OpenReview into the local database.",
    )
    parser.add_argument(
        "--conference", "-c",
        required=True,
        choices=list(settings.conferences.CONFERENCES.keys()),
        help="Conference name",
    )
    parser.add_argument(
        "--year", "-y",
        required=True,
        type=int,
        help="Conference year",
    )
    parser.add_argument(
        "--decision", "-d",
        default=None,
        help="Specific decision to crawl (oral/spotlight/poster). Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-crawl even if data already exists locally.",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Re-index from existing raw JSON without re-crawling.",
    )
    args = parser.parse_args(argv)

    pipeline = CrawlPipeline()

    if args.from_raw:
        print(f"Re-indexing {args.conference} {args.year} from raw data...")
        count = pipeline.load_from_raw(args.conference, args.year)
        print(f"Indexed {count} papers.")
        return

    if not args.force and pipeline.check_local(args.conference, args.year):
        print(f"{args.conference} {args.year} already in local database. Use --force to re-crawl.")
        return

    print(f"Crawling {args.conference} {args.year}" + (f" ({args.decision})" if args.decision else "") + "...")
    count = pipeline.run_sync(args.conference, args.year, decision=args.decision)
    print(f"Done. Indexed {count} papers.")


if __name__ == "__main__":
    main()
