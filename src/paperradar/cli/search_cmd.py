from __future__ import annotations

import argparse
import json

from ..retrieval.hybrid_search import HybridSearcher
from ..schemas.tools import SearchPapersInput


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="paperradar search",
        description="Search papers in the local database using hybrid retrieval.",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--decision", "-d",
        nargs="+",
        default=None,
        metavar="DECISION",
        help="Filter by decision type(s): oral spotlight poster accepted rejected",
    )
    parser.add_argument(
        "--conference", "-c",
        nargs="+",
        default=None,
        metavar="CONF",
        help="Filter by conference(s): NeurIPS ICML ICLR",
    )
    parser.add_argument(
        "--year", "-y",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Filter by year(s)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args(argv)

    searcher = HybridSearcher()
    inp = SearchPapersInput(
        query=args.query,
        decision_filter=args.decision,
        conference_filter=args.conference,
        year_filter=args.year,
        top_k=args.top_k,
    )
    output = searcher.search(inp)

    if args.json:
        print(json.dumps(output.model_dump(mode="json"), indent=2))
        return

    print(f"\nFound {output.total_found} results for: '{output.query}'\n")
    for i, r in enumerate(output.results, 1):
        authors = ", ".join(r.authors[:3]) + (" et al." if len(r.authors) > 3 else "")
        print(f"{i:>3}. [{r.decision:>10}] {r.title}")
        print(f"      {r.conference} {r.year} | {authors}")
        print(f"      RRF: {r.rrf_score:.4f} | {r.forum_url}")
        print()


if __name__ == "__main__":
    main()
