from __future__ import annotations

import argparse
import json
import sys
import uuid

from ..agent.runner import run_agent, stream_agent
from ..memory.episodic import EpisodicMemory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="paperradar agent",
        description="Run the research agent on a question.",
    )
    parser.add_argument("query", help="Research question or task")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream agent events to stdout",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output final report as JSON",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID for memory tracking (auto-generated if not provided)",
    )
    args = parser.parse_args(argv)

    session_id = args.session or str(uuid.uuid4())[:8]
    episodic = EpisodicMemory()
    episodic.record_query(args.query, session_id=session_id)

    print(f"[session:{session_id}] Running agent for: {args.query}\n")

    if args.stream:
        for event in stream_agent(args.query, session_id=session_id):
            msgs = event.get("messages", []) if isinstance(event, dict) else []
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", "")
                if content:
                    print(content)
        return

    report = run_agent(args.query, session_id=session_id)

    if report is None:
        print("Agent did not produce a report.")
        sys.exit(1)

    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return

    print("=" * 60)
    print(report.title)
    print("=" * 60)
    print(report.executive_summary)
    if report.search_results:
        print(f"\nSearch results: {report.search_results.total_found} papers")
        for r in report.search_results.results[:5]:
            print(f"  - [{r.decision}] {r.title} ({r.conference} {r.year})")
    if report.gap_report:
        print(f"\nResearch gaps identified: {len(report.gap_report.gaps)}")
        for g in report.gap_report.gaps[:3]:
            print(f"  - {g.gap_description}")
        print(f"\nSubmission advice: {report.gap_report.submission_advice}")
    print()


if __name__ == "__main__":
    main()
