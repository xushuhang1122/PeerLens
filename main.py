from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: paperradar <command> [options]")
        print("Commands: crawl, search, agent")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "crawl":
        from src.paperradar.cli.crawl_cmd import main as crawl_main
        crawl_main(args)
    elif command == "search":
        from src.paperradar.cli.search_cmd import main as search_main
        search_main(args)
    elif command == "agent":
        from src.paperradar.cli.agent_cmd import main as agent_main
        agent_main(args)
    else:
        print(f"Unknown command: {command}")
        print("Commands: crawl, search, agent")
        sys.exit(1)


if __name__ == "__main__":
    main()
