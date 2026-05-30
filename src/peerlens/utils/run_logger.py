from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class RunLogger:
    """Lightweight per-run logger: node timing + LLM token usage."""

    def __init__(self, run_id: str, log_dir: str = "logs") -> None:
        self._run_id = run_id
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(exist_ok=True)
        self._nodes: list[dict] = []
        self._current: dict | None = None

    # ------------------------------------------------------------------
    # Node timing
    # ------------------------------------------------------------------

    def start_node(self, name: str) -> None:
        self._current = {"node": name, "start": time.time(), "llm_calls": []}

    def end_node(self) -> None:
        if self._current is None:
            return
        elapsed = round(time.time() - self._current["start"], 2)
        self._current["elapsed_s"] = elapsed
        self._nodes.append(self._current)
        self._current = None

    @contextmanager
    def node(self, name: str) -> Iterator[None]:
        self.start_node(name)
        try:
            yield
        finally:
            self.end_node()

    # ------------------------------------------------------------------
    # LLM call recording
    # ------------------------------------------------------------------

    def record_llm(self, input_tokens: int, output_tokens: int, elapsed_s: float) -> None:
        if self._current is None:
            return
        self._current["llm_calls"].append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "elapsed_s": round(elapsed_s, 2),
        })

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------

    def summary_lines(self) -> list[str]:
        lines = [f"[run_logger] run_id={self._run_id}"]
        total_tok = 0
        for n in self._nodes:
            calls = n.get("llm_calls", [])
            in_tok = sum(c["input_tokens"] for c in calls)
            out_tok = sum(c["output_tokens"] for c in calls)
            total_tok += in_tok + out_tok
            lines.append(
                f"  {n['node']:25s}  {n['elapsed_s']:6.1f}s  "
                f"llm_calls={len(calls)}  in={in_tok}  out={out_tok}"
            )
        lines.append(f"  {'TOTAL':25s}  total_tokens={total_tok}")
        return lines

    def print_summary(self) -> None:
        print("\n".join(self.summary_lines()))

    def save(self) -> Path:
        rows = []
        for n in self._nodes:
            calls = n.get("llm_calls", [])
            rows.append({
                "node": n["node"],
                "elapsed_s": n.get("elapsed_s", 0),
                "llm_calls": len(calls),
                "input_tokens": sum(c["input_tokens"] for c in calls),
                "output_tokens": sum(c["output_tokens"] for c in calls),
                "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls),
                "calls_detail": calls,
            })
        path = self._log_dir / f"diagnosis_{self._run_id}.json"
        path.write_text(json.dumps({"run_id": self._run_id, "nodes": rows}, indent=2))
        return path
