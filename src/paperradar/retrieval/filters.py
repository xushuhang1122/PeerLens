from __future__ import annotations

from typing import Optional


def build_where(
    decision_filter: Optional[list[str]] = None,
    conference_filter: Optional[list[str]] = None,
    year_filter: Optional[list[int]] = None,
) -> Optional[dict]:
    clauses = []

    if decision_filter:
        clauses.append({"decision": {"$in": decision_filter}})

    if conference_filter:
        clauses.append({"conference": {"$in": conference_filter}})

    if year_filter:
        if len(year_filter) == 1:
            clauses.append({"year": year_filter[0]})
        else:
            clauses.append({"year": {"$in": year_filter}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
