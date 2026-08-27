#!/usr/bin/env python3
"""Validate the schema, answer offsets and matched pairs in challenge.json."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REQUIRED_FIELDS = {
    "id",
    "pair_id",
    "created_by",
    "question",
    "context",
    "lang",
    "answerable",
    "answer_start",
    "answer",
    "source_url",
}


def validate_records(
    records: Sequence[Mapping[str, Any]], expected_per_member: int = 10
) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    member_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        prefix = f"record {index}"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix}: must be a JSON object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
            continue

        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append(f"{prefix}: id must be a non-empty string")
        elif record_id in ids:
            errors.append(f"{prefix}: duplicate id {record_id!r}")
        else:
            ids.add(record_id)

        pair_id = record["pair_id"]
        creator = record["created_by"]
        if not isinstance(pair_id, str) or not pair_id.strip():
            errors.append(f"{prefix}: pair_id must be a non-empty string")
        else:
            pairs[pair_id].append(record)
        if not isinstance(creator, str) or not creator.strip():
            errors.append(f"{prefix}: created_by must be a non-empty string")
        else:
            member_counts[creator] += 1

        for field in ("question", "context", "lang"):
            if not isinstance(record[field], str) or not record[field].strip():
                errors.append(f"{prefix}: {field} must be a non-empty string")
        if not isinstance(record["source_url"], str) or not re.match(
            r"^https?://", record["source_url"]
        ):
            errors.append(f"{prefix}: source_url must start with http:// or https://")

        answerable = record["answerable"]
        start = record["answer_start"]
        answer = record["answer"]
        context = record["context"]
        if not isinstance(answerable, bool):
            errors.append(f"{prefix}: answerable must be a JSON boolean")
            continue
        if not isinstance(start, int) or isinstance(start, bool):
            errors.append(f"{prefix}: answer_start must be an integer")
            continue
        if not isinstance(answer, str):
            errors.append(f"{prefix}: answer must be a string")
            continue
        if answerable:
            if not answer:
                errors.append(f"{prefix}: answerable record must have a non-empty answer")
            elif start < 0 or context[start : start + len(answer)] != answer:
                errors.append(f"{prefix}: answer and answer_start do not match context")
        elif start != -1 or answer != "":
            errors.append(
                f"{prefix}: unanswerable record must use answer_start=-1 and an empty answer"
            )

    for member, count in sorted(member_counts.items()):
        if count != expected_per_member:
            errors.append(
                f"member {member!r}: expected {expected_per_member} records, found {count}"
            )

    for pair_id, pair_records in sorted(pairs.items()):
        if len(pair_records) != 2:
            errors.append(f"pair {pair_id!r}: expected 2 records, found {len(pair_records)}")
            continue
        for field in ("context", "lang", "created_by", "source_url"):
            if pair_records[0][field] != pair_records[1][field]:
                errors.append(f"pair {pair_id!r}: {field} must be identical")
        if {pair_records[0]["answerable"], pair_records[1]["answerable"]} != {False, True}:
            errors.append(f"pair {pair_id!r}: exactly one record must be answerable")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("challenge.json"))
    parser.add_argument("--expected-per-member", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("challenge file must contain a JSON array")
    errors = validate_records(records, expected_per_member=args.expected_per_member)
    if errors:
        print("Challenge set is invalid:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Challenge set is valid: {len(records)} records.")


if __name__ == "__main__":
    main()
