#!/usr/bin/env python3
"""
Merge national ZCTA attributes with population into a CSV.

Inputs expected to exist in the working directory by default:
- zcta_national.txt     (tab-delimited with header; includes GEOID and geometry/centroid fields)
- zcta_population.txt   (Census API-like JSON array-of-arrays; includes NAME, population, and 5-digit ZCTA code)

The script merges on the 5-digit ZCTA code (GEOID) and writes a CSV with all
columns from the national file, plus an added "population" column.

Usage examples:
- Merge all ZCTAs:
    python build_utah_zctas.py

- Merge and write to a custom output csv:
    python build_utah_zctas.py --out merged.csv

- Merge only Utah ZCTAs (ZIP 840-847 prefixes):
    python build_utah_zctas.py --only-utah

- Merge with explicit prefix filters (multiple allowed):
    python build_utah_zctas.py --prefix 840 --prefix 841 --prefix 842 --prefix 843 --prefix 844 --prefix 845 --prefix 846 --prefix 847

- Specify custom input file paths:
    python build_utah_zctas.py --national "National ZCTA/zcta_national.txt" --population "National ZCTA/zcta_population.txt"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


def read_national_tsv(path: str) -> Tuple[List[dict], List[str]]:
    """
    Read the national ZCTA attributes TSV and return (rows, header_columns).

    The TSV is expected to be tab-delimited with a header row. The first column
    must be GEOID (5-digit zero-padded string).
    """
    rows: List[dict] = []
    header: List[str] = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        # csv can handle tabs via delimiter
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        # Normalize header: strip whitespace around names
        header = [h.strip() if h is not None else "" for h in reader.fieldnames]

        # Ensure GEOID exists
        if "GEOID" not in header:
            # Try to find case-insensitively or with odd whitespace
            maybe_geoid = next(
                (h for h in header if h.strip().upper() == "GEOID"), None
            )
            if maybe_geoid is None:
                raise ValueError(
                    f"GEOID column not found in header of {path}. Header: {header}"
                )

        # Read and normalize each row: strip whitespace from values
        for raw in reader:
            row = {}
            for k, v in raw.items():
                k2 = k.strip() if k is not None else ""
                # Some files can place trailing spaces; also keep as strings
                row[k2] = v.strip() if isinstance(v, str) else v
            # Ensure GEOID is zero-padded string of length 5
            geoid = row.get("GEOID", "")
            if geoid is None:
                continue
            geoid = str(geoid).strip()
            # If it's purely digits and length < 5, pad.
            if geoid.isdigit() and len(geoid) < 5:
                geoid = geoid.zfill(5)
            row["GEOID"] = geoid
            if geoid:  # skip empty geoid rows
                rows.append(row)

    return rows, header


def read_population_file(path: str) -> Dict[str, Optional[str]]:
    """
    Read the population mapping from zcta_population.txt and return a dict:
        { '00601': '16721', ... }

    The file is expected to be a JSON array-of-arrays in the shape:
      [["NAME","B01003_001E","zip code tabulation area"],
       ["ZCTA5 00601","16721","00601"], ...]
    But we parse defensively and can handle line-by-line patterns as well.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    content_stripped = content.strip()

    # First try: JSON parse if it looks like a single JSON array
    pop_map: Dict[str, Optional[str]] = {}
    parsed_ok = False

    if content_stripped.startswith("[") and content_stripped.endswith("]"):
        try:
            data = json.loads(content_stripped)
            # Expect a list of rows; first row is header
            for i, row in enumerate(data):
                if not isinstance(row, list) or len(row) < 3:
                    continue
                # Skip header if detected
                if i == 0 and (row[0] == "NAME" or "NAME" in str(row[0]).upper()):
                    continue
                # row format: [ "ZCTA5 00601", "16721", "00601" ]
                zcta = str(row[2]).strip()
                pop_val = row[1]
                # Keep numbers as strings for CSV consistency
                pop_str: Optional[str]
                if pop_val is None:
                    pop_str = None
                else:
                    pop_str = str(pop_val).strip()
                if zcta and len(zcta) == 5 and zcta.isdigit():
                    pop_map[zcta] = pop_str
            parsed_ok = True
        except Exception:
            parsed_ok = False

    # Fallback: parse line-by-line using regex to grab three quoted fields.
    if not parsed_ok:
        # Use regex to capture quoted strings per line
        # Example line: ["ZCTA5 00601","16721","00601"],
        line_re = re.compile(r'"([^"]*)"')
        for line in content.splitlines():
            m = line_re.findall(line)
            if not m:
                continue
            # Skip header-like lines
            if m[0].upper() == "NAME":
                continue
            # Expect at least 3 fields: NAME-like, population, zcta
            if len(m) >= 3:
                zcta = m[2].strip()
                pop_str = m[1].strip()
                if len(zcta) == 5 and zcta.isdigit():
                    pop_map[zcta] = pop_str if pop_str != "" else None

    return pop_map


def filter_by_prefix(rows: List[dict], prefixes: List[str]) -> List[dict]:
    """
    Filter rows where GEOID starts with any of the provided 3-digit prefixes.
    """
    if not prefixes:
        return rows
    normalized = [p.strip() for p in prefixes if p and p.strip()]
    if not normalized:
        return rows
    result = []
    for r in rows:
        geoid = str(r.get("GEOID", "")).strip()
        if any(geoid.startswith(p) for p in normalized):
            result.append(r)
    return result


def merge_rows(
    national_rows: List[dict],
    population_map: Dict[str, Optional[str]],
    population_field_name: str = "population",
) -> List[dict]:
    """
    Return new list of dicts where each row has all national columns plus the population field.
    """
    merged = []
    for r in national_rows:
        geoid = str(r.get("GEOID", "")).strip()
        pop_val = population_map.get(geoid)
        # Copy to avoid mutating input
        out = dict(r)
        out[population_field_name] = pop_val if pop_val is not None else ""
        merged.append(out)
    return merged


def write_csv(path: str, rows: List[dict], header: List[str]) -> None:
    """
    Write rows to CSV at path using the given header column order.
    """
    # Ensure header uniqueness and existence of new population column
    seen = set()
    ordered_header: List[str] = []
    for h in header:
        if h not in seen:
            ordered_header.append(h)
            seen.add(h)
    if "population" not in seen:
        ordered_header.append("population")

    # Write
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_header)
        writer.writeheader()
        for row in rows:
            # Only include keys that are in header; ignore extras
            safe_row = {k: row.get(k, "") for k in ordered_header}
            writer.writerow(safe_row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge national ZCTA attributes with population into a CSV"
    )
    parser.add_argument(
        "--national",
        default="zcta_national.txt",
        help="Path to zcta_national.txt (TSV with GEOID). Default: zcta_national.txt",
    )
    parser.add_argument(
        "--population",
        default="zcta_population.txt",
        help="Path to zcta_population.txt (JSON-like array-of-arrays). Default: zcta_population.txt",
    )
    parser.add_argument(
        "--out",
        default="zcta_merged.csv",
        help="Output CSV path. Default: zcta_merged.csv",
    )
    parser.add_argument(
        "--only-utah",
        action="store_true",
        help="If set, filter to Utah ZCTAs by ZIP prefix 840-847.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="3-digit ZIP prefix filter (e.g., 840). Can be specified multiple times.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Read inputs
    national_rows, national_header = read_national_tsv(args.national)
    population_map = read_population_file(args.population)

    # Determine filters
    prefixes: List[str] = list(args.prefix) if args.prefix else []
    if args.only_utah:
        # Utah ZIP prefixes are 840-847 inclusive
        prefixes.extend([f"{p:03d}" for p in range(840, 848)])
    # Deduplicate prefixes
    prefixes = sorted(set(prefixes))

    if prefixes:
        before = len(national_rows)
        national_rows = filter_by_prefix(national_rows, prefixes)
        after = len(national_rows)
        print(
            f"Filtered national rows by prefixes {prefixes}: {before} -> {after}",
            file=sys.stderr,
        )

    # Merge
    merged_rows = merge_rows(
        national_rows, population_map, population_field_name="population"
    )

    # Diagnostics
    matched = sum(1 for r in merged_rows if str(r.get("population", "")).strip() != "")
    total = len(merged_rows)
    print(
        f"Merged rows: {total} (with population: {matched}, without population: {total - matched})",
        file=sys.stderr,
    )

    # Write output
    write_csv(args.out, merged_rows, national_header)
    print(f"Wrote merged CSV to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
