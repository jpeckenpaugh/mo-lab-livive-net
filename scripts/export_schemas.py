#!/usr/bin/env python3
"""Regenerate committed JSON Schemas for the frozen contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelops_lab.contracts.export import write_schemas

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "contracts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated schemas (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    for path in write_schemas(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
