#!/usr/bin/env python3
"""
Extract all `"value"` fields from a jsonl file and write them to a text file.

Usage
-----
$ python jsonl_extract.py input.jsonl output.txt
"""

import argparse
import json
from pathlib import Path


def extract_values(jsonl_path: Path, txt_path: Path) -> None:
    """
    Stream the jsonl file, gather every `conversations[*].value`,
    and write them to `txt_path`, one per line.
    """
    with jsonl_path.open("r", encoding="utf-8") as src, \
            txt_path.open("w", encoding="utf-8") as dst:

        for raw in src:
            if not raw.strip():          # skip blank lines, if any
                continue
            obj = json.loads(raw)

            # Safely walk the "conversations" list (if present)
            for message in obj.get("conversations", []):
                value = message.get("value")
                if value is not None:
                    # You can choose to keep embedded new‑lines;
                    # here we flatten them so every entry is on one line.
                    dst.write(value.replace("\n", " ") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract conversation values from a jsonl file.")
    parser.add_argument("input", type=Path, help="Path to the .jsonl file")
    parser.add_argument("output", type=Path, help="Path for the resulting .txt file")
    args = parser.parse_args()

    extract_values(args.input, args.output)
    print(f"✓ Extracted values written to {args.output}")


if __name__ == "__main__":
    main()
