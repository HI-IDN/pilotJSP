#!/usr/bin/env python3
import argparse
from pathlib import Path
from code.utils.convert import pfsp_txt_to_json
from code.utils.enrich import extend_json
import json


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert PFSP txt files to JSON and optionally enrich them."
    )
    parser.add_argument("--input", "-i",
                        help="Input .txt file (PFSP format).")
    parser.add_argument("--output", "-o",
                        help="Output .json file.")
    parser.add_argument("--methods", "-m", nargs="*",
                        help="Optional enrichment methods.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # -------- default behavior if nothing supplied --------
    if not args.input:
        print("No input supplied; running default example conversion.")
        return

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")

    out_path = args.output or in_path.with_suffix(".json")

    # 1. Base conversion
    data = pfsp_txt_to_json(in_path)

    # 2. Optional enrichment
    if args.methods:
        data = extend_json(data, args.methods)

    # 3. Write JSON
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"JSON written to {out_path}")


if __name__ == "__main__":
    main()
