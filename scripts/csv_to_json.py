"""
Convert verbalized CSV to descriptions.json for training.

Usage:
    python csv_to_json.py --input verbalized_features_all.csv --output data/descriptions.json
"""

import argparse
import csv
import json


def convert(input_path: str, output_path: str) -> None:
    descriptions = {}

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["filename"].rsplit(".", 1)[0]  # "clip_001.flac" -> "clip_001"
            desc = row.get("quality_description", "")
            if desc and not desc.startswith("[ERROR]"):
                descriptions[stem] = desc

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2)

    print(f"Converted {len(descriptions)} descriptions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="verbalized_features_all.csv")
    parser.add_argument("--output", default="data/descriptions.json")
    args = parser.parse_args()

    convert(args.input, args.output)
