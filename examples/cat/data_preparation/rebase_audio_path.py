#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

OLD_PREFIX = "/inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/audioset/wav_16k"
NEW_PREFIX = "/inspire/dataset/audioset/v2/wav_16k"


def rewrite_wav_path(obj, old_prefix, new_prefix):
    """
    Rewrite wav_path prefix if matched, otherwise keep unchanged.
    """
    if "wav_path" in obj and isinstance(obj["wav_path"], str):
        if obj["wav_path"].startswith(old_prefix):
            obj["wav_path"] = obj["wav_path"].replace(old_prefix, new_prefix, 1)
    return obj


def main(input_json, output_json, old_prefix, new_prefix):
    input_json = Path(input_json)
    output_json = Path(output_json)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a JSON object")
        rewrite_wav_path(item, old_prefix, new_prefix)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done. Processed {len(data)} entries.")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rewrite wav_path prefix in AudioSet-style JSON manifest"
    )
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--old-prefix",
        default=OLD_PREFIX,
        help=f"Old wav_path prefix (default: {OLD_PREFIX})",
    )
    parser.add_argument(
        "--new-prefix",
        default=NEW_PREFIX,
        help=f"New wav_path prefix (default: {NEW_PREFIX})",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.old_prefix, args.new_prefix)
