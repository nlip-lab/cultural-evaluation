"""
Download and preprocess all 4 source datasets for the CSI detection pipeline.

Datasets:
  1. GSM8K          (openai/gsm8k)               — 8,792 items  — Educational/Math
  2. ROCStories     (mintujupally/ROCStories)     — 98,161 items — Stories
  3. DailyDialog    (roskoN/dailydialog)          — 13,118 convos — Dialogue
  4. RecipeNLG      (Kaggle: paultimothymooney/recipenlg) — 2,231,142 items — Procedural

Each dataset is saved as a JSONL file under 03_data/raw/<name>/<name>.jsonl
with a unified schema: { id, split, index, text, source, domain }.
RecipeNLG items additionally carry: { title, ingredients, directions }.
"""

import ast
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parents[3] / "03_data" / "raw"

csv.field_size_limit(sys.maxsize)


def save_jsonl(items: list[dict], path: Path):
    """Write a list of records to UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(items):,} items → {path}")


def download_gsm8k():
    """Download GSM8K and export normalized JSONL rows."""
    print("\n" + "=" * 60)
    print("  Downloading GSM8K  (openai/gsm8k)")
    print("=" * 60)

    ds = load_dataset("openai/gsm8k", "main")
    items = []
    for split_name in ds:
        for idx, row in enumerate(ds[split_name]):
            items.append({
                "id": f"gsm8k_{split_name}_{idx}",
                "split": split_name,
                "index": idx,
                "text": row["question"],
                "source": "openai/gsm8k",
                "domain": "educational",
            })
    save_jsonl(items, BASE_DIR / "gsm8k" / "gsm8k.jsonl")
    for s in ds:
        print(f"    {s}: {len(ds[s]):,} items")
    print(f"    Total: {len(items):,}")
    return items


def download_rocstories():
    """Download ROCStories and export normalized JSONL rows."""
    print("\n" + "=" * 60)
    print("  Downloading ROCStories  (mintujupally/ROCStories)")
    print("=" * 60)

    ds = load_dataset("mintujupally/ROCStories")
    items = []
    for split_name in ds:
        for idx, row in enumerate(ds[split_name]):
            items.append({
                "id": f"rocstories_{split_name}_{idx}",
                "split": split_name,
                "index": idx,
                "text": row["text"],
                "source": "mintujupally/ROCStories",
                "domain": "stories",
            })
    save_jsonl(items, BASE_DIR / "rocstories" / "rocstories.jsonl")
    for s in ds:
        print(f"    {s}: {len(ds[s]):,} items")
    print(f"    Total: {len(items):,}")
    return items


def download_dailydialog():
    """Download DailyDialog parquet splits and export normalized JSONL rows."""
    print("\n" + "=" * 60)
    print("  Downloading DailyDialog  (roskoN/dailydialog)")
    print("=" * 60)

    ds = load_dataset("parquet", data_files={
        "train": "hf://datasets/roskoN/dailydialog@refs/convert/parquet/full/train/0000.parquet",
        "validation": "hf://datasets/roskoN/dailydialog@refs/convert/parquet/full/validation/0000.parquet",
        "test": "hf://datasets/roskoN/dailydialog@refs/convert/parquet/full/test/0000.parquet",
    })
    items = []
    for split_name in ds:
        for idx, row in enumerate(ds[split_name]):
            utterances = row["utterances"]
            lines = []
            for i, utt in enumerate(utterances):
                speaker = "A" if i % 2 == 0 else "B"
                lines.append(f"Speaker {speaker}: {utt.strip()}")
            dialogue_text = "\n".join(lines)
            items.append({
                "id": f"dailydialog_{split_name}_{idx}",
                "split": split_name,
                "index": idx,
                "text": dialogue_text,
                "source": "roskoN/dailydialog",
                "domain": "dialogue",
            })
    save_jsonl(items, BASE_DIR / "dailydialog" / "dailydialog.jsonl")
    for s in ds:
        print(f"    {s}: {len(ds[s]):,} items")
    print(f"    Total: {len(items):,}")
    return items


def _parse_list_field(raw: str) -> list[str]:
    """Parse a stringified Python list field into a real list."""
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [raw.strip()]


def download_recipenlg():
    """Download RecipeNLG from Kaggle, normalize fields, and export JSONL."""
    print("\n" + "=" * 60)
    print("  Downloading RecipeNLG  (Kaggle: paultimothymooney/recipenlg)")
    print("=" * 60)

    out_dir = BASE_DIR / "recipenlg"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "RecipeNLG_dataset.csv"

    if not csv_path.exists():
        print("  Downloading from Kaggle ...")
        subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "paultimothymooney/recipenlg",
             "-p", str(out_dir)],
            check=True,
        )
        zips = list(out_dir.glob("*.zip"))
        if zips:
            print(f"  Unzipping {zips[0].name} ...")
            with zipfile.ZipFile(zips[0], "r") as zf:
                zf.extractall(out_dir)
            zips[0].unlink()
        csvs = list(out_dir.glob("*.csv"))
        if csvs:
            csv_path = csvs[0]
        else:
            print("  ERROR: No CSV found after unzipping!")
            sys.exit(1)
    else:
        print(f"  CSV already exists: {csv_path}")

    print(f"  Parsing {csv_path.name} ...")
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            title = row.get("title", "").strip()
            ingredients = _parse_list_field(row.get("ingredients", ""))
            directions = _parse_list_field(row.get("directions", ""))

            ing_str = ", ".join(ingredients)
            dir_str = " ".join(
                f"{i+1}. {step}" for i, step in enumerate(directions)
            )
            recipe_text = (
                f"Title: {title}\n"
                f"Ingredients: {ing_str}\n"
                f"Directions: {dir_str}"
            )
            items.append({
                "id": f"recipenlg_train_{idx}",
                "split": "train",
                "index": idx,
                "text": recipe_text,
                "title": title,
                "ingredients": ingredients,
                "directions": directions,
                "source": "kaggle/paultimothymooney/recipenlg",
                "domain": "procedural",
            })
            if (idx + 1) % 500_000 == 0:
                print(f"    ... parsed {idx + 1:,} rows")

    save_jsonl(items, out_dir / "recipenlg.jsonl")
    print(f"    Total: {len(items):,}")
    return items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for CSI pipeline")
    parser.add_argument("--datasets", nargs="+",
                        default=["gsm8k", "rocstories", "dailydialog", "recipenlg"],
                        choices=["gsm8k", "rocstories", "dailydialog", "recipenlg"])
    args = parser.parse_args()

    print("=" * 60)
    print("  DATASET DOWNLOADER — CSI Detection Pipeline")
    print("=" * 60)
    print(f"  Output dir: {BASE_DIR}")
    print(f"  Datasets : {args.datasets}")

    funcs = {
        "gsm8k": download_gsm8k,
        "rocstories": download_rocstories,
        "dailydialog": download_dailydialog,
        "recipenlg": download_recipenlg,
    }

    for ds_name in args.datasets:
        funcs[ds_name]()

    print("\n" + "=" * 60)
    print("  ALL DATASETS DOWNLOADED")
    print("=" * 60)
    for d in args.datasets:
        p = BASE_DIR / d
        if p.exists():
            for f in p.glob("*"):
                size_mb = f.stat().st_size / 1e6
                print(f"  {f.name:30s}  {size_mb:.1f} MB")
    print()
