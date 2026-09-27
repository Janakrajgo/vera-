"""
Generate submission.jsonl — produces composed messages for all 30 test pairs.

Usage:
    # First expand the dataset
    python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded

    # Then generate the submission
    python run_submission.py

    # Or with a custom test-pairs file:
    python run_submission.py --test-pairs dataset/expanded/test_pairs.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from composer import compose, compose_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("vera.submission")


def load_expanded_dataset(expanded_dir: Path) -> dict:
    """Load the expanded dataset into memory."""
    data = {"categories": {}, "merchants": {}, "customers": {}, "triggers": {}}

    # Categories
    cat_dir = expanded_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                cat = json.load(fp)
                data["categories"][cat["slug"]] = cat

    # Merchants
    mx_dir = expanded_dir / "merchants"
    if mx_dir.exists():
        for f in mx_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                mx = json.load(fp)
                data["merchants"][mx["merchant_id"]] = mx

    # Customers
    cx_dir = expanded_dir / "customers"
    if cx_dir.exists():
        for f in cx_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                cx = json.load(fp)
                data["customers"][cx["customer_id"]] = cx

    # Triggers
    trg_dir = expanded_dir / "triggers"
    if trg_dir.exists():
        for f in trg_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                trg = json.load(fp)
                data["triggers"][trg["id"]] = trg

    return data


def load_seed_dataset(seed_dir: Path) -> dict:
    """Load the seed dataset directly (without expansion)."""
    data = {"categories": {}, "merchants": {}, "customers": {}, "triggers": {}}

    # Categories
    cat_dir = seed_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                cat = json.load(fp)
                data["categories"][cat["slug"]] = cat

    # Merchants
    mx_file = seed_dir / "merchants_seed.json"
    if mx_file.exists():
        with open(mx_file, encoding="utf-8") as fp:
            for mx in json.load(fp)["merchants"]:
                data["merchants"][mx["merchant_id"]] = mx

    # Customers
    cx_file = seed_dir / "customers_seed.json"
    if cx_file.exists():
        with open(cx_file, encoding="utf-8") as fp:
            for cx in json.load(fp)["customers"]:
                data["customers"][cx["customer_id"]] = cx

    # Triggers
    trg_file = seed_dir / "triggers_seed.json"
    if trg_file.exists():
        with open(trg_file, encoding="utf-8") as fp:
            for trg in json.load(fp)["triggers"]:
                data["triggers"][trg["id"]] = trg

    return data


async def generate_submission(
    test_pairs: list[dict],
    data: dict,
    output_path: Path,
):
    """Compose messages for each test pair and write submission.jsonl."""
    results = []

    for pair in test_pairs:
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        logger.info(f"Composing {test_id}: trigger={trigger_id} merchant={merchant_id}")

        # Resolve contexts
        trigger = data["triggers"].get(trigger_id)
        if not trigger:
            logger.warning(f"  Trigger {trigger_id} not found — skipping")
            continue

        merchant = data["merchants"].get(merchant_id)
        if not merchant:
            logger.warning(f"  Merchant {merchant_id} not found — skipping")
            continue

        cat_slug = merchant.get("category_slug", "")
        category = data["categories"].get(cat_slug)
        if not category:
            logger.warning(f"  Category {cat_slug} not found — skipping")
            continue

        customer = data["customers"].get(customer_id) if customer_id else None

        # Compose
        try:
            composed = await compose_async(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
            )
        except Exception as e:
            logger.error(f"  Composition failed: {e}")
            composed = {
                "body": "",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Error: {str(e)[:100]}",
            }

        result = {
            "test_id": test_id,
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "none"),
            "send_as": composed.get("send_as", "vera"),
            "suppression_key": composed.get("suppression_key", ""),
            "rationale": composed.get("rationale", ""),
        }
        results.append(result)
        logger.info(f"  ✓ {test_id}: {len(result['body'])} chars, cta={result['cta']}")

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"\nSubmission written: {output_path} ({len(results)} entries)")
    return results


def build_test_pairs_from_seeds(data: dict) -> list[dict]:
    """Build test pairs from the seed triggers (for use without expansion)."""
    pairs = []
    for i, (trg_id, trg) in enumerate(sorted(data["triggers"].items()), 1):
        if i > 30:
            break
        pairs.append({
            "test_id": f"T{i:02d}",
            "trigger_id": trg_id,
            "merchant_id": trg.get("merchant_id", ""),
            "customer_id": trg.get("customer_id"),
        })
    return pairs


async def main():
    parser = argparse.ArgumentParser(description="Generate submission.jsonl")
    parser.add_argument("--test-pairs", type=str, default=None,
                        help="Path to test_pairs.json")
    parser.add_argument("--expanded-dir", type=str, default="dataset/expanded",
                        help="Path to expanded dataset directory")
    parser.add_argument("--seed-dir", type=str, default="dataset",
                        help="Path to seed dataset directory")
    parser.add_argument("--output", type=str, default="submission.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Try loading expanded dataset first, fall back to seeds
    expanded_dir = base_dir / args.expanded_dir
    seed_dir = base_dir / args.seed_dir

    if expanded_dir.exists() and (expanded_dir / "categories").exists():
        logger.info(f"Loading expanded dataset from {expanded_dir}")
        data = load_expanded_dataset(expanded_dir)
    else:
        logger.info(f"Expanded dataset not found; loading seeds from {seed_dir}")
        data = load_seed_dataset(seed_dir)

    logger.info(
        f"Loaded: {len(data['categories'])} categories, "
        f"{len(data['merchants'])} merchants, "
        f"{len(data['customers'])} customers, "
        f"{len(data['triggers'])} triggers"
    )

    # Load or build test pairs
    if args.test_pairs:
        tp_path = Path(args.test_pairs)
    else:
        tp_path = expanded_dir / "test_pairs.json"

    if tp_path.exists():
        with open(tp_path, encoding="utf-8") as f:
            test_pairs = json.load(f).get("pairs", [])
        logger.info(f"Loaded {len(test_pairs)} test pairs from {tp_path}")
    else:
        logger.info("No test_pairs.json found; building from seed triggers")
        test_pairs = build_test_pairs_from_seeds(data)
        logger.info(f"Built {len(test_pairs)} test pairs from seeds")

    # Generate
    output_path = base_dir / args.output
    await generate_submission(test_pairs, data, output_path)


if __name__ == "__main__":
    asyncio.run(main())
