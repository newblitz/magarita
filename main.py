import pandas as pd
import os
import re
from collections import Counter

BASE = "dataset"

FILES = {
    "train_s1": f"{BASE}/train/train_source1.tsv",
    "train_s2": f"{BASE}/train/train_source2.tsv",
    "train_s3": f"{BASE}/train/train_source3.tsv",
    "ground_truth": f"{BASE}/train/train_ground_truth.tsv",
    "test_s1": f"{BASE}/test/test_source1.tsv",
    "test_s2": f"{BASE}/test/test_source2.tsv",
    "test_s3": f"{BASE}/test/test_source3.tsv",
}


def basic_stats(path, name, chunksize=200_000):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(path)

    total = 0
    missing = Counter()
    countries = Counter()
    name_lengths = []
    address_lengths = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False
    ):

        total += len(chunk)

        for col in chunk.columns:
            missing[col] += (chunk[col].str.strip() == "").sum()

        if "country" in chunk.columns:
            countries.update(chunk["country"].str.strip())

        if "business_name" in chunk.columns:
            name_lengths.extend(
                chunk["business_name"].str.len().tolist()
            )

        if "business_address" in chunk.columns:
            address_lengths.extend(
                chunk["business_address"].str.len().tolist()
            )

    print(f"\nRows: {total:,}")

    print("\nMissing / empty values:")
    for col, count in missing.items():
        pct = 100 * count / total
        print(f"  {col:20s}: {count:12,} ({pct:7.3f}%)")

    print("\nCountries:")
    for country, count in countries.most_common():
        pct = 100 * count / total
        print(f"  {country:20s}: {count:12,} ({pct:7.3f}%)")

    if name_lengths:
        s = pd.Series(name_lengths)
        print("\nBusiness name length:")
        print(s.describe())

    if address_lengths:
        s = pd.Series(address_lengths)
        print("\nBusiness address length:")
        print(s.describe())

    return total


def analyze_ground_truth(path, chunksize=200_000):

    print("\n" + "=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    match_count_distribution = Counter()
    s2_count_distribution = Counter()
    s3_count_distribution = Counter()

    total_s1 = 0
    total_matches = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False
    ):

        for value in chunk["matched_entity_ids"]:

            value = value.strip()

            if not value:
                n = 0
                s2 = 0
                s3 = 0

            else:
                ids = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]

                n = len(ids)
                s2 = sum(x.startswith("S2-") for x in ids)
                s3 = sum(x.startswith("S3-") for x in ids)

            total_s1 += 1
            total_matches += n

            match_count_distribution[n] += 1
            s2_count_distribution[s2] += 1
            s3_count_distribution[s3] += 1

    print(f"\nS1 entities: {total_s1:,}")
    print(f"Total positive links: {total_matches:,}")

    print("\nMatches per S1:")
    for n, count in sorted(match_count_distribution.items()):
        pct = 100 * count / total_s1
        print(f"  {n:4d} matches: {count:12,} ({pct:7.3f}%)")

    print("\nS2 matches per S1:")
    for n, count in sorted(s2_count_distribution.items()):
        pct = 100 * count / total_s1
        print(f"  {n:4d}: {count:12,} ({pct:7.3f}%)")

    print("\nS3 matches per S1:")
    for n, count in sorted(s3_count_distribution.items()):
        pct = 100 * count / total_s1
        print(f"  {n:4d}: {count:12,} ({pct:7.3f}%)")


def main():

    print("\nBUSINESS ENTITY RESOLUTION DATASET ANALYSIS")

    basic_stats(
        FILES["train_s1"],
        "TRAIN SOURCE 1"
    )

    basic_stats(
        FILES["train_s2"],
        "TRAIN SOURCE 2"
    )

    basic_stats(
        FILES["train_s3"],
        "TRAIN SOURCE 3"
    )

    basic_stats(
        FILES["test_s1"],
        "TEST SOURCE 1"
    )

    basic_stats(
        FILES["test_s2"],
        "TEST SOURCE 2"
    )

    basic_stats(
        FILES["test_s3"],
        "TEST SOURCE 3"
    )

    analyze_ground_truth(
        FILES["ground_truth"]
    )


if __name__ == "__main__":
    main()