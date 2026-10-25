"""Build a balanced human PPI dataset from STRING v12 physical interactions.

Positives: STRING physical links with combined_score >= MIN_SCORE.
Negatives: "degree-preserving" random pairs - endpoints of the positive pairs
are shuffled and re-paired, so every protein appears as often in negatives as
in positives. This stops the model from just learning "hub proteins interact".
Any shuffled pair that appears anywhere in STRING physical links is discarded.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import requests

STRING_VERSION = "v12.0"
BASE = "https://stringdb-downloads.org/download"
LINKS_URL = f"{BASE}/protein.physical.links.{STRING_VERSION}/9606.protein.physical.links.{STRING_VERSION}.txt.gz"
SEQS_URL = f"{BASE}/protein.sequences.{STRING_VERSION}/9606.protein.sequences.{STRING_VERSION}.fa.gz"

MIN_SCORE = 900
MIN_LEN, MAX_LEN = 50, 2000


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def read_fasta_gz(path: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def build_dataset(raw_dir: Path, n_pairs: int = 20000, seed: int = 42) -> tuple[pd.DataFrame, dict[str, str]]:
    links_path = download(LINKS_URL, raw_dir / Path(LINKS_URL).name)
    seqs_path = download(SEQS_URL, raw_dir / Path(SEQS_URL).name)

    seqs = {k: v for k, v in read_fasta_gz(seqs_path).items() if MIN_LEN <= len(v) <= MAX_LEN}
    links = pd.read_csv(links_path, sep=" ")
    a, b = links["protein1"].to_numpy(), links["protein2"].to_numpy()
    all_known = set(zip(np.minimum(a, b), np.maximum(a, b)))

    pos = links[links["combined_score"] >= MIN_SCORE]
    pos = pos[pos["protein1"].isin(seqs) & pos["protein2"].isin(seqs)]
    pa, pb = pos["protein1"].to_numpy(), pos["protein2"].to_numpy()
    pos_pairs = sorted(set(zip(np.minimum(pa, pb), np.maximum(pa, pb))))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pos_pairs), size=min(n_pairs, len(pos_pairs)), replace=False)
    pos_pairs = [pos_pairs[i] for i in idx]

    endpoints = np.array([p for pair in pos_pairs for p in pair])
    neg_pairs: set[tuple[str, str]] = set()
    while len(neg_pairs) < len(pos_pairs):
        shuffled = rng.permutation(endpoints)
        for x, y in zip(shuffled[0::2], shuffled[1::2]):
            pair = (min(x, y), max(x, y))
            if x != y and pair not in all_known:
                neg_pairs.add(pair)
                if len(neg_pairs) == len(pos_pairs):
                    break

    df = pd.DataFrame(
        [(x, y, 1) for x, y in pos_pairs] + [(x, y, 0) for x, y in neg_pairs],
        columns=["protein_a", "protein_b", "label"],
    )
    used = set(df["protein_a"]) | set(df["protein_b"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), {k: seqs[k] for k in used}
