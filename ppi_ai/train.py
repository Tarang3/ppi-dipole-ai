"""Train the PPI classifier.

    python -m ppi_ai.train

Downloads STRING data into data/raw, trains a gradient-boosted tree model and
writes models/ppi_model.joblib + models/metrics.json.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .dataset import MIN_SCORE, STRING_VERSION, build_dataset
from .features import pair_features, protein_features

ROOT = Path(__file__).resolve().parent.parent


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0, random_state=seed
    )


def evaluate(model, X, y) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=20000, help="number of positive pairs (same number of negatives)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    df, seqs = build_dataset(ROOT / "data" / "raw", args.n_pairs, args.seed)
    print(f"Dataset: {len(df)} pairs, {len(seqs)} proteins ({time.time() - t0:.0f}s)")

    feats = {pid: protein_features(s) for pid, s in seqs.items()}
    X = np.stack([pair_features(feats[a], feats[b]) for a, b in zip(df["protein_a"], df["protein_b"])])
    y = df["label"].to_numpy()
    print(f"Features: {X.shape} ({time.time() - t0:.0f}s)")

    # 1) Standard random split of pairs.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    model = make_model(args.seed).fit(X_tr, y_tr)
    random_split = evaluate(model, X_te, y_te)
    print("Random pair split:", random_split)

    # 2) Harder split: test pairs where BOTH proteins were never seen during training.
    rng = np.random.default_rng(args.seed)
    proteins = np.array(sorted(seqs))
    held_out = set(rng.choice(proteins, size=int(0.3 * len(proteins)), replace=False))
    in_a, in_b = df["protein_a"].isin(held_out).to_numpy(), df["protein_b"].isin(held_out).to_numpy()
    train_mask, test_mask = ~in_a & ~in_b, in_a & in_b
    cold_model = make_model(args.seed).fit(X[train_mask], y[train_mask])
    cold_split = evaluate(cold_model, X[test_mask], y[test_mask])
    print("Unseen-protein split:", cold_split)

    # Final model is trained on everything.
    final = make_model(args.seed).fit(X, y)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(final, out / "ppi_model.joblib", compress=3)
    metrics = {
        "data": f"STRING {STRING_VERSION} human physical links, score >= {MIN_SCORE}",
        "n_pairs": int(len(df)),
        "n_proteins": len(seqs),
        "n_features": int(X.shape[1]),
        "model": "HistGradientBoostingClassifier",
        "random_pair_split": random_split,
        "unseen_protein_split": cold_split,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
