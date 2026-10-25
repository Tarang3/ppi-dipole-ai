"""Sequence -> numeric feature vectors for the PPI model."""
from itertools import product

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Conjoint triad (Shen et al., PNAS 2007): 7 classes based on dipoles and side-chain volume.
TRIAD_CLASSES = {
    **dict.fromkeys("AGV", 0),
    **dict.fromkeys("ILFP", 1),
    **dict.fromkeys("YMTS", 2),
    **dict.fromkeys("HNQW", 3),
    **dict.fromkeys("RK", 4),
    **dict.fromkeys("DE", 5),
    "C": 6,
}
_TRIAD_INDEX = {t: i for i, t in enumerate(product(range(7), repeat=3))}

PROTEIN_FEATURE_NAMES = (
    [f"aac_{a}" for a in AMINO_ACIDS]
    + [f"ct_{a}{b}{c}" for a, b, c in _TRIAD_INDEX]
    + ["log_length", "gravy", "pI", "charge_pH7", "aromaticity", "instability"]
)


def clean_sequence(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    return "".join(c for c in seq if c in AMINO_ACIDS)


def amino_acid_composition(seq: str) -> np.ndarray:
    counts = np.array([seq.count(a) for a in AMINO_ACIDS], dtype=float)
    return counts / max(len(seq), 1)


def conjoint_triad(seq: str) -> np.ndarray:
    vec = np.zeros(len(_TRIAD_INDEX))
    classes = [TRIAD_CLASSES[c] for c in seq]
    for i in range(len(classes) - 2):
        vec[_TRIAD_INDEX[tuple(classes[i : i + 3])]] += 1
    if vec.max() > 0:
        vec = (vec - vec.min()) / (vec.max() - vec.min())
    return vec


def physicochemical(seq: str) -> np.ndarray:
    pa = ProteinAnalysis(seq)
    return np.array(
        [
            np.log(len(seq)),
            pa.gravy(),
            pa.isoelectric_point(),
            pa.charge_at_pH(7.0),
            pa.aromaticity(),
            pa.instability_index(),
        ]
    )


def protein_features(seq: str) -> np.ndarray:
    seq = clean_sequence(seq)
    if len(seq) < 3:
        raise ValueError("Sequence must contain at least 3 standard amino acids.")
    return np.concatenate([amino_acid_composition(seq), conjoint_triad(seq), physicochemical(seq)])


def pair_features(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Order-independent pair representation: f(A,B) == f(B,A)."""
    return np.concatenate([fa + fb, np.abs(fa - fb), fa * fb])
