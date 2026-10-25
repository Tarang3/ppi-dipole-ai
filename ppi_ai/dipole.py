"""Protein dipole moment from 3D structure.

Each ionizable group gets a fractional charge from the Henderson-Hasselbalch
equation at the chosen pH. The dipole is computed about the center of mass:

    mu = sum_i q_i * (r_i - r_cm)

1 e*Angstrom = 4.803 Debye.
"""
from dataclasses import dataclass

import numpy as np
from Bio.PDB.Structure import Structure

E_ANGSTROM_TO_DEBYE = 4.80320

# residue -> (atom names used as charge position, pKa, sign)
IONIZABLE = {
    "ASP": (("OD1", "OD2"), 3.9, -1),
    "GLU": (("OE1", "OE2"), 4.07, -1),
    "HIS": (("ND1", "NE2"), 6.04, +1),
    "LYS": (("NZ",), 10.54, +1),
    "ARG": (("NH1", "NH2", "NE"), 12.48, +1),
    "CYS": (("SG",), 8.18, -1),
    "TYR": (("OH",), 10.46, -1),
}
N_TERM_PKA = 8.0
C_TERM_PKA = 3.1

ATOMIC_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "H": 1.008, "P": 30.974, "SE": 78.97}


def fractional_charge(pka: float, ph: float, sign: int) -> float:
    if sign > 0:
        return 1.0 / (1.0 + 10 ** (ph - pka))
    return -1.0 / (1.0 + 10 ** (pka - ph))


@dataclass
class DipoleResult:
    magnitude_debye: float
    vector_debye: np.ndarray
    net_charge: float
    center_of_mass: np.ndarray
    n_charged_sites: int
    n_residues: int

    @property
    def debye_per_residue(self) -> float:
        return self.magnitude_debye / max(self.n_residues, 1)


def _protein_residues(chain):
    return [r for r in chain if r.id[0] == " " and "CA" in r]


def charge_sites(structure: Structure, ph: float = 7.0, chains=None):
    """Return list of (position, charge, label)."""
    model = next(iter(structure))
    sites = []
    for chain in model:
        if chains and chain.id not in chains:
            continue
        residues = _protein_residues(chain)
        if not residues:
            continue
        for res in residues:
            spec = IONIZABLE.get(res.get_resname())
            if spec:
                names, pka, sign = spec
                atoms = [res[n] for n in names if n in res]
                if atoms:
                    pos = np.mean([a.coord for a in atoms], axis=0)
                    sites.append((pos, fractional_charge(pka, ph, sign), f"{res.get_resname()}{res.id[1]}{chain.id}"))
        first, last = residues[0], residues[-1]
        if "N" in first:
            sites.append((first["N"].coord, fractional_charge(N_TERM_PKA, ph, +1), f"Nterm{chain.id}"))
        c_atoms = [last[n] for n in ("O", "OXT") if n in last]
        if c_atoms:
            pos = np.mean([a.coord for a in c_atoms], axis=0)
            sites.append((pos, fractional_charge(C_TERM_PKA, ph, -1), f"Cterm{chain.id}"))
    return sites


def center_of_mass(structure: Structure, chains=None) -> np.ndarray:
    model = next(iter(structure))
    coords, masses = [], []
    for chain in model:
        if chains and chain.id not in chains:
            continue
        for res in _protein_residues(chain):
            for atom in res:
                coords.append(atom.coord)
                masses.append(ATOMIC_MASS.get(atom.element.upper(), 12.0))
    if not coords:
        raise ValueError("No protein atoms found in structure.")
    coords, masses = np.array(coords), np.array(masses)
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()


def compute_dipole(structure: Structure, ph: float = 7.0, chains=None) -> DipoleResult:
    com = center_of_mass(structure, chains)
    sites = charge_sites(structure, ph, chains)
    model = next(iter(structure))
    n_res = sum(len(_protein_residues(c)) for c in model if not chains or c.id in chains)
    if not sites:
        return DipoleResult(0.0, np.zeros(3), 0.0, com, 0, n_res)
    pos = np.array([s[0] for s in sites])
    q = np.array([s[1] for s in sites])
    mu = (q[:, None] * (pos - com)).sum(axis=0) * E_ANGSTROM_TO_DEBYE
    return DipoleResult(float(np.linalg.norm(mu)), mu, float(q.sum()), com, len(sites), n_res)
