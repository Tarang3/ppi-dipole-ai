"""Turn user input (UniProt ID, PDB ID[:chain], raw sequence, or PDB file) into a protein record."""
from dataclasses import dataclass

from Bio.PDB.Structure import Structure

from . import fetch
from .features import clean_sequence


@dataclass
class Protein:
    label: str
    sequence: str
    structure: Structure | None
    source: str
    chain: str | None = None


def resolve_text(text: str) -> Protein:
    text = text.strip()
    if not text:
        raise ValueError("Empty input.")
    token = text.split(":")[0]
    if fetch.is_uniprot_id(token):
        acc = token.upper()
        seq = fetch.fetch_uniprot_sequence(acc)
        try:
            structure, src = fetch.fetch_alphafold_structure(acc), "UniProt sequence + AlphaFold structure"
        except Exception:
            structure, src = None, "UniProt sequence (no AlphaFold model found)"
        return Protein(acc, seq, structure, src)
    if fetch.is_pdb_id(token):
        structure = fetch.fetch_pdb_structure(token)
        return _from_structure(structure, token.upper(), text, "RCSB PDB")
    seq = clean_sequence(text)
    if len(seq) < 20:
        raise ValueError("Enter a UniProt ID (e.g. P69905), a PDB ID (e.g. 4HHB:A) or a protein sequence (20+ residues).")
    return Protein(f"Sequence ({len(seq)} aa)", seq, None, "Raw sequence (no structure, so no dipole)")


def resolve_pdb_file(content: str, filename: str, chain: str | None = None) -> Protein:
    fmt = "cif" if filename.lower().endswith((".cif", ".mmcif")) else "pdb"
    structure = fetch.parse_structure(content, filename, fmt)
    label = filename + (f":{chain}" if chain else "")
    return _from_structure(structure, filename, label, "Uploaded structure")


def _from_structure(structure: Structure, name: str, text: str, source: str) -> Protein:
    chains = fetch.chain_ids(structure)
    if not chains:
        raise ValueError(f"No protein chains found in {name}.")
    chain = text.split(":")[1].strip() if ":" in text else chains[0]
    if chain not in chains:
        raise ValueError(f"Chain {chain} not in {name}. Available: {', '.join(chains)}")
    seq = fetch.structure_sequence(structure, chain)
    return Protein(f"{name}:{chain}", seq, structure, f"{source}, chain {chain}", chain)
