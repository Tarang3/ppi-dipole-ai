"""Download sequences and structures from UniProt, AlphaFold DB and RCSB PDB."""
import io
import re

import requests
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
from Bio.PDB.Structure import Structure

TIMEOUT = 30
UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
PDB_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def is_uniprot_id(text: str) -> bool:
    return bool(UNIPROT_RE.match(text.strip().upper()))


def is_pdb_id(text: str) -> bool:
    return bool(PDB_RE.match(text.strip()))


def fetch_uniprot_sequence(accession: str) -> str:
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{accession.upper()}.fasta", timeout=TIMEOUT)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"No UniProt entry found for {accession}")
    return "".join(lines[1:])


def parse_structure(text: str, name: str = "protein", fmt: str = "pdb") -> Structure:
    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)
    return parser.get_structure(name, io.StringIO(text))


def fetch_alphafold_structure(accession: str) -> Structure:
    r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession.upper()}", timeout=TIMEOUT)
    if r.status_code == 404:
        raise ValueError(f"No AlphaFold model for {accession}")
    r.raise_for_status()
    entry = r.json()[0]
    pdb = requests.get(entry["pdbUrl"], timeout=TIMEOUT)
    pdb.raise_for_status()
    return parse_structure(pdb.text, accession)


def fetch_pdb_structure(pdb_id: str) -> Structure:
    pdb_id = pdb_id.strip().upper()
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=TIMEOUT)
    if r.ok:
        return parse_structure(r.text, pdb_id)
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.cif", timeout=TIMEOUT)
    r.raise_for_status()
    return parse_structure(r.text, pdb_id, fmt="cif")


def structure_sequence(structure: Structure, chain_id: str | None = None) -> str:
    model = next(iter(structure))
    builder = PPBuilder()
    seqs = []
    for chain in model:
        if chain_id and chain.id != chain_id:
            continue
        seqs.extend(str(pp.get_sequence()) for pp in builder.build_peptides(chain))
    return "".join(seqs)


def chain_ids(structure: Structure) -> list[str]:
    model = next(iter(structure))
    return [c.id for c in model if any(r.id[0] == " " for r in c)]
