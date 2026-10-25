import io

import numpy as np
import pytest
from Bio.PDB import PDBParser

from ppi_ai.dipole import E_ANGSTROM_TO_DEBYE, center_of_mass, charge_sites, compute_dipole, fractional_charge
from ppi_ai.features import clean_sequence, pair_features, protein_features
from ppi_ai.predict import MODEL_DIR

HBA = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
HBB = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"


def _atom(serial, name, resname, resseq, x, y, z, element):
    return (f"ATOM  {serial:5d} {name:<4} {resname} A{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n")


def test_feature_shapes_and_symmetry():
    fa, fb = protein_features(HBA), protein_features(HBB)
    assert fa.shape == fb.shape == (369,)
    np.testing.assert_allclose(pair_features(fa, fb), pair_features(fb, fa))


def test_clean_sequence():
    assert clean_sequence("mk v\nXZ*a") == "MKVA"


def test_fractional_charge():
    assert fractional_charge(10.54, 7.0, +1) == pytest.approx(1.0, abs=1e-3)
    assert fractional_charge(3.9, 7.0, -1) == pytest.approx(-1.0, abs=1e-3)
    assert fractional_charge(6.0, 6.0, +1) == pytest.approx(0.5)


def test_dipole_two_charges():
    # Lys at +x, Asp at -x, 10 A apart, at pH 7 -> about +1/-1 charges
    pdb = (
        _atom(1, "N", "LYS", 1, 4.0, 0, 0, "N") + _atom(2, "CA", "LYS", 1, 4.5, 0, 0, "C")
        + _atom(3, "NZ", "LYS", 1, 5.0, 0, 0, "N")
        + _atom(4, "CA", "ASP", 2, -4.5, 0, 0, "C") + _atom(5, "OD1", "ASP", 2, -5.0, 0.5, 0, "O")
        + _atom(6, "OD2", "ASP", 2, -5.0, -0.5, 0, "O")
    )
    s = PDBParser(QUIET=True).get_structure("t", io.StringIO(pdb))
    res = compute_dipole(s, ph=7.0)
    sites = charge_sites(s, 7.0)
    com = center_of_mass(s)
    expected = sum(q * (pos - com) for pos, q, _ in sites) * E_ANGSTROM_TO_DEBYE
    np.testing.assert_allclose(res.vector_debye, expected, rtol=1e-6)
    assert {label for _, _, label in sites} == {"LYS1A", "ASP2A", "NtermA"}
    # Positive charges (Lys, N-terminus) on +x, Asp on -x -> dipole points along +x.
    assert res.vector_debye[0] > 0 and abs(res.vector_debye[1]) < 1e-3
    assert res.net_charge == pytest.approx(0.9, abs=0.05)


@pytest.mark.skipif(not (MODEL_DIR / "ppi_model.joblib").exists(), reason="model not trained")
def test_predict_range():
    from ppi_ai.predict import predict_interaction
    p = predict_interaction(HBA, HBB)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(predict_interaction(HBB, HBA))
