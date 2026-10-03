"""Constants for T6SE feature extraction.

Sources
-------
* Amino-acid classes of the seven physicochemical properties: Table 1 of
  Wang et al., Bioinformatics 34(15), 2018, 2546-2555 (Bastion6).
* "pypredt6" groups: the physicochemical groups hard-coded in
  ``featureextraction()`` of the PyPredT6 project (pypredt6.py).
* QSO distance matrices (Schneider-Wrede, Grantham): taken from ``propy3``.
"""
from collections import OrderedDict

import numpy as np

# Alphabetical one-letter order; used for all column names and matrices.
AA = "ACDEFGHIKLMNPQRSTVWY"
N_AA = len(AA)
PAD = N_AA  # index used for padding in batched tensors

# --- Table 1 (Bastion6): 7 properties x 3 classes ---------------------------
CTD_PROPERTIES = OrderedDict(
    [
        ("Hydrophobicity", ("RKEDQN", "GASTPHY", "CLVIMFW")),
        ("NormalizedVDWV", ("GASTPDC", "NVEQIL", "MHKFRYW")),
        ("Polarity", ("LIFWCMVY", "PATGS", "HQRKNED")),
        ("Polarizability", ("GASDT", "CPNVEQIL", "KMHFRYW")),
        ("Charge", ("KR", "ANCQGHILMFPSTWYV", "DE")),
        ("SecondaryStr", ("EALMQKRH", "VIYCWFT", "GNPSD")),
        ("SolventAccessibility", ("ALFCGIVW", "RKQEND", "MPSTHY")),
    ]
)

# --- Extra groups from pypredt6.py (optional, not part of Bastion6) ---------
PYPREDT6_GROUPS = OrderedDict(
    [
        ("Charged", "DEKHR"),
        ("Aliphatic", "ILV"),
        ("Aromatic", "FHWY"),
        ("Polar", "DERKQN"),
        ("Neutral", "AGHPSTY"),
        ("Hydrophobic", "CFILMVW"),
        ("PositiveCharged", "KRH"),
        ("NegativeCharged", "DE"),
        ("Tiny", "ACDGST"),
        ("Small", "EHILKMNPQV"),
        ("Large", "FRWY"),
        ("Transmembrane", "ILVA"),
        ("Dipole_lt1", "AGVILFP"),
        ("Dipole_1_2", "YMTS"),
        ("Dipole_2_3", "HNQW"),
        ("Dipole_gt3", "RK"),
        ("Dipole_gt3_opposite", "DE"),
    ]
)

# Group names understood by compute_features()
CORE_GROUPS = ("AAC", "DPC", "QSO", "CTDC", "CTDT")
EXTRA_GROUPS = ("PYPREDT6_PHYSCHEM",)
ALL_GROUPS = CORE_GROUPS + EXTRA_GROUPS


def _check_tables():
    std = set(AA)
    for name, classes in CTD_PROPERTIES.items():
        letters = "".join(classes)
        if len(letters) != N_AA or set(letters) != std:
            raise AssertionError(f"CTD property {name} is not a partition of 20 aa")
    for name, letters in PYPREDT6_GROUPS.items():
        if not set(letters) <= std or len(set(letters)) != len(letters):
            raise AssertionError(f"bad pypredt6 group {name}")


_check_tables()


def ctd_class_map():
    """(7, 21) int array: class index 0..2 per amino acid, 3 for padding."""
    m = np.full((len(CTD_PROPERTIES), N_AA + 1), 3, dtype=np.int64)
    for p, classes in enumerate(CTD_PROPERTIES.values()):
        for c, letters in enumerate(classes):
            for a in letters:
                m[p, AA.index(a)] = c
    return m


def pypredt6_indicator():
    """(17, 20) 0/1 matrix: membership of each amino acid in each group."""
    m = np.zeros((len(PYPREDT6_GROUPS), N_AA))
    for g, letters in enumerate(PYPREDT6_GROUPS.values()):
        for a in letters:
            m[g, AA.index(a)] = 1.0
    return m


def load_qso_matrices():
    """Return (Schneider-Wrede, Grantham) distance matrices, shape (21, 21).

    Rows/columns follow ``AA`` order, the last row/column (padding) is zero.
    ``M[a, b]`` is the distance for the residue pair (a at i, b at i+d); the
    Schneider-Wrede matrix is *not* symmetric, so the direction matters and
    matches ``propy3`` / Chou's QSO definition.
    """
    try:
        from propy import QuasiSequenceOrder as Q
    except ImportError as exc:  # pragma: no cover
        raise ImportError("propy3 is required for QSO: pip install propy3") from exc

    def build(d):
        m = np.zeros((N_AA + 1, N_AA + 1))
        for i, a in enumerate(AA):
            for j, b in enumerate(AA):
                m[i, j] = d[a + b]
        return m

    return build(Q._Distance1), build(Q._Distance2)
