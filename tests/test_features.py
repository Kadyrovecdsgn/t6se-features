"""Correctness tests.

1. Fast vectorised code == slow, line-by-line implementation of the formulas
   in the Bastion6 paper (exact, atol 1e-12).
2. Results agree with the ready-made propy3 functions (propy rounds its
   output to 3-6 decimals, hence the looser tolerance).
3. Padding / batching does not change any value.
"""
import math
from pathlib import Path

import numpy as np
import pytest
from propy import CTD as PCTD
from propy import QuasiSequenceOrder as PQSO

from t6se_features import compute_features, load_dataset
from t6se_features.constants import (
    AA,
    CTD_PROPERTIES,
    PYPREDT6_GROUPS,
    load_qso_matrices,
)
from t6se_features.core import encode_sequence, feature_names

DATA = Path(__file__).resolve().parents[1] / "data"
POS = DATA / "T6SE_Training_Pos_138.fasta"
NEG = DATA / "T6SE_Training_Neg_1112.fasta"

SEQS = [
    "MLAGIYLKVKGKTQGEIKGSVVQEGHDGKIHILAFKNDYDMPARLQEGLTPAAAARGTITLTKEMDRSSPQFLQALGKREMMEEFEITIHRPKTDTTGGDLTELLFTYKFEKVLITHMDQYSPTPHKDDSNGIKEGLLGYIEEIKFAYSGYSLEHAESGIAGAANWTNG",
    "MSYDYEKTSLTLYRAVFKANYDGDVGRYLHPDKELAEAAEVAPLLHPTFDSPNTPGVPARAPDIVAGRDGLYAPDTGGTSVFDRAGVLRRADGDFVIPDGTDIPPDLKVKQDSYNKRLQATHYTIMPAKPMYREVLMGQLDNFVRNAIRRQWEKARGL",
    "MSLYYVRYWNTIKNDGRMVLMGKSRVAIYTDGACSGNPGPGGWGAVLRFGDGGERRISGGSDDTTNNRMELTAVIMALAALSGPCSVCVNTDSTYVKNGITEWIRKWKLNGWRTSNKSAVKNVDLWVELERLTLLHSIEWRWVKAHAGNEYNEEADMLARGEVERRMVIPK",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # homopolymer: overlapping dipeptides
    "MKV" * 5 + "ACDEFGHIKLMNPQRSTVWY",  # shorter than maxlag + 1 after 35 -> still > 30
]


@pytest.fixture(scope="module")
def feats():
    return compute_features(SEQS, groups=["AAC", "DPC", "QSO", "CTDC", "CTDT", "PYPREDT6_PHYSCHEM"], device="cpu")


# --------------------------- naive reference --------------------------------
def naive_aac(s):
    return [s.count(a) / len(s) for a in AA]


def naive_dpc(s):
    n = len(s) - 1
    return [sum(1 for i in range(n) if s[i] == a and s[i + 1] == b) / n for a in AA for b in AA]


def naive_qso(s, maxlag=30, w=0.1):
    sw, gr = load_qso_matrices()
    idx = {a: i for i, a in enumerate(AA)}
    f = naive_aac(s)
    out = []
    for M in (sw, gr):
        tau = [sum(M[idx[s[i]], idx[s[i + d]]] ** 2 for i in range(len(s) - d)) for d in range(1, maxlag + 1)]
        den = 1 + w * sum(tau)
        out += [x / den for x in f] + [w * t / den for t in tau]
    return out


def naive_ctd(s):
    c_out, t_out = [], []
    for classes in CTD_PROPERTIES.values():
        cls = {a: k for k, letters in enumerate(classes) for a in letters}
        seq = [cls[a] for a in s]
        c_out += [seq.count(k) / len(seq) for k in range(3)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            n = sum(1 for i in range(len(seq) - 1) if {seq[i], seq[i + 1]} == {a, b})
            t_out.append(n / (len(seq) - 1))
    return c_out, t_out


def test_shapes(feats):
    assert feats["AAC"].shape == (len(SEQS), 20)
    assert feats["DPC"].shape == (len(SEQS), 400)
    assert feats["QSO"].shape == (len(SEQS), 100)
    assert feats["CTDC"].shape == (len(SEQS), 21)
    assert feats["CTDT"].shape == (len(SEQS), 21)
    assert feats["PYPREDT6_PHYSCHEM"].shape == (len(SEQS), len(PYPREDT6_GROUPS))
    for g, df in feats.items():
        assert list(df.columns) == feature_names(g)


@pytest.mark.parametrize("k", range(len(SEQS)))
def test_against_naive(feats, k):
    s = SEQS[k]
    np.testing.assert_allclose(feats["AAC"].iloc[k], naive_aac(s), atol=1e-12)
    np.testing.assert_allclose(feats["DPC"].iloc[k], naive_dpc(s), atol=1e-12)
    np.testing.assert_allclose(feats["QSO"].iloc[k], naive_qso(s), atol=1e-12)
    c, t = naive_ctd(s)
    np.testing.assert_allclose(feats["CTDC"].iloc[k], c, atol=1e-12)
    np.testing.assert_allclose(feats["CTDT"].iloc[k], t, atol=1e-12)


def test_distributions_sum_to_one(feats):
    np.testing.assert_allclose(feats["AAC"].sum(axis=1), 1.0)
    np.testing.assert_allclose(feats["DPC"].sum(axis=1), 1.0)
    c = feats["CTDC"].values.reshape(len(SEQS), 7, 3)
    np.testing.assert_allclose(c.sum(axis=2), 1.0)
    t = feats["CTDT"].values.reshape(len(SEQS), 7, 3)
    assert (t.sum(axis=2) <= 1.0 + 1e-12).all()


# ---------------------------- vs propy3 --------------------------------------
@pytest.mark.parametrize("k", range(len(SEQS)))
def test_qso_vs_propy(feats, k):
    s = SEQS[k]
    ref = PQSO.GetQuasiSequenceOrder(s, maxlag=30, weight=0.1)
    # propy order: SW Xr (propy AALetter order), SW Xd, Grantham Xr, Grantham Xd
    ref_vals = list(ref.values())
    assert len(ref_vals) == 100
    mine = feats["QSO"].iloc[k]
    perm = [AA.index(a) for a in PQSO.AALetter]  # propy position -> my AA position
    expected = np.empty(100)
    for block, (xr0, xd0) in enumerate(((0, 20), (50, 70))):
        expected[xr0 : xr0 + 20] = [mine.iloc[xr0 + p] for p in perm]
        expected[xd0 : xd0 + 30] = mine.iloc[xd0 : xd0 + 30].values
    np.testing.assert_allclose(expected, ref_vals, atol=5e-5)  # propy rounds; wrong aa order gives ~3e-4


PROP_FUNCS = {
    "Hydrophobicity": "Hydrophobicity", "NormalizedVDWV": "NormalizedVDWV", "Polarity": "Polarity",
    "Polarizability": "Polarizability", "Charge": "Charge", "SecondaryStr": "SecondaryStr",
    "SolventAccessibility": "SolventAccessibility",
}


@pytest.mark.parametrize("k", range(len(SEQS)))
def test_ctd_vs_propy(feats, k):
    s = SEQS[k]
    for prop in CTD_PROPERTIES:
        comp = getattr(PCTD, f"CalculateComposition{prop}")(s)
        trans = getattr(PCTD, f"CalculateTransition{prop}")(s)
        np.testing.assert_allclose(
            [feats["CTDC"].loc[:, f"CTDC_{prop}_C{c}"].iloc[k] for c in (1, 2, 3)], list(comp.values()), atol=6e-4
        )
        np.testing.assert_allclose(
            [feats["CTDT"].loc[:, f"CTDT_{prop}_T{t}"].iloc[k] for t in ("12", "13", "23")],
            list(trans.values()),
            atol=6e-4,
        )


def test_pypredt6_groups_hand_checked():
    f = compute_features(["ACDEKR"], groups=["PYPREDT6_PHYSCHEM"], device="cpu")["PYPREDT6_PHYSCHEM"].iloc[0]
    assert f["PHYSCHEM_Charged"] == pytest.approx(4 / 6)  # D E K R
    assert f["PHYSCHEM_Tiny"] == pytest.approx(3 / 6)  # A C D
    assert f["PHYSCHEM_NegativeCharged"] == pytest.approx(2 / 6)  # D E


# -------------------------- padding / batching -------------------------------
def test_batching_invariance():
    """Same values whether sequences are alone, in one batch, or split in many."""
    groups = ["AAC", "DPC", "QSO", "CTDC", "CTDT", "PYPREDT6_PHYSCHEM"]
    together = compute_features(SEQS, groups=groups, device="cpu", max_tokens=10**9)
    split = compute_features(SEQS, groups=groups, device="cpu", max_tokens=1)  # 1 sequence per batch
    for g in groups:
        np.testing.assert_allclose(together[g].values, split[g].values, atol=1e-12)


def test_input_order_preserved():
    a = compute_features(SEQS, groups=["AAC"], ids=list("abcde"), device="cpu")["AAC"]
    b = compute_features(SEQS[::-1], groups=["AAC"], ids=list("edcba"), device="cpu")["AAC"]
    np.testing.assert_allclose(a.loc[list("edcba")].values, b.values, atol=1e-12)


def test_nonstandard_residues():
    codes, removed = encode_sequence("acdXBz*K")
    assert removed == 4 and len(codes) == 4
    with pytest.raises(ValueError):
        encode_sequence("ACDX", nonstandard="error")
    with pytest.warns(UserWarning):
        r = compute_features(["ACDXK"], groups=["AAC"], device="cpu")["AAC"]
    np.testing.assert_allclose(r.iloc[0].sum(), 1.0)


def test_short_sequence_qso_zero_lags():
    with pytest.warns(UserWarning):
        r = compute_features(["ACDEFGHIK"], groups=["QSO"], device="cpu")["QSO"].iloc[0]
    assert (r[[c for c in r.index if c.startswith("QSO_SW_Xd_") and int(c.split("_")[-1]) >= 9]] == 0).all()


# ------------------------------ real data ------------------------------------
def test_real_dataset_sanity():
    df = load_dataset(POS, NEG)
    assert len(df) == 1250 and df.label.sum() == 138
    f = compute_features(df.seq.tolist(), ids=df.id.tolist(), device="cpu")
    for g, t in f.items():
        assert np.isfinite(t.values).all(), g
        assert (t.values >= 0).all(), g
        assert t.index.is_unique


def test_gpu_equals_cpu_if_available():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no CUDA device")
    groups = ["AAC", "DPC", "QSO", "CTDC", "CTDT", "PYPREDT6_PHYSCHEM"]
    c = compute_features(SEQS, groups=groups, device="cpu")
    g = compute_features(SEQS, groups=groups, device="gpu")
    for k in groups:
        np.testing.assert_allclose(c[k].values, g[k].values, atol=1e-10)
