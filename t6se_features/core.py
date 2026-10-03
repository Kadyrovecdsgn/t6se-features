"""Vectorised calculation of Bastion6 sequence-based and physicochemical features.

Features (names follow Wang et al., Bioinformatics 2018, Section 2.2):

* AAC   - amino acid composition, 20 dims
* DPC   - dipeptide composition, 400 dims
* QSO   - quasi-sequence-order, Schneider-Wrede + Grantham, 2 x (20 + maxlag) = 100 dims
* CTDC  - composition of the 7 physicochemical properties of Table 1, 7 x 3 = 21 dims
* CTDT  - transition of the 7 properties, 7 x 3 = 21 dims
* PYPREDT6_PHYSCHEM - (optional) 17 residue-group fractions from pypredt6.py

All numeric work is written against an array module ``xp`` so the very same
code runs on the CPU (NumPy) or on an NVIDIA GPU (CuPy, e.g. in Google Colab).
Sequences are sorted by length, split into padded batches and processed as
tensors; padding never contributes to any feature (checked in the tests).
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    AA,
    ALL_GROUPS,
    CORE_GROUPS,
    CTD_PROPERTIES,
    N_AA,
    PAD,
    PYPREDT6_GROUPS,
    ctd_class_map,
    load_qso_matrices,
    pypredt6_indicator,
)

# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------


def get_backend(device: str = "auto"):
    """Return ``(xp, name)`` where xp is numpy or cupy.

    device: ``"cpu"``, ``"gpu"`` (error if CuPy / CUDA are unavailable) or
    ``"auto"`` (GPU when CuPy and a CUDA device exist, otherwise CPU).
    """
    device = device.lower()
    if device not in {"auto", "cpu", "gpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu' or 'gpu'")
    if device == "cpu":
        return np, "cpu"
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp, "gpu"
        raise RuntimeError("no CUDA device found")
    except Exception as exc:  # ImportError, CUDA runtime errors
        if device in {"gpu", "cuda"}:
            raise RuntimeError(f"GPU backend requested but unavailable: {exc}") from exc
        return np, "cpu"


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# --------------------------------------------------------------------------
# sequence encoding
# --------------------------------------------------------------------------

_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _a in enumerate(AA):
    _LUT[ord(_a)] = _i
    _LUT[ord(_a.lower())] = _i


def encode_sequence(seq: str, nonstandard: str = "drop") -> Tuple[np.ndarray, int]:
    """Encode a protein string as uint8 codes 0..19 (order of ``AA``).

    Non-standard symbols (X, B, Z, U, '*', '-', ...) are removed when
    ``nonstandard="drop"`` or raise ``ValueError`` when ``"error"``.
    Returns ``(codes, number_of_removed_symbols)``.
    """
    raw = np.frombuffer(seq.strip().encode("ascii", "replace"), dtype=np.uint8)
    codes = _LUT[raw]
    ok = codes < N_AA
    removed = int((~ok).sum())
    if removed and nonstandard == "error":
        raise ValueError(f"sequence contains {removed} non-standard symbol(s)")
    codes = codes[ok]
    if codes.size == 0:
        raise ValueError("empty sequence after removing non-standard symbols")
    return codes, removed


# --------------------------------------------------------------------------
# column names
# --------------------------------------------------------------------------


def feature_names(group: str, maxlag: int = 30) -> List[str]:
    if group == "AAC":
        return [f"AAC_{a}" for a in AA]
    if group == "DPC":
        return [f"DPC_{a}{b}" for a in AA for b in AA]
    if group == "QSO":
        names = []
        for m in ("SW", "Grantham"):
            names += [f"QSO_{m}_Xr_{a}" for a in AA]
            names += [f"QSO_{m}_Xd_{d}" for d in range(1, maxlag + 1)]
        return names
    if group == "CTDC":
        return [f"CTDC_{p}_C{c}" for p in CTD_PROPERTIES for c in (1, 2, 3)]
    if group == "CTDT":
        return [f"CTDT_{p}_T{t}" for p in CTD_PROPERTIES for t in ("12", "13", "23")]
    if group == "PYPREDT6_PHYSCHEM":
        return [f"PHYSCHEM_{g}" for g in PYPREDT6_GROUPS]
    raise ValueError(f"unknown feature group {group!r}; choose from {ALL_GROUPS}")


# --------------------------------------------------------------------------
# batch kernels
# --------------------------------------------------------------------------


def _onehot(X, n_classes, xp):
    return (X[..., None] == xp.arange(n_classes)).astype(xp.float64)


def _pair_counts(oh, xp):
    """Counts of ordered neighbour pairs: (B, k, k)."""
    return xp.einsum("bli,blj->bij", oh[:, :-1], oh[:, 1:])


def _qso(X, f, mats, maxlag, weight, xp):
    """Quasi-sequence-order descriptors, (B, 2*(20+maxlag)).

    tau_d = sum_{i=1}^{N-d} dist(i, i+d)^2
    Xr = f_r / (1 + w*sum tau),   Xd = w*tau_d / (1 + w*sum tau)
    (sum f_r = 1 because f is the normalised occurrence).
    """
    blocks = []
    for M in mats:
        taus = []
        for d in range(1, maxlag + 1):
            # padding index PAD points to the all-zero row/column of M
            dist = M[X[:, :-d], X[:, d:]]
            taus.append((dist * dist).sum(axis=1))
        tau = xp.stack(taus, axis=1)
        denom = 1.0 + weight * tau.sum(axis=1, keepdims=True)
        blocks += [f / denom, weight * tau / denom]
    return xp.concatenate(blocks, axis=1)


def _ctd(X, n, cls_map, xp):
    """CTDC (B,21) and CTDT (B,21)."""
    comp, trans = [], []
    denom = xp.maximum(n - 1.0, 1.0)
    for p in range(cls_map.shape[0]):
        C = cls_map[p][X]  # (B, L) class index, 3 = padding
        oh = _onehot(C, 3, xp)
        comp.append(oh.sum(axis=1) / n)
        M = _pair_counts(oh, xp)
        t12 = M[:, 0, 1] + M[:, 1, 0]
        t13 = M[:, 0, 2] + M[:, 2, 0]
        t23 = M[:, 1, 2] + M[:, 2, 1]
        trans.append(xp.stack([t12, t13, t23], axis=1) / denom)  # (B,3)/(B,1)
    return xp.concatenate(comp, axis=1), xp.concatenate(trans, axis=1)


def _make_batches(lengths: np.ndarray, max_tokens: int) -> List[np.ndarray]:
    order = np.argsort(lengths, kind="stable")
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * int(lengths[i]) > max_tokens:
            batches.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        batches.append(np.array(cur))
    return batches


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def compute_features(
    sequences: Sequence[str],
    ids: Optional[Sequence[str]] = None,
    groups: Iterable[str] = CORE_GROUPS,
    device: str = "auto",
    maxlag: int = 30,
    weight: float = 0.1,
    max_tokens: int = 400_000,
    nonstandard: str = "drop",
) -> Dict[str, pd.DataFrame]:
    """Compute feature groups for protein sequences.

    Parameters
    ----------
    sequences : protein strings (one-letter code).
    ids       : row labels (default: 0..N-1).
    groups    : subset of ``ALL_GROUPS``.
    device    : "auto" | "cpu" | "gpu".
    maxlag, weight : QSO parameters (Bastion6 defaults: 30 and 0.1).
    max_tokens: upper bound for ``batch_size * longest_sequence`` (memory knob).
    nonstandard: "drop" (default) or "error" for symbols outside the 20 aa.

    Returns
    -------
    dict ``group -> DataFrame`` (rows in the input order).
    """
    groups = tuple(groups)
    for g in groups:
        if g not in ALL_GROUPS:
            raise ValueError(f"unknown feature group {g!r}; choose from {ALL_GROUPS}")
    if maxlag < 1:
        raise ValueError("maxlag must be >= 1")
    n_seq = len(sequences)
    ids = list(range(n_seq)) if ids is None else list(ids)
    if len(ids) != n_seq:
        raise ValueError("ids and sequences must have the same length")

    xp, backend = get_backend(device)

    encoded = []
    n_removed = 0
    for s in sequences:
        codes, rem = encode_sequence(s, nonstandard)
        encoded.append(codes)
        n_removed += rem
    if n_removed:
        warnings.warn(f"{n_removed} non-standard residue(s) were removed", stacklevel=2)
    lengths = np.array([len(c) for c in encoded])
    if "QSO" in groups and (lengths <= maxlag).any():
        warnings.warn(
            f"{int((lengths <= maxlag).sum())} sequence(s) are not longer than maxlag={maxlag}; "
            "QSO lags beyond the sequence length are set to 0",
            stacklevel=2,
        )

    qso_mats = [xp.asarray(m) for m in load_qso_matrices()] if "QSO" in groups else None
    cls_map = xp.asarray(ctd_class_map()) if ("CTDC" in groups or "CTDT" in groups) else None
    indicator = xp.asarray(pypredt6_indicator()) if "PYPREDT6_PHYSCHEM" in groups else None

    out = {g: np.zeros((n_seq, len(feature_names(g, maxlag)))) for g in groups}

    for idx in _make_batches(lengths, max_tokens):
        L = int(lengths[idx].max())
        Xnp = np.full((len(idx), L), PAD, dtype=np.int64)
        for r, i in enumerate(idx):
            Xnp[r, : lengths[i]] = encoded[i]
        X = xp.asarray(Xnp)
        n = xp.asarray(lengths[idx].astype(np.float64))[:, None]  # (B,1)

        oh = _onehot(X, N_AA, xp)  # padding rows are all-zero
        counts = oh.sum(axis=1)
        aac = counts / n

        res = {}
        if "AAC" in groups:
            res["AAC"] = aac
        if "DPC" in groups:
            pair = _pair_counts(oh, xp).reshape(len(idx), N_AA * N_AA)
            res["DPC"] = pair / xp.maximum(n - 1.0, 1.0)
        if "QSO" in groups:
            res["QSO"] = _qso(X, aac, qso_mats, maxlag, weight, xp)
        if "CTDC" in groups or "CTDT" in groups:
            c, t = _ctd(X, n, cls_map, xp)
            if "CTDC" in groups:
                res["CTDC"] = c
            if "CTDT" in groups:
                res["CTDT"] = t
        if "PYPREDT6_PHYSCHEM" in groups:
            res["PYPREDT6_PHYSCHEM"] = aac @ indicator.T

        for g, arr in res.items():
            out[g][idx] = _to_numpy(arr)

    return {
        g: pd.DataFrame(out[g], index=pd.Index(ids, name="id"), columns=feature_names(g, maxlag))
        for g in groups
    }
