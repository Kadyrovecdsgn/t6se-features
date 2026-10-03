"""FASTA reading (Biopython SeqIO) and table writing."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO


def short_id(record) -> str:
    """'gi|77358963|ref|YP_338391.1|hypothetical ...' -> 'gi|77358963'.

    The gi number is unique in the Bastion6 training files, whereas the
    accession (e.g. 22607806) is repeated for several positive records.
    Records without a gi header keep Biopython's ``record.id``.
    """
    parts = record.description.split("|")
    if len(parts) > 1 and parts[0] == "gi":
        return f"gi|{parts[1]}"
    return record.id


def read_fasta(path, label: Optional[int] = None, class_name: Optional[str] = None) -> pd.DataFrame:
    """Read a FASTA file into a DataFrame [id, label, class, seq]."""
    rows = [(short_id(r), str(r.seq)) for r in SeqIO.parse(str(path), "fasta")]
    df = pd.DataFrame(rows, columns=["id", "seq"])
    if df["id"].duplicated().any():
        raise ValueError(f"duplicate ids in {path}")
    df.insert(1, "label", label)
    df.insert(2, "class", class_name)
    return df


def load_dataset(pos_fasta, neg_fasta) -> pd.DataFrame:
    """Positive (T6SE, label=1) + negative (non-T6SE, label=0) sets."""
    pos = read_fasta(pos_fasta, 1, "T6SE")
    neg = read_fasta(neg_fasta, 0, "non-T6SE")
    df = pd.concat([pos, neg], ignore_index=True)
    if df["id"].duplicated().any():
        raise ValueError("ids overlap between the positive and negative files")
    return df


def write_table(df: pd.DataFrame, path, fmt: str = "csv") -> Path:
    path = Path(path).with_suffix(".tsv" if fmt == "tsv" else ".csv")
    df.to_csv(path, sep="\t" if fmt == "tsv" else ",", float_format="%.8g")
    return path
