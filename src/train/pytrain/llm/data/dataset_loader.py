"""
Multi-TDP dataset loader for LLM fine-tuning.

Reads data from ``/mnt/remote/<tdp_name>/`` mount points (pre-decrypted
by the encfs sidecar), supports Parquet / JSONL / JSON / CSV / Arrow,
and returns HuggingFace ``datasets.Dataset`` objects that all four
framework runners can consume.
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
from datasets import (
    Dataset,
    concatenate_datasets,
    load_dataset,
)

SUPPORTED_FORMATS = {"parquet", "jsonl", "json", "csv", "arrow"}

_LOAD_KWARGS = {
    "parquet": {"data_files": None},
    "jsonl": {"data_files": None},
    "json": {"data_files": None},
    "csv": {"data_files": None},
    "arrow": {"data_files": None},
}

_HF_FORMAT_TYPE = {
    "parquet": "parquet",
    "jsonl": "json",
    "json": "json",
    "csv": "csv",
    "arrow": "arrow",
}


def _resolve_data_files(path: str) -> list[str]:
    """Return a list of concrete file paths from a path that may be a
    file or a directory."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(str(f) for f in p.rglob("*") if f.is_file() and not f.name.startswith("."))
        if not files:
            raise FileNotFoundError(f"No data files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Data path does not exist: {path}")


def _apply_column_mapping(ds: Dataset, columns: dict | None) -> Dataset:
    """Rename columns according to the mapping declared in config."""
    if not columns:
        return ds
    for src_col, dst_col in columns.items():
        if src_col in ds.column_names and src_col != dst_col:
            ds = ds.rename_column(src_col, dst_col)
    return ds


def load_single_source(source: dict) -> Dataset:
    """Load a single dataset source into a HuggingFace Dataset.

    Parameters
    ----------
    source : dict
        A source entry from ``config.dataset.sources``.
        Must contain ``path``, and optionally ``format``, ``split``, ``columns``.
    """
    path = source["path"]
    fmt = source.get("format", "jsonl")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {SUPPORTED_FORMATS}")

    data_files = _resolve_data_files(path)
    hf_type = _HF_FORMAT_TYPE[fmt]

    ds = load_dataset(hf_type, data_files=data_files, split=source.get("split", "train"))

    ds = _apply_column_mapping(ds, source.get("columns"))

    return ds


def load_datasets(sources: list[dict]) -> list[Dataset]:
    """Load each source independently and return a list of Datasets."""
    return [load_single_source(src) for src in sources]


def load_and_merge_datasets(cfg: dict) -> Dataset:
    """Load all dataset sources, apply column mappings, concatenate
    horizontally (row-wise), and optionally cap sample count.

    Parameters
    ----------
    cfg : dict
        The full validated config dict (must contain ``dataset.sources``).

    Returns
    -------
    datasets.Dataset
    """
    ds_cfg = cfg["dataset"]
    datasets_list = load_datasets(ds_cfg["sources"])

    if len(datasets_list) == 1:
        merged = datasets_list[0]
    else:
        common_cols = set(datasets_list[0].column_names)
        for ds in datasets_list[1:]:
            common_cols &= set(ds.column_names)

        if not common_cols:
            raise ValueError(
                "No common columns across TDP datasets after column mapping. "
                "Ensure all sources map to the same schema."
            )

        aligned = []
        for ds in datasets_list:
            extra = set(ds.column_names) - common_cols
            if extra:
                ds = ds.remove_columns(list(extra))
            aligned.append(ds)

        merged = concatenate_datasets(aligned)

    max_samples = ds_cfg.get("max_samples")
    if max_samples and len(merged) > max_samples:
        merged = merged.select(range(max_samples))

    if ds_cfg.get("shuffle", True):
        merged = merged.shuffle(seed=ds_cfg.get("seed", 42))

    return merged
