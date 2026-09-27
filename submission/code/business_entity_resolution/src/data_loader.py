"""TSV reading, structural inspection, and validation."""
import logging
from pathlib import Path
import pandas as pd

LOG = logging.getLogger(__name__)
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str | Path, *, chunksize: int | None = None):
    """Always read source files as TSV; chunksize returns a pandas reader."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def inspect_files(paths: list[Path], chunksize: int = 250_000) -> dict:
    stats = {}
    for path in paths:
        rows, missing, columns = 0, {}, []
        for chunk in load_source(path, chunksize=chunksize):
            rows += len(chunk); columns = list(chunk.columns)
            for col in columns:
                missing[col] = missing.get(col, 0) + int(chunk[col].eq("").sum())
        stats[str(path)] = {"rows": rows, "columns": columns, "missing": missing}
        LOG.info("Inspected %s: %s rows, columns=%s, missing=%s", path, rows, columns, missing)
    return stats


def validate_source(df: pd.DataFrame, prefix: str) -> None:
    absent = set(SOURCE_COLUMNS) - set(df.columns)
    if absent:
        raise ValueError(f"Missing required columns: {sorted(absent)}")
    if df["entity_id"].duplicated().any():
        raise ValueError("entity_id values must be unique within a source")
    bad = ~df["entity_id"].str.startswith(prefix)
    if bad.any():
        raise ValueError(f"Expected all entity_id values to start with {prefix}")


def read_all_sources(directory: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Project sizes are large; string dtypes avoid expensive mixed-type inference.
    frames = tuple(load_source(directory / f"{stem}_source{i}.tsv") for i in (1, 2, 3))
    for i, frame in enumerate(frames, 1):
        validate_source(frame, f"S{i}-")
    return frames
