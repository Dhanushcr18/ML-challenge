"""End-to-end training, validation, inference, and submission checks."""
import logging
from pathlib import Path
from .duckdb_pipeline import run_disk_backed_pipeline

LOG=logging.getLogger(__name__)


def run_pipeline():
    return run_disk_backed_pipeline(Path(__file__).resolve().parents[1])
