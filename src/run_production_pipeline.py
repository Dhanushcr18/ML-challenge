"""
Production Inference Script: Enhanced 5A + 4A + 1B Pipeline @ Threshold 0.97.

Executes the full production inference for 1,732,544 test S1 entities against 9.97M target entities.
Preserves all safety safeguards, checkpoint/resume capabilities, and executes official validation.
"""

import sys
import os
import time
import logging
from pathlib import Path
import duckdb
import joblib

REPO_ROOT = Path(r"c:\ML Challange")
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "output" / "production_inference.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_PATH), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
LOG = logging.getLogger("production_inference")

from src.duckdb_pipeline import (
    _candidate_table,
    _score_and_export,
    _set_temp,
    PRODUCTION_THRESHOLD,
    MEMORY_LIMIT
)
from src.config import MODEL_DIR, TEST_DIR, OUTPUT_DIR

DB_PATH = OUTPUT_DIR / "pipeline.duckdb"

def main():
    LOG.info("="*80)
    LOG.info("STARTING FULL PRODUCTION INFERENCE: 5A + 4A + 1B @ THRESHOLD %.2f", PRODUCTION_THRESHOLD)
    LOG.info("="*80)

    # Verify backup exists
    backup_dir = OUTPUT_DIR / "backup_pre_enhancement"
    if not (backup_dir / "matching_results.tsv").exists() or not (backup_dir / "candidate_pairs.tsv").exists():
        LOG.error("Safety backup in %s is incomplete! Halting.", backup_dir)
        sys.exit(1)
    LOG.info("Verified safety backup exists in %s", backup_dir)

    con = duckdb.connect(str(DB_PATH))
    _set_temp(con, REPO_ROOT)

    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    LOG.info("Existing tables in pipeline.duckdb: %s", sorted(tables))

    if not {"s1", "target"}.issubset(tables):
        LOG.error("Required staging tables ('s1', 'target') missing from pipeline.duckdb! Halting.")
        sys.exit(1)

    s1_count = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    target_count = con.execute("SELECT count(*) FROM target").fetchone()[0]
    LOG.info("Source S1 entities: %d | Target entities: %d", s1_count, target_count)

    checkpoint_path = OUTPUT_DIR / "inference_checkpoint.txt"
    has_checkpoint = checkpoint_path.exists()
    if has_checkpoint:
        last_id = checkpoint_path.read_text(encoding="utf-8").strip()
        LOG.info("Active checkpoint found: last completed S1 entity = '%s'. Will resume.", last_id)

    # Check if enhanced candidates table has been generated
    is_enhanced_candidates = "candidates_enhanced" in tables

    if not is_enhanced_candidates and not has_checkpoint:
        LOG.info("Generating enhanced candidate table with Rules 5A, 4A, 1B...")
        t0 = time.time()
        # Drop old candidates and selected tables to ensure clean generation
        con.execute("DROP TABLE IF EXISTS candidates")
        con.execute("DROP TABLE IF EXISTS selected")
        candidate_count = _candidate_table(con, source1="s1", target="target", out="candidates")
        con.execute("CREATE TABLE candidates_enhanced AS SELECT 1")
        con.execute("CHECKPOINT")
        LOG.info("Enhanced candidates generated: %d pairs in %.2f minutes", candidate_count, (time.time()-t0)/60.0)
    else:
        candidate_count = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
        LOG.info("Using existing enhanced candidate pool (%d pairs).", candidate_count)

    # Load production model
    matcher_path = MODEL_DIR / "matcher.joblib"
    LOG.info("Loading matcher model from %s...", matcher_path)
    artifact = joblib.load(matcher_path)
    model = artifact["model"]
    best = {"threshold": PRODUCTION_THRESHOLD, "feature_columns": artifact.get("feature_columns")}

    LOG.info("Starting scoring and export pipeline at threshold %.2f...", PRODUCTION_THRESHOLD)
    t_start = time.time()
    _score_and_export(con, model, best, candidate_count)
    t_total = time.time() - t_start
    LOG.info("Production scoring completed in %.2f minutes (%.2f hours).", t_total/60.0, t_total/3600.0)

    con.close()
    LOG.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
