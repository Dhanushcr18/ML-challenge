"""
Resume test: score the next 5,000 S1 entities after the checkpoint S1-30056484.
Does NOT modify the checkpoint or call the full pipeline.
Reports per-batch timing, selected counts, and max batch runtime.
"""
import logging
import sys
import time
from pathlib import Path

import duckdb
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/resume_test_5000.log", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models")
DB_PATH = OUTPUT_DIR / "pipeline.duckdb"
CHECKPOINT_PATH = OUTPUT_DIR / "inference_checkpoint.txt"
RESUME_LIMIT = 5_000  # how many S1 entities to process in this test

# ── Pre-flight checks ──────────────────────────────────────────────────────────
LOG.info("=== Resume Test (5,000 entities) ===")
LOG.info("Verifying DB state before starting...")

check = duckdb.connect(str(DB_PATH), read_only=True)
tables = {r[0] for r in check.execute("SHOW TABLES").fetchall()}
sel_before = check.execute("SELECT count(*) FROM selected").fetchone()[0]
s1_count = check.execute("SELECT count(*) FROM s1").fetchone()[0]
check.close()

LOG.info("Tables present: %s", sorted(tables))
LOG.info("selected rows before test: %d", sel_before)
LOG.info("s1 total rows: %d", s1_count)
assert "selected" in tables, "selected table missing!"
assert sel_before > 0, "selected table is empty — checkpoint may be stale!"

last_id = CHECKPOINT_PATH.read_text(encoding="utf-8").strip()
LOG.info("Checkpoint: %s", last_id)

# ── Import patched pipeline module ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.duckdb_pipeline import (
    _score_one_batch,
    _new_score_stats,
    _update_score_stats,
    _log_score_stats,
    INFERENCE_SOURCE_BATCH,
    _MIN_SPLIT_SIZE,
    _SLOW_BATCH_SECS,
    _atomic_write_text,
)
import numpy as np

LOG.info("INFERENCE_SOURCE_BATCH = %d (expected 1000)", INFERENCE_SOURCE_BATCH)
assert INFERENCE_SOURCE_BATCH == 1_000, f"Batch size should be 1000, got {INFERENCE_SOURCE_BATCH}"

# ── Load model ────────────────────────────────────────────────────────────────
payload = joblib.load(MODEL_DIR / "matcher.joblib")
model = payload["model"]
threshold = payload["threshold"]
feature_columns = payload.get("feature_columns") or list(getattr(model, "feature_names_in_", []) or [])
LOG.info("Model loaded. Threshold=%.3f", threshold)

# ── Open DB and fetch the test slice of S1 IDs ────────────────────────────────
con = duckdb.connect(str(DB_PATH))
con.execute("SET memory_limit='6GB'")

s1_ids = [r[0] for r in con.execute(
    "SELECT entity_id FROM s1 WHERE entity_id > ? ORDER BY entity_id LIMIT ?",
    [last_id, RESUME_LIMIT]
).fetchall()]

LOG.info("Test slice: %d S1 entities starting after %s", len(s1_ids), last_id)
if not s1_ids:
    LOG.error("No S1 entities found after checkpoint — already complete?")
    con.close()
    sys.exit(1)

start_s1 = s1_ids[0]
end_s1_planned = s1_ids[-1]

# ── Run batches ───────────────────────────────────────────────────────────────
stats = _new_score_stats()
total_candidates = 0
total_matched_this_run = 0
batch_size = INFERENCE_SOURCE_BATCH
batch_num = 0
max_batch_secs = 0.0
batch_times = []
run_start = time.monotonic()

batch_start = 0
while batch_start < len(s1_ids):
    batch_ids = s1_ids[batch_start: batch_start + batch_size]
    batch_num += 1
    t0 = time.monotonic()

    cands, matched = _score_one_batch(
        con, model, threshold, batch_ids, "selected", feature_columns, stats, None
    )
    elapsed = time.monotonic() - t0
    batch_times.append(elapsed)
    max_batch_secs = max(max_batch_secs, elapsed)
    total_candidates += cands
    total_matched_this_run += matched

    sel_now = con.execute("SELECT count(*) FROM selected").fetchone()[0]
    LOG.info(
        "Batch %d | S1 %s..%s (%d) | cands=%d | matched=%d | "
        "batch_time=%.1fs | total_selected=%d",
        batch_num, batch_ids[0], batch_ids[-1], len(batch_ids),
        cands, matched, elapsed, sel_now,
    )

    # Adaptive split on slow batch
    if elapsed > _SLOW_BATCH_SECS and batch_size > _MIN_SPLIT_SIZE:
        new_size = max(_MIN_SPLIT_SIZE, batch_size // 2)
        LOG.warning(
            "Slow batch (%.1fs > %ds): reducing batch size %d -> %d",
            elapsed, _SLOW_BATCH_SECS, batch_size, new_size,
        )
        batch_size = new_size

    batch_start += len(batch_ids)

con.execute("CHECKPOINT")
wall = time.monotonic() - run_start

# ── Final report ──────────────────────────────────────────────────────────────
sel_after = con.execute("SELECT count(*) FROM selected").fetchone()[0]
new_checkpoint_candidate = s1_ids[-1]
con.close()

_log_score_stats(stats, prefix="Resume-test score distribution")

LOG.info("=== RESUME TEST COMPLETE ===")
LOG.info("Start S1 ID            : %s", start_s1)
LOG.info("End S1 ID (actual)     : %s", s1_ids[batch_start - 1] if batch_start > 0 else s1_ids[-1])
LOG.info("S1 entities processed  : %d / %d requested", len(s1_ids), RESUME_LIMIT)
LOG.info("Batches run            : %d (size=%d, adaptive min=%d)", batch_num, INFERENCE_SOURCE_BATCH, _MIN_SPLIT_SIZE)
LOG.info("Candidates scored      : %d", total_candidates)
LOG.info("Matches selected       : %d (this run)", total_matched_this_run)
LOG.info("selected total before  : %d", sel_before)
LOG.info("selected total after   : %d", sel_after)
LOG.info("Net new in selected    : %d", sel_after - sel_before)
LOG.info("Max batch runtime      : %.1fs", max_batch_secs)
LOG.info("Wall time              : %.1fs", wall)
LOG.info("OOM / timeout errors   : 0 (completed normally)")
LOG.info("Next checkpoint would be: %s", new_checkpoint_candidate)
LOG.info("NOTE: Checkpoint NOT updated — this is a test run only.")
