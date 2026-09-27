"""
Full resume run: score all remaining S1 entities after checkpoint, then export.
- Resumes from inference_checkpoint.txt
- Batch size 1000 with adaptive split
- Preserves all existing selected rows
- Writes matching_results.tsv and candidate_pairs.tsv after completion
"""
import logging
import sys
import time
from pathlib import Path

import duckdb
import joblib

OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models")
DB_PATH = OUTPUT_DIR / "pipeline.duckdb"
CHECKPOINT_PATH = OUTPUT_DIR / "inference_checkpoint.txt"
LOG_PATH = OUTPUT_DIR / "pipeline_full_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_PATH), mode="a", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)

# ── Pre-flight ────────────────────────────────────────────────────────────────
LOG.info("=== Full Resume Run ===")

check = duckdb.connect(str(DB_PATH), read_only=True)
tables = {r[0] for r in check.execute("SHOW TABLES").fetchall()}
sel_before = check.execute("SELECT count(*) FROM selected").fetchone()[0]
s1_total = check.execute("SELECT count(*) FROM s1").fetchone()[0]
check.close()

last_id = CHECKPOINT_PATH.read_text(encoding="utf-8").strip()
LOG.info("selected rows at start : %d", sel_before)
LOG.info("s1 total               : %d", s1_total)
LOG.info("checkpoint             : %s", last_id)
assert "selected" in tables, "selected table missing!"
assert sel_before >= 875_000, f"selected count {sel_before} looks wrong — aborting to protect results"
assert last_id == "S1-30056484", f"unexpected checkpoint {last_id!r}"

# ── Load patched pipeline helpers ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.duckdb_pipeline import (
    _score_one_batch,
    _new_score_stats,
    _log_score_stats,
    _write_id_pairs,
    _atomic_write_text,
    INFERENCE_SOURCE_BATCH,
    _MIN_SPLIT_SIZE,
    _SLOW_BATCH_SECS,
)

LOG.info("INFERENCE_SOURCE_BATCH = %d", INFERENCE_SOURCE_BATCH)
assert INFERENCE_SOURCE_BATCH == 1_000, f"Expected 1000, got {INFERENCE_SOURCE_BATCH}"

payload = joblib.load(MODEL_DIR / "matcher.joblib")
model = payload["model"]
threshold = payload["threshold"]
feature_columns = payload.get("feature_columns") or list(getattr(model, "feature_names_in_", []) or [])
LOG.info("Model loaded. threshold=%.3f", threshold)

# ── Open DB and load remaining S1 IDs ─────────────────────────────────────────
con = duckdb.connect(str(DB_PATH))
con.execute("SET memory_limit='6GB'")

s1_ids = [r[0] for r in con.execute(
    "SELECT entity_id FROM s1 WHERE entity_id > ? ORDER BY entity_id",
    [last_id]
).fetchall()]
LOG.info("Remaining S1 entities  : %d (after checkpoint)", len(s1_ids))

# ── Scoring loop ──────────────────────────────────────────────────────────────
stats = _new_score_stats()
total_candidates = 0
total_matched_this_run = 0
batch_size = INFERENCE_SOURCE_BATCH
batch_num = 0
max_batch_secs = 0.0
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
    max_batch_secs = max(max_batch_secs, elapsed)
    total_candidates += cands
    total_matched_this_run += matched

    # Checkpoint AFTER successful batch
    _atomic_write_text(CHECKPOINT_PATH, str(batch_ids[-1]))

    sel_now = con.execute("SELECT count(*) FROM selected").fetchone()[0]
    entities_done = sel_before + total_matched_this_run  # approx; actual is sel_now
    wall = time.monotonic() - run_start
    LOG.info(
        "Batch %d | S1 %s..%s (%d) | cands=%d | matched=%d | "
        "batch_time=%.1fs | total_selected=%d | wall=%.0fs",
        batch_num, batch_ids[0], batch_ids[-1], len(batch_ids),
        cands, matched, elapsed, sel_now, wall,
    )

    # Adaptive split
    if elapsed > _SLOW_BATCH_SECS and batch_size > _MIN_SPLIT_SIZE:
        new_size = max(_MIN_SPLIT_SIZE, batch_size // 2)
        LOG.warning(
            "Slow batch (%.1fs > %ds): reducing batch_size %d -> %d",
            elapsed, _SLOW_BATCH_SECS, batch_size, new_size,
        )
        batch_size = new_size

    # Periodic DuckDB WAL flush
    if batch_num % 50 == 0:
        con.execute("CHECKPOINT")
        LOG.info("DuckDB CHECKPOINT at batch %d", batch_num)

    batch_start += len(batch_ids)

con.execute("CHECKPOINT")
wall_total = time.monotonic() - run_start
_log_score_stats(stats, prefix="Full-inference score distribution")

sel_final = con.execute("SELECT count(*) FROM selected").fetchone()[0]
LOG.info("=== SCORING COMPLETE ===")
LOG.info("Wall time              : %.0fs (%.1f min)", wall_total, wall_total / 60)
LOG.info("Batches run            : %d", batch_num)
LOG.info("Candidates scored      : %d", total_candidates)
LOG.info("Matches this run       : %d", total_matched_this_run)
LOG.info("Total selected (all)   : %d", sel_final)
LOG.info("Max batch runtime      : %.1fs", max_batch_secs)

# ── Export files ──────────────────────────────────────────────────────────────
LOG.info("Writing candidate_pairs.tsv ...")
candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
_write_id_pairs(con, candidate_path, "candidate_entity_ids", "s1", "candidates")
LOG.info("candidate_pairs.tsv written: %.1f MB", candidate_path.stat().st_size / 1e6)

LOG.info("Writing matching_results.tsv ...")
matching_path = OUTPUT_DIR / "matching_results.tsv"
_write_id_pairs(con, matching_path, "matched_entity_ids", "s1", "selected")
LOG.info("matching_results.tsv written: %.1f MB", matching_path.stat().st_size / 1e6)

# ── Quick sanity checks ───────────────────────────────────────────────────────
mr_rows = con.execute(
    f"SELECT count(*) FROM read_csv_auto('{str(matching_path).replace(chr(92), '/')}', delim='\\t', header=true)"
).fetchone()[0]
cp_rows = con.execute(
    f"SELECT count(*) FROM read_csv_auto('{str(candidate_path).replace(chr(92), '/')}', delim='\\t', header=true)"
).fetchone()[0]
distinct_matched = con.execute("SELECT count(DISTINCT source1_entity_id) FROM selected").fetchone()[0]
singletons = s1_total - distinct_matched

LOG.info("matching_results.tsv rows  : %d (expected %d)", mr_rows, s1_total)
LOG.info("candidate_pairs.tsv rows   : %d", cp_rows)
LOG.info("S1 with matches            : %d", distinct_matched)
LOG.info("Singletons                 : %d", singletons)
LOG.info("Match rate                 : %.2f%%", 100.0 * distinct_matched / s1_total)

con.close()

# Remove checkpoint on clean completion
CHECKPOINT_PATH.unlink(missing_ok=True)
LOG.info("Checkpoint removed — run complete.")
LOG.info("=== RUN FINISHED — awaiting validator ===")
