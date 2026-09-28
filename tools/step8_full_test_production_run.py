import os
import sys
sys.path.insert(0, '.')
import time
import json
import psutil
import duckdb
import numpy as np
import pandas as pd
import joblib
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from src.features import _text_features

FEATURE_COLS = [
    'name_exact', 'name_char_jaccard', 'name_token_jaccard', 'name_token_containment',
    'name_tfidf_cosine', 'name_number_overlap', 'name_edit', 'name_length_delta',
    'name_missing_a', 'name_missing_b', 'country_exact', 'combined_similarity',
    'address_exact', 'address_char_jaccard', 'address_token_jaccard', 'address_token_containment',
    'address_tfidf_cosine', 'address_number_overlap', 'address_edit', 'address_length_delta',
    'address_missing_a', 'address_missing_b'
]

def worker_batch(tuples):
    rows = []
    for an, bn, aa, ba, ac, bc in tuples:
        nf = _text_features(an, bn, "name")
        af = _text_features(aa, ba, "address")
        nf["country_exact"] = int(bool(ac) and ac == bc)
        nf["combined_similarity"] = 0.5 * (nf["name_char_jaccard"] + nf.get("name_token_jaccard", 0.0))
        rows.append({**nf, **af})
    return pd.DataFrame(rows)[FEATURE_COLS]

def norm_sql(expr, address=False):
    x = f"lower(trim(regexp_replace(regexp_replace(coalesce({expr}, ''), '[^[:alnum:] ]', ' ', 'g'), '\\s+', ' ', 'g')))"
    if address:
        for a, b in (("street","st"),("road","rd"),("avenue","ave"),("boulevard","blvd"),("lane","ln"),("drive","dr"),("apartment","apt"),("suite","ste"),("highway","hwy")):
            x = f"regexp_replace({x}, '\\b{a}\\b', '{b}', 'g')"
    else:
        for a, b in (("corporation","corp"),("corporate","corp"),("private","pvt"),("limited","ltd")):
            x = f"regexp_replace({x}, '\\b{a}\\b', '{b}', 'g')"
    return x

if __name__ == '__main__':
    t0 = time.time()
    out_dir = Path("tmp/step8_full_submission")
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = out_dir / "candidate_pairs.tsv"
    match_path = out_dir / "matching_results.tsv"
    checkpoint_file = out_dir / "inference_checkpoint.json"
    db_path = out_dir / "test_pipeline_full.duckdb"

    print("="*85)
    print(f"STEP 8: MULTI-PROCESS RESUMED RUN — STARTED AT {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*85)

    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA max_temp_directory_size='25GB'")
    con.execute("SET preserve_insertion_order=false")

    total_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    total_target = con.execute("SELECT count(*) FROM target").fetchone()[0]
    total_candidates = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    print(f"   Total Test S1: {total_s1:,} | Total Candidates: {total_candidates:,}")

    # Ensure candidate_pairs.tsv exists
    if not cand_path.exists() or cand_path.stat().st_size == 0:
        print("   candidate_pairs.tsv missing, regenerating...")
        s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM s1 ORDER BY entity_id").fetchall()]
        with open(cand_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("source1_entity_id\tcandidate_entity_ids\n")
            BATCH_SIZE = 10000
            for batch_start in range(0, len(s1_ids), BATCH_SIZE):
                batch = s1_ids[batch_start: batch_start + BATCH_SIZE]
                batch_df = pd.DataFrame({"entity_id": batch})
                con.register("_cand_batch", batch_df)
                rows = con.execute("""
                    SELECT s.entity_id, c.candidate_entity_id
                    FROM _cand_batch s
                    LEFT JOIN candidates c ON c.source1_entity_id = s.entity_id
                    ORDER BY s.entity_id, c.candidate_entity_id
                """).fetchall()
                con.unregister("_cand_batch")
                current = None
                ids = []
                for source_id, candidate_id in rows:
                    if current is not None and source_id != current:
                        handle.write(f"{current}\t{','.join(ids)}\n")
                        ids = []
                    current = source_id
                    if candidate_id is not None and str(candidate_id) != "" and str(candidate_id) != "None":
                        ids.append(str(candidate_id))
                if current is not None:
                    handle.write(f"{current}\t{','.join(ids)}\n")
    else:
        print(f"   candidate_pairs.tsv verified ({cand_path.stat().st_size / 1024**2:.2f} MB)")

    # Model & Config
    print("\n5. Initializing multi-process inference pool (10 workers, Threshold=0.97)...")
    model_dict = joblib.load("tmp/retrained_rule9_model.joblib")
    model = model_dict["model"]
    threshold = 0.97

    con.execute("CREATE TABLE IF NOT EXISTS selected(source1_entity_id VARCHAR, candidate_entity_id VARCHAR, PRIMARY KEY(source1_entity_id, candidate_entity_id))")

    s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM s1 ORDER BY entity_id").fetchall()]
    INFERENCE_BATCH = 4000
    total_batches = (len(s1_ids) + INFERENCE_BATCH - 1) // INFERENCE_BATCH

    start_batch_idx = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                cp_data = json.load(f)
                start_batch_idx = cp_data.get("last_completed_batch", -1) + 1
                print(f"   CHECKPOINT CONFIRMED: Resuming at batch {start_batch_idx + 1}/{total_batches} (batches 1..{start_batch_idx} skipped)")
        except Exception as e:
            print("   Checkpoint read warning:", e)

    total_matched = con.execute("SELECT count(*) FROM selected").fetchone()[0]
    print(f"   Already Matched in DB from previous batches: {total_matched:,} pairs")

    n_workers = min(10, cpu_count())
    pool = Pool(processes=n_workers)

    t_score = time.time()
    total_scored_resumed = 0

    try:
        for batch_idx in range(start_batch_idx, total_batches):
            batch_start = batch_idx * INFERENCE_BATCH
            batch = s1_ids[batch_start: batch_start + INFERENCE_BATCH]
            batch_df = pd.DataFrame({"entity_id": batch})
            con.register("_score_batch", batch_df)
            
            df = con.execute("""
                SELECT p.source1_entity_id, p.candidate_entity_id,
                       a.name_norm an, b.name_norm bn, a.address_norm aa, b.address_norm ba,
                       a.country_norm ac, b.country_norm bc
                FROM candidates p
                JOIN _score_batch sb ON sb.entity_id = p.source1_entity_id
                JOIN s1 a ON a.entity_id = p.source1_entity_id
                JOIN target b ON b.entity_id = p.candidate_entity_id
            """).df()
            con.unregister("_score_batch")
            
            if not df.empty:
                n_rows = len(df)
                chunk_size = max(1000, (n_rows + n_workers - 1) // n_workers)
                tuples = list(zip(df.an, df.bn, df.aa, df.ba, df.ac, df.bc))
                chunks = [tuples[i:i+chunk_size] for i in range(0, n_rows, chunk_size)]
                
                # Multi-process featurization across 10 workers
                res_dfs = pool.map(worker_batch, chunks)
                X = pd.concat(res_dfs, ignore_index=True)
                
                probs = model.predict_proba(X)[:, 1]
                
                keep_mask = probs >= threshold
                kept = df.loc[keep_mask, ["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
                if not kept.empty:
                    con.register("kept_df", kept)
                    con.execute("INSERT OR IGNORE INTO selected SELECT CAST(source1_entity_id AS VARCHAR), CAST(candidate_entity_id AS VARCHAR) FROM kept_df")
                    con.unregister("kept_df")
                    total_matched += len(kept)
                total_scored_resumed += len(df)
            
            # Atomic checkpoint write after every single batch
            with open(checkpoint_file, "w") as f:
                json.dump({"last_completed_batch": batch_idx, "total_matched": total_matched}, f)
                
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - t_score
                rate = total_scored_resumed / max(0.1, elapsed)
                remaining_batches = total_batches - 1 - batch_idx
                est_remaining_sec = (remaining_batches * (elapsed / max(1, batch_idx - start_batch_idx + 1)))
                print(f"   Batch {batch_idx+1}/{total_batches} | Resumed Scored: {total_scored_resumed:,} | Total Matched: {total_matched:,} | Rate: {rate:,.1f} pairs/s | Elapsed: {elapsed/60:.1f}m | ETA: {est_remaining_sec/60:.1f}m ({est_remaining_sec/3600:.2f}h)")
    finally:
        pool.close()
        pool.join()

    con.execute("CHECKPOINT")
    score_duration = time.time() - t_score
    print(f"\n   Inference completed in {score_duration:.2f}s ({score_duration/60:.2f} min / {score_duration/3600:.2f} hours)")

    # 6. Stream matching_results.tsv
    t_write_match = time.time()
    print("\n6. Streaming final matching_results.tsv to disk...")
    with open(match_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        BATCH_SIZE = 10000
        for batch_start in range(0, len(s1_ids), BATCH_SIZE):
            batch = s1_ids[batch_start: batch_start + BATCH_SIZE]
            batch_df = pd.DataFrame({"entity_id": batch})
            con.register("_match_batch", batch_df)
            rows = con.execute("""
                SELECT s.entity_id, c.candidate_entity_id
                FROM _match_batch s
                LEFT JOIN selected c ON c.source1_entity_id = s.entity_id
                ORDER BY s.entity_id, c.candidate_entity_id
            """).fetchall()
            con.unregister("_match_batch")
            
            current = None
            ids = []
            for source_id, candidate_id in rows:
                if current is not None and source_id != current:
                    handle.write(f"{current}\t{','.join(ids)}\n")
                    ids = []
                current = source_id
                if candidate_id is not None and str(candidate_id) != "" and str(candidate_id) != "None":
                    ids.append(str(candidate_id))
            if current is not None:
                handle.write(f"{current}\t{','.join(ids)}\n")

    print(f"   Wrote matching_results.tsv ({match_path.stat().st_size / 1024**2:.2f} MB) in {time.time()-t_write_match:.2f}s")
    con.close()

    # 7. Run official validate_submission.py WITH --check-ids
    print("\n" + "="*85)
    print("7. RUNNING OFFICIAL VALIDATE_SUBMISSION.PY WITH --check-ids")
    print("="*85)
    cmd = [
        sys.executable, "utils/validate_submission.py",
        "--matching", str(match_path),
        "--candidate", str(cand_path),
        "--test-dir", "dataset/test",
        "--check-ids"
    ]
    print("Running command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    print(f"Validation Exit Code: {res.returncode}")

    # 8. Structural Comparison vs Proven 0.813 Submission
    print("\n" + "="*85)
    print("8. STRUCTURAL COMPARISON VS PROVEN 0.813 BASELINE (FULL TEST)")
    print("="*85)

    def file_stats(path):
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            rows = 0
            matches = 0
            singletons = 0
            for line in f:
                rows += 1
                parts = line.strip().split("\t")
                if len(parts) > 1 and parts[1]:
                    matches += len(parts[1].split(","))
                else:
                    singletons += 1
        return {
            "header": header,
            "rows": rows,
            "total_matches": matches,
            "singletons": singletons,
            "avg_matches_per_s1": matches / max(1, rows),
            "singleton_pct": singletons / max(1, rows) * 100
        }

    stat_new = file_stats(match_path)
    stat_0813 = file_stats("tmp/final_optimization/proven_0813/matching_results.tsv")

    comp_df = pd.DataFrame([stat_0813, stat_new], index=["Proven 0.813 Baseline (Full Test)", "New Candidate (Full Test Run)"])
    print(comp_df.to_string())

    print("\n" + "="*85)
    print(f"STEP 8 FULL PRODUCTION RUN COMPLETED IN {time.time()-t0:.2f}s ({(time.time()-t0)/60:.2f} min / {(time.time()-t0)/3600:.2f} hours)")
    print("="*85)
