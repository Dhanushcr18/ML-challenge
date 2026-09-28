import sys
sys.path.insert(0, '.')
import time
import duckdb
import numpy as np
import pandas as pd
import joblib
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

if __name__ == '__main__':
    print("Testing Standalone Multi-Core Worker on 100,000 real pairs...")
    con = duckdb.connect('tmp/experiment_blocking.duckdb', read_only=True)
    df = con.execute("""
        SELECT s.name_norm an, t.name_norm bn, s.address_norm aa, t.address_norm ba,
               s.country_norm ac, t.country_norm bc
        FROM full_train_candidates p
        JOIN s1 s ON s.entity_id = p.source1_entity_id
        JOIN target t ON t.entity_id = p.candidate_entity_id
        LIMIT 100000
    """).df()
    con.close()

    model_dict = joblib.load("tmp/retrained_rule9_model.joblib")
    model = model_dict["model"]

    t0 = time.time()
    n_cores = min(10, cpu_count())
    chunk_size = len(df) // n_cores + 1
    tuples = list(zip(df.an, df.bn, df.aa, df.ba, df.ac, df.bc))
    chunks = [tuples[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(processes=n_cores) as pool:
        dfs = pool.map(worker_batch, chunks)

    X = pd.concat(dfs, ignore_index=True)
    probs = model.predict_proba(X)[:, 1]
    dur = time.time() - t0
    rate = len(df) / dur

    print(f"Workers: {n_cores} processes")
    print(f"Featurize + Inference time for 100,000 pairs: {dur:.2f}s")
    print(f"New Throughput Rate: {rate:,.1f} pairs / second")
    print(f"Speedup vs 4,300 pairs/sec: {rate/4306.0:.2f}x")
