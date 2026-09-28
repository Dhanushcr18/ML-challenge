import os
import sys
sys.path.insert(0, '.')
import time
import psutil
import duckdb
import numpy as np
import pandas as pd
import joblib
import subprocess
from pathlib import Path
from src.features import _text_features

process = psutil.Process()
t0 = time.time()

out_dir = Path("tmp/step7_test_run")
out_dir.mkdir(parents=True, exist_ok=True)

cand_path = out_dir / "candidate_pairs.tsv"
match_path = out_dir / "matching_results.tsv"
test_slice_s1_path = out_dir / "test_slice_source1.tsv"
db_path = out_dir / "test_pipeline_20pct.duckdb"

if db_path.exists():
    db_path.unlink()

print("="*80)
print("STEP 7: REAL 20% TEST SET CANDIDATE GENERATION, SCORING & OFFICIAL VALIDATION")
print("="*80)

con = duckdb.connect(str(db_path))
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='3GB'")
con.execute("SET preserve_insertion_order=false")

def norm_sql(expr, address=False):
    x = f"lower(trim(regexp_replace(regexp_replace(coalesce({expr}, ''), '[^[:alnum:] ]', ' ', 'g'), '\\s+', ' ', 'g')))"
    if address:
        for a, b in (("street","st"),("road","rd"),("avenue","ave"),("boulevard","blvd"),("lane","ln"),("drive","dr"),("apartment","apt"),("suite","ste"),("highway","hwy")):
            x = f"regexp_replace({x}, '\\b{a}\\b', '{b}', 'g')"
    else:
        for a, b in (("corporation","corp"),("corporate","corp"),("private","pvt"),("limited","ltd")):
            x = f"regexp_replace({x}, '\\b{a}\\b', '{b}', 'g')"
    return x

# 1. Ingest test sets (20% S1 slice, 100% target search space)
t_ingest = time.time()
print("\n1. Ingesting 20% test_source1 slice and 100% test target (s2 + s3)...")

con.execute(f"""
    CREATE TABLE full_s1 AS 
    SELECT entity_id, business_name, business_address, country,
           {norm_sql('business_name')} AS name_norm,
           {norm_sql('business_address', True)} AS address_norm,
           lower(trim(country)) AS country_norm
    FROM read_csv_auto('dataset/test/test_source1.tsv', delim='\\t', all_varchar=true, header=true)
""")
total_full_s1 = con.execute("SELECT count(*) FROM full_s1").fetchone()[0]

con.execute("""
    CREATE TABLE s1 AS
    SELECT * FROM (
        SELECT *, row_number() OVER (PARTITION BY country_norm ORDER BY hash(entity_id)) as rn,
                  count(*) OVER (PARTITION BY country_norm) as c_cnt
        FROM full_s1
    ) WHERE rn <= ceil(c_cnt * 0.20)
""")
slice_s1_count = con.execute("SELECT count(*) FROM s1").fetchone()[0]
con.execute("DROP TABLE full_s1")

# Export test slice S1 to a standalone TSV for the validator
con.execute(f"""
    COPY (SELECT entity_id, business_name, business_address, country FROM s1)
    TO '{str(test_slice_s1_path).replace(chr(92), '/')}' (HEADER, DELIMITER '\t')
""")

con.execute(f"""
    CREATE TABLE target AS 
    SELECT entity_id, business_name, business_address, country,
           {norm_sql('business_name')} AS name_norm,
           {norm_sql('business_address', True)} AS address_norm,
           lower(trim(country)) AS country_norm
    FROM read_csv_auto('dataset/test/test_source2.tsv', delim='\\t', all_varchar=true, header=true)
    UNION ALL
    SELECT entity_id, business_name, business_address, country,
           {norm_sql('business_name')} AS name_norm,
           {norm_sql('business_address', True)} AS address_norm,
           lower(trim(country)) AS country_norm
    FROM read_csv_auto('dataset/test/test_source3.tsv', delim='\\t', all_varchar=true, header=true)
""")
total_target = con.execute("SELECT count(*) FROM target").fetchone()[0]
print(f"   Test S1 20% slice: {slice_s1_count:,} (out of {total_full_s1:,}) | Full Target rows: {total_target:,} (ingested in {time.time()-t_ingest:.2f}s)")

# 2. Build staging tables
t_stage = time.time()
print("\n2. Building staging tables for 20% test S1 and 100% target...")

for side, src in (("l", "s1"), ("r", "target")):
    t_side = time.time()
    print(f"   Building {side} staging tables from {src}...")
    con.execute(f"CREATE OR REPLACE TABLE {side}fn AS SELECT country_norm, name_norm, count(*) n FROM {src} WHERE name_norm<>'' GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE {side}fa AS SELECT country_norm, address_norm, count(*) n FROM {src} WHERE address_norm<>'' GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE {side}gn AS SELECT name_norm, count(*) n FROM {src} WHERE name_norm<>'' GROUP BY 1")
    con.execute(f"CREATE OR REPLACE TABLE {side}ga AS SELECT address_norm, count(*) n FROM {src} WHERE address_norm<>'' GROUP BY 1")
    con.execute(f"CREATE OR REPLACE TABLE {side}fp AS SELECT country_norm, substr(split_part(name_norm,' ',1),1,4) prefix, count(*) n FROM {src} WHERE length(name_norm)>=4 GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE {side}combo AS SELECT country_norm, substr(split_part(name_norm,' ',1),1,3) np, substr(split_part(address_norm,' ',1),1,4) ap, count(*) n FROM {src} WHERE length(name_norm)>=3 AND length(address_norm)>=4 GROUP BY 1, 2, 3")
    con.execute(f"CREATE OR REPLACE TABLE {side}combo_g AS SELECT substr(split_part(name_norm,' ',1),1,4) np, substr(split_part(address_norm,' ',1),1,4) ap, count(*) n FROM {src} WHERE length(name_norm)>=4 AND length(address_norm)>=4 GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE {side}nw AS SELECT entity_id, country_norm, regexp_extract(name_norm, '[a-z]{{5,}}', 0) token FROM {src} WHERE regexp_extract(name_norm, '[a-z]{{5,}}', 0) <> ''")
    con.execute(f"CREATE OR REPLACE TABLE {side}n2 AS SELECT entity_id, country_norm, regexp_extract(name_norm, '^[^ ]+ ([a-z]{{4,}})', 1) token FROM {src} WHERE regexp_extract(name_norm, '^[^ ]+ ([a-z]{{4,}})', 1) NOT IN ('','inc','llc','ltd','corp','pvt','company','limited','group','services')")
    con.execute(f"CREATE OR REPLACE TABLE {side}aw AS SELECT entity_id, country_norm, regexp_extract(address_norm, '[a-z]{{4,}}', 0) token FROM {src} WHERE regexp_extract(address_norm, '[a-z]{{4,}}', 0) <> ''")
    con.execute(f"CREATE OR REPLACE TABLE {side}nf AS SELECT country_norm, token, count(*) n FROM {side}nw GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE {side}n2f AS SELECT country_norm, token, count(*) n FROM {side}n2 GROUP BY 1, 2")
    con.execute(f"CREATE OR REPLACE TABLE {side}afw AS SELECT country_norm, token, count(*) n FROM {side}aw GROUP BY 1, 2")

    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_clean_num AS
        SELECT entity_id, country_norm,
               ltrim(regexp_extract(address_norm, '([0-9]{{1,6}})', 1), '0') as clean_num
        FROM {src}
        WHERE address_norm <> '' AND regexp_extract(address_norm, '([0-9]{{1,6}})', 1) <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_name_tok2_num AS
        SELECT a.entity_id, a.country_norm, a.token as name_tok2, b.clean_num
        FROM (
          SELECT entity_id, country_norm, token,
                 row_number() OVER (PARTITION BY entity_id ORDER BY length(token) DESC, token) as rn
          FROM (
            SELECT entity_id, country_norm, unnest(string_split(name_norm, ' ')) as token
            FROM {src} WHERE name_norm <> ''
          )
          WHERE length(token) >= 5 AND token NOT IN ('inc','llc','ltd','corp','pvt','private','limited','corporation','company','co','group','services','enterprises','the','and','solutions','technologies','consulting','associates','foundation')
        ) a
        JOIN {side}_clean_num b USING (entity_id, country_norm)
        WHERE a.rn = 2 AND b.clean_num <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_name_folded AS
        SELECT a.entity_id, a.country_norm,
               substr(replace(a.name_norm, ' ', ''), 1, 4) as fold_p4,
               b.clean_num
        FROM {src} a
        JOIN {side}_clean_num b USING (entity_id, country_norm)
        WHERE b.clean_num <> '' AND length(a.name_norm) >= 4
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_multi_num_tok AS
        SELECT n.entity_id, n.country_norm, n.clean_num, t.token as addr_tok1
        FROM (
            SELECT DISTINCT entity_id, country_norm, ltrim(num, '0') as clean_num
            FROM (
                SELECT entity_id, country_norm, unnest(regexp_extract_all(address_norm, '[0-9]{{1,6}}')) as num
                FROM {src} WHERE address_norm <> ''
            )
            WHERE ltrim(num, '0') <> ''
        ) n
        JOIN (
          SELECT entity_id, country_norm, token,
                 row_number() OVER (PARTITION BY entity_id ORDER BY length(token) DESC, token) as rn
          FROM (
            SELECT entity_id, country_norm, unnest(string_split(address_norm, ' ')) as token
            FROM {src} WHERE address_norm <> ''
          )
          WHERE length(token) >= 5 AND token NOT IN ('st','rd','ave','blvd','ln','dr','apt','ste','hwy','near','opp','opposite','floor','plot','flat','shop','no','block','road','street','avenue','lane','drive','delhi','mumbai','kolkata','chennai','bangalore','hyderabad','pune','city','state','india','us','district','nagar','extn','colony')
        ) t ON t.entity_id = n.entity_id AND t.country_norm = n.country_norm AND t.rn = 1
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_addr_num_3plus AS
        SELECT DISTINCT entity_id, country_norm, ltrim(num, '0') as clean_num
        FROM (
          SELECT entity_id, country_norm, unnest(regexp_extract_all(address_norm, '[0-9]{{3,8}}')) as num
          FROM {src} WHERE address_norm <> ''
        )
        WHERE ltrim(num, '0') <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_addr_2tok_pairs AS
        SELECT a.entity_id, a.country_norm, a.token as tok1, b.token as tok2
        FROM (
          SELECT entity_id, country_norm, token,
                 row_number() OVER (PARTITION BY entity_id ORDER BY length(token) DESC, token) as trn
          FROM (
              SELECT DISTINCT entity_id, country_norm, token
              FROM (
                SELECT entity_id, country_norm, unnest(string_split(address_norm, ' ')) as token
                FROM {src} WHERE address_norm <> ''
              )
              WHERE length(token) >= 4 AND regexp_matches(token, '^[a-z]+$')
                AND token NOT IN ('street','road','avenue','lane','drive','apartment','suite','highway','roadway','near','opp','opposite','floor','plot','flat','shop','block','delhi','mumbai','kolkata','chennai','bangalore','hyderabad','pune','city','state','india','us','district','nagar','extn','colony')
          )
        ) a
        JOIN (
          SELECT entity_id, country_norm, token,
                 row_number() OVER (PARTITION BY entity_id ORDER BY length(token) DESC, token) as trn
          FROM (
              SELECT DISTINCT entity_id, country_norm, token
              FROM (
                SELECT entity_id, country_norm, unnest(string_split(address_norm, ' ')) as token
                FROM {src} WHERE address_norm <> ''
              )
              WHERE length(token) >= 4 AND regexp_matches(token, '^[a-z]+$')
                AND token NOT IN ('street','road','avenue','lane','drive','apartment','suite','highway','roadway','near','opp','opposite','floor','plot','flat','shop','block','delhi','mumbai','kolkata','chennai','bangalore','hyderabad','pune','city','state','india','us','district','nagar','extn','colony')
          )
        ) b ON a.entity_id = b.entity_id AND a.country_norm = b.country_norm AND a.token < b.token
        WHERE a.trn <= 5 AND b.trn <= 5
    """)
    print(f"   {side} tables built in {time.time()-t_side:.2f}s")

# Frequency tables
con.execute("CREATE OR REPLACE TABLE r_fname_tok2_num AS SELECT country_norm, name_tok2, clean_num, count(*) n FROM r_name_tok2_num GROUP BY 1, 2, 3")
con.execute("CREATE OR REPLACE TABLE r_ffold_p4_num AS SELECT country_norm, fold_p4, clean_num, count(*) n FROM r_name_folded GROUP BY 1, 2, 3")
con.execute("CREATE OR REPLACE TABLE r_fmulti_num_tok AS SELECT country_norm, clean_num, addr_tok1, count(*) n FROM r_multi_num_tok GROUP BY 1, 2, 3")
con.execute("CREATE OR REPLACE TABLE r_faddr_num_3plus AS SELECT country_norm, clean_num, count(*) n FROM r_addr_num_3plus GROUP BY 1, 2")
con.execute("CREATE OR REPLACE TABLE r_faddr_2tok AS SELECT country_norm, tok1, tok2, count(*) n FROM r_addr_2tok_pairs GROUP BY 1, 2, 3")
con.execute("CHECKPOINT")
print(f"   Staging completed in {time.time()-t_stage:.2f}s")

# 3. Candidate Generation Join
t_cand = time.time()
print("\n3. Generating candidate pairs for 20% test S1 slice against 100% target...")

cand_sql = """
CREATE OR REPLACE TABLE candidates AS
SELECT DISTINCT source1_entity_id, candidate_entity_id FROM (
  SELECT source1_entity_id, candidate_entity_id FROM (
    SELECT source1_entity_id, candidate_entity_id,
           row_number() OVER (PARTITION BY source1_entity_id ORDER BY evidence DESC, candidate_entity_id) as rn
    FROM (
      SELECT source1_entity_id, candidate_entity_id, sum(evidence) as evidence FROM (
        -- 1. Exact Name + Country
        SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id, 10 evidence
        FROM s1 a JOIN target b USING (country_norm, name_norm)
        JOIN lfn af ON af.country_norm=a.country_norm AND af.name_norm=a.name_norm
        JOIN rfn bf ON bf.country_norm=b.country_norm AND bf.name_norm=b.name_norm
        WHERE af.n <= 300 AND bf.n <= 300

        UNION ALL
        -- 2. Exact Name (global)
        SELECT a.entity_id, b.entity_id, 9
        FROM s1 a JOIN target b USING (name_norm)
        JOIN lgn af ON af.name_norm=a.name_norm
        JOIN rgn bf ON bf.name_norm=b.name_norm
        WHERE af.n <= 80 AND bf.n <= 80

        UNION ALL
        -- 3. Exact Address + Country
        SELECT a.entity_id, b.entity_id, 8
        FROM s1 a JOIN target b USING (country_norm, address_norm)
        JOIN lfa af ON af.country_norm=a.country_norm AND af.address_norm=a.address_norm
        JOIN rfa bf ON bf.country_norm=b.country_norm AND bf.address_norm=b.address_norm
        WHERE af.n <= 200 AND bf.n <= 200

        UNION ALL
        -- 4. Exact Address (global)
        SELECT a.entity_id, b.entity_id, 7
        FROM s1 a JOIN target b USING (address_norm)
        JOIN lga af ON af.address_norm=a.address_norm
        JOIN rga bf ON bf.address_norm=b.address_norm
        WHERE af.n <= 80 AND bf.n <= 80

        UNION ALL
        -- 5. Combo: Name Prefix (3) + Address Prefix (4) within Country
        SELECT a.entity_id, b.entity_id, 6
        FROM s1 a JOIN target b
          ON a.country_norm=b.country_norm
          AND substr(split_part(a.name_norm,' ',1),1,3) = substr(split_part(b.name_norm,' ',1),1,3)
          AND substr(split_part(a.address_norm,' ',1),1,4) = substr(split_part(b.address_norm,' ',1),1,4)
        JOIN lcombo af ON af.country_norm=a.country_norm AND af.np=substr(split_part(a.name_norm,' ',1),1,3) AND af.ap=substr(split_part(a.address_norm,' ',1),1,4)
        JOIN rcombo bf ON bf.country_norm=b.country_norm AND bf.np=substr(split_part(b.name_norm,' ',1),1,3) AND bf.ap=substr(split_part(b.address_norm,' ',1),1,4)
        WHERE af.n <= 200 AND bf.n <= 200

        UNION ALL
        -- 6. Combo Global (country-agnostic)
        SELECT a.entity_id, b.entity_id, 5
        FROM s1 a JOIN target b
          ON substr(split_part(a.name_norm,' ',1),1,4) = substr(split_part(b.name_norm,' ',1),1,4)
          AND substr(split_part(a.address_norm,' ',1),1,4) = substr(split_part(b.address_norm,' ',1),1,4)
        JOIN lcombo_g af ON af.np=substr(split_part(a.name_norm,' ',1),1,4) AND af.ap=substr(split_part(a.address_norm,' ',1),1,4)
        JOIN rcombo_g bf ON bf.np=substr(split_part(b.name_norm,' ',1),1,4) AND bf.ap=substr(split_part(b.address_norm,' ',1),1,4)
        WHERE af.n <= 100 AND bf.n <= 100

        UNION ALL
        -- 7. Name Prefix (4 chars) within Country
        SELECT a.entity_id, b.entity_id, 4
        FROM s1 a JOIN target b
          ON a.country_norm=b.country_norm AND substr(split_part(a.name_norm,' ',1),1,4) = substr(split_part(b.name_norm,' ',1),1,4)
        JOIN lfp af ON af.country_norm=a.country_norm AND af.prefix=substr(split_part(a.name_norm,' ',1),1,4)
        JOIN rfp bf ON bf.country_norm=b.country_norm AND bf.prefix=substr(split_part(b.name_norm,' ',1),1,4)
        WHERE af.n <= 120 AND bf.n <= 120

        UNION ALL
        -- 8. First 5-letter name token within Country
        SELECT a.entity_id, b.entity_id, 4
        FROM lnw a JOIN rnw b USING (country_norm, token)
        JOIN lnf af USING (country_norm, token)
        JOIN rnf bf USING (country_norm, token)
        WHERE af.n <= 250 AND bf.n <= 250

        UNION ALL
        -- 9. Second 4-letter name token within Country
        SELECT a.entity_id, b.entity_id, 3
        FROM ln2 a JOIN rn2 b USING (country_norm, token)
        JOIN ln2f af USING (country_norm, token)
        JOIN rn2f bf USING (country_norm, token)
        WHERE af.n <= 250 AND bf.n <= 250

        UNION ALL
        -- 10. Address 4-letter token within Country
        SELECT a.entity_id, b.entity_id, 2
        FROM law a JOIN raw b USING (country_norm, token)
        JOIN lafw af USING (country_norm, token)
        JOIN rafw bf USING (country_norm, token)
        WHERE af.n <= 200 AND bf.n <= 200
      ) raw_pairs
      GROUP BY 1, 2
    ) evidence_pairs
  ) ranked
  WHERE rn <= 40

  UNION ALL
  -- Rule 5A: R_name_tok2_num
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM l_name_tok2_num a
  JOIN r_name_tok2_num b USING (country_norm, name_tok2, clean_num)
  JOIN r_fname_tok2_num f USING (country_norm, name_tok2, clean_num)
  WHERE f.n <= 100

  UNION ALL
  -- Rule 4A: R_folded_p4_num
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM l_name_folded a
  JOIN r_name_folded b USING (country_norm, fold_p4, clean_num)
  JOIN r_ffold_p4_num f USING (country_norm, fold_p4, clean_num)
  WHERE f.n <= 100

  UNION ALL
  -- Rule 1B: R_empty_num_tok1
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM l_multi_num_tok a
  JOIN r_multi_num_tok b USING (country_norm, clean_num, addr_tok1)
  JOIN target tgt ON tgt.entity_id = b.entity_id
  JOIN r_fmulti_num_tok f USING (country_norm, clean_num, addr_tok1)
  WHERE (tgt.name_norm IS NULL OR tgt.name_norm = '')
    AND f.n <= 50

  UNION ALL
  -- Rule 9A: Address Numeric Match (3+ digits) within Country
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM l_addr_num_3plus a
  JOIN r_addr_num_3plus b USING (country_norm, clean_num)
  JOIN r_faddr_num_3plus f USING (country_norm, clean_num)
  WHERE f.n <= 80

  UNION ALL
  -- Rule 9B: Address 2-Token Alphabetic Match (4+ chars each) within Country
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM l_addr_2tok_pairs a
  JOIN r_addr_2tok_pairs b USING (country_norm, tok1, tok2)
  JOIN r_faddr_2tok f USING (country_norm, tok1, tok2)
  WHERE f.n <= 50
) all_cands
"""

con.execute(cand_sql)
cand_duration = time.time() - t_cand
total_candidates = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
distinct_s1_cands = con.execute("SELECT count(DISTINCT source1_entity_id) FROM candidates").fetchone()[0]
avg_cands = total_candidates / slice_s1_count

print(f"   Candidate table created in {cand_duration:.2f}s ({cand_duration/60:.2f} min)")
print(f"   Actual Total Candidates:      {total_candidates:,}")
print(f"   Actual Distinct S1 with Cands: {distinct_s1_cands:,} ({distinct_s1_cands/slice_s1_count*100:.2f}%)")
print(f"   Actual Avg Candidates / S1:    {avg_cands:.4f}")

# 4. Write candidate_pairs.tsv in streaming batches
t_write_cands = time.time()
print("\n4. Streaming candidate_pairs.tsv to disk...")
s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM s1 ORDER BY entity_id").fetchall()]

with open(cand_path, "w", encoding="utf-8", newline="") as handle:
    handle.write("source1_entity_id\tcandidate_entity_ids\n")
    BATCH_SIZE = 5000
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

print(f"   Wrote candidate_pairs.tsv ({cand_path.stat().st_size / 1024**2:.2f} MB) in {time.time()-t_write_cands:.2f}s")

# 5. Featurize & Score Candidates with Retrained Model at Threshold 0.97
t_score = time.time()
print("\n5. Featurizing and scoring candidate pairs with retrained model (Threshold = 0.97)...")
model_dict = joblib.load("tmp/retrained_rule9_model.joblib")
model = model_dict["model"]
feature_cols = model_dict["feature_columns"]
threshold = 0.97

con.execute("CREATE OR REPLACE TABLE selected(source1_entity_id VARCHAR, candidate_entity_id VARCHAR, PRIMARY KEY(source1_entity_id, candidate_entity_id))")

INFERENCE_BATCH = 2000
total_scored = 0
total_matched = 0

for batch_start in range(0, len(s1_ids), INFERENCE_BATCH):
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
        rows = []
        for row in df.itertuples(index=False):
            nf = _text_features(row.an, row.bn, "name")
            af = _text_features(row.aa, row.ba, "address")
            nf["country_exact"] = int(bool(row.ac) and row.ac == row.bc)
            nf["combined_similarity"] = 0.5 * (nf["name_char_jaccard"] + nf.get("name_token_jaccard", 0.0))
            rows.append({**nf, **af})
        X = pd.DataFrame(rows)[feature_cols]
        probs = model.predict_proba(X)[:, 1]
        
        keep_mask = probs >= threshold
        kept = df.loc[keep_mask, ["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
        if not kept.empty:
            con.register("kept_df", kept)
            con.execute("INSERT OR IGNORE INTO selected SELECT CAST(source1_entity_id AS VARCHAR), CAST(candidate_entity_id AS VARCHAR) FROM kept_df")
            con.unregister("kept_df")
            total_matched += len(kept)
        total_scored += len(df)
        
    if (batch_start // INFERENCE_BATCH) % 25 == 0 or batch_start + INFERENCE_BATCH >= len(s1_ids):
        elapsed = time.time() - t_score
        rate = total_scored / max(0.1, elapsed)
        print(f"   Batch {batch_start // INFERENCE_BATCH + 1}/{(len(s1_ids)+INFERENCE_BATCH-1)//INFERENCE_BATCH} | Scored: {total_scored:,} | Matched: {total_matched:,} | Rate: {rate:,.1f} pairs/s | Elapsed: {elapsed:.1f}s")

score_duration = time.time() - t_score
print(f"   Inference completed in {score_duration:.2f}s ({score_duration/60:.2f} min)")

# 6. Stream matching_results.tsv to disk
t_write_match = time.time()
print("\n6. Streaming matching_results.tsv to disk...")

with open(match_path, "w", encoding="utf-8", newline="") as handle:
    handle.write("source1_entity_id\tmatched_entity_ids\n")
    BATCH_SIZE = 5000
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

# 7. Run official validate_submission.py
print("\n" + "="*80)
print("7. RUNNING OFFICIAL VALIDATE_SUBMISSION.PY")
print("="*80)
cmd = [
    sys.executable, "utils/validate_submission.py",
    "--matching", str(match_path),
    "--candidate", str(cand_path),
    "--test-dir", str(out_dir)
]
# Note: validator looks for test_source1.tsv in --test-dir, so link/copy test_slice_source1.tsv as test_source1.tsv in out_dir
target_s1_link = out_dir / "test_source1.tsv"
if target_s1_link.exists():
    target_s1_link.unlink()
import shutil
shutil.copy(test_slice_s1_path, target_s1_link)

print("Running command:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
print(f"Validation Exit Code: {res.returncode}")

# 8. Structural Comparison vs Proven 0.813 Submission
print("\n" + "="*80)
print("8. STRUCTURAL COMPARISON VS PROVEN 0.813 BASELINE")
print("="*80)

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

comp_df = pd.DataFrame([stat_0813, stat_new], index=["Proven 0.813 Baseline (Full Test)", "New Candidate Run (20% Slice)"])
print(comp_df.to_string())

print("\n" + "="*80)
print(f"STEP 7 REAL 20% TEST RUN COMPLETED IN {time.time()-t0:.2f}s ({ (time.time()-t0)/60:.2f} min)")
print("="*80)
