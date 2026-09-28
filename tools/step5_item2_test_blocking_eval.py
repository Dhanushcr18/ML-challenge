import os
import sys
import time
import psutil
import duckdb
import pandas as pd

process = psutil.Process()
t_start = time.time()

print("="*60)
print("TEST SET BLOCKING EVALUATION (10% REPRESENTATIVE SLICE)")
print("="*60)

db_path = "tmp/test_blocking_10pct.duckdb"
if os.path.exists(db_path):
    os.remove(db_path)

con = duckdb.connect(db_path)
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='4GB'")
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

t0 = time.time()
print("1. Loading 10% test_source1 slice and full test target...")

# Load full test_source1 first to sample exactly 10% (stratified by country)
con.execute(f"""
    CREATE TABLE full_s1 AS 
    SELECT entity_id, business_name, business_address, country,
           {norm_sql('business_name')} AS name_norm,
           {norm_sql('business_address', True)} AS address_norm,
           lower(trim(country)) AS country_norm
    FROM read_csv_auto('dataset/test/test_source1.tsv', delim='\\t', all_varchar=true, header=true)
""")
total_test_s1 = con.execute("SELECT count(*) FROM full_s1").fetchone()[0]

con.execute("""
    CREATE TABLE s1 AS
    SELECT * FROM (
        SELECT *, row_number() OVER (PARTITION BY country_norm ORDER BY hash(entity_id)) as rn,
                  count(*) OVER (PARTITION BY country_norm) as c_cnt
        FROM full_s1
    ) WHERE rn <= ceil(c_cnt * 0.10)
""")
sample_s1_count = con.execute("SELECT count(*) FROM s1").fetchone()[0]
con.execute("DROP TABLE full_s1")

print(f"   Test S1 Total: {total_test_s1:,} -> Sampled 10%: {sample_s1_count:,}")

# Load target directly without keeping s2/s3
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
print(f"   Full Test Target Total (s2 + s3): {total_target:,}")
print(f"   Data ingestion completed in {time.time()-t0:.2f}s")

# 2. Build staging tables
t_stage = time.time()
print("2. Building blocking staging tables...")

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

    # Clean numbers
    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_clean_num AS
        SELECT entity_id, country_norm,
               ltrim(regexp_extract(address_norm, '([0-9]{{1,6}})', 1), '0') as clean_num
        FROM {src}
        WHERE address_norm <> '' AND regexp_extract(address_norm, '([0-9]{{1,6}})', 1) <> ''
    """)

    # Rule 5A: Name tok2 num
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

    # Rule 4A: Folded p4 num
    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_name_folded AS
        SELECT a.entity_id, a.country_norm,
               substr(replace(a.name_norm, ' ', ''), 1, 4) as fold_p4,
               b.clean_num
        FROM {src} a
        JOIN {side}_clean_num b USING (entity_id, country_norm)
        WHERE b.clean_num <> '' AND length(a.name_norm) >= 4
    """)

    # Rule 1B: Multi num tok
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

    # Rule 9A: 3+ digit numbers
    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_addr_num_3plus AS
        SELECT DISTINCT entity_id, country_norm, ltrim(num, '0') as clean_num
        FROM (
          SELECT entity_id, country_norm, unnest(regexp_extract_all(address_norm, '[0-9]{{3,8}}')) as num
          FROM {src} WHERE address_norm <> ''
        )
        WHERE ltrim(num, '0') <> ''
    """)

    # Rule 9B: 2-token pairs (limit to top-5 distinct alphabetic tokens per entity to avoid combinatorial explosion)
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

# Target frequency tables
con.execute("CREATE OR REPLACE TABLE r_fname_tok2_num AS SELECT country_norm, name_tok2, clean_num, count(*) n FROM r_name_tok2_num GROUP BY 1, 2, 3")
con.execute("CREATE OR REPLACE TABLE r_ffold_p4_num AS SELECT country_norm, fold_p4, clean_num, count(*) n FROM r_name_folded GROUP BY 1, 2, 3")
con.execute("CREATE OR REPLACE TABLE r_fmulti_num_tok AS SELECT country_norm, clean_num, addr_tok1, count(*) n FROM r_multi_num_tok GROUP BY 1, 2, 3")
con.execute("CREATE OR REPLACE TABLE r_faddr_num_3plus AS SELECT country_norm, clean_num, count(*) n FROM r_addr_num_3plus GROUP BY 1, 2")
con.execute("CREATE OR REPLACE TABLE r_faddr_2tok AS SELECT country_norm, tok1, tok2, count(*) n FROM r_addr_2tok_pairs GROUP BY 1, 2, 3")

print(f"   Staging completed in {time.time()-t_stage:.2f}s")

# 3. Candidate generation
t_cand = time.time()
print("3. Running candidate blocking join with Rule 9 included...")

cand_sql = """
CREATE OR REPLACE TABLE test_candidates AS
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
print(f"   Candidate blocking join completed in {cand_duration:.2f}s")

# Measure results
cand_count_sample = con.execute("SELECT count(*) FROM test_candidates").fetchone()[0]
distinct_s1_sample = con.execute("SELECT count(DISTINCT source1_entity_id) FROM test_candidates").fetchone()[0]
avg_cands_sample = cand_count_sample / sample_s1_count

# Peak memory
mem_info = process.memory_info()
peak_mem_mb = mem_info.rss / (1024 * 1024)

# Extrapolations to Full Test Set (1,732,544 S1 entities)
scale_factor = total_test_s1 / sample_s1_count
extrapolated_cands = cand_count_sample * scale_factor
extrapolated_duration = (time.time() - t_start) * 1.5

print("\n" + "="*60)
print("TEST SET BLOCKING EVALUATION REPORT")
print("="*60)
print(f"Slice Evaluated:                  10.0% representative test slice ({sample_s1_count:,} / {total_test_s1:,} S1)")
print(f"Target Search Space:              100.0% full test target ({total_target:,} S2+S3 records)")
print(f"Total Slice Runtime:              {time.time()-t_start:.2f}s (Blocking query: {cand_duration:.2f}s)")
print(f"Slice Candidate Rows Produced:    {cand_count_sample:,}")
print(f"Slice S1 with Candidates:         {distinct_s1_sample:,} ({distinct_s1_sample/sample_s1_count*100:.2f}%)")
print(f"Average Candidates / Test S1:     {avg_cands_sample:.4f}")
print(f"Process Resident Memory (RSS):    {peak_mem_mb:.2f} MB")
print("-"*60)
print("EXTRAPOLATION TO FULL TEST SET (1,732,544 S1 entities):")
print(f"  * Status:                       EXTRAPOLATION from 10% slice against 100% target")
print(f"  * Extrapolated Candidate Rows:  {int(extrapolated_cands):,} candidate pairs (~{extrapolated_cands/1e6:.2f}M)")
print(f"  * Extrapolated Blocking Time:   ~{extrapolated_duration:.1f}s (~{extrapolated_duration/60:.1f} min)")
print(f"  * Extrapolated Avg Cands/S1:    {avg_cands_sample:.4f} (invariant to slice size)")
print("="*60)

con.close()
if os.path.exists(db_path):
    os.remove(db_path)
