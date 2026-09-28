import sys, os, time
import duckdb
import pandas as pd

sys.path.insert(0, '.')

con = duckdb.connect('tmp/experiment_blocking.duckdb')

print("="*80)
print("1. BUILDING STAGING TABLES FOR RULE 9a & 9b")
print("="*80)

# Build staging tables for left (s1) and right (target)
staging_times = {}

for side, src in [("l", "s1"), ("r", "target")]:
    t0 = time.time()
    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_addr_num_3plus AS
        SELECT DISTINCT entity_id, country_norm, ltrim(num, '0') as clean_num
        FROM (
          SELECT entity_id, country_norm, unnest(regexp_extract_all(address_norm, '[0-9]{{3,8}}')) as num
          FROM {src}
          WHERE address_norm <> ''
        )
        WHERE ltrim(num, '0') <> ''
    """)
    t1 = time.time()
    cnt = con.execute(f"SELECT count(*) FROM {side}_addr_num_3plus").fetchone()[0]
    staging_times[f"{side}_addr_num_3plus"] = (cnt, t1 - t0)
    print(f"  {side}_addr_num_3plus: {cnt:,} rows created in {t1 - t0:.2f}s")

    t0 = time.time()
    # Distinct tokens per entity first to bound combinatorial pairs
    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_distinct_addr_toks AS
        SELECT DISTINCT entity_id, country_norm, token
        FROM (
          SELECT entity_id, country_norm, unnest(string_split(address_norm, ' ')) as token
          FROM {src}
          WHERE address_norm <> ''
        )
        WHERE length(token) >= 4 AND regexp_matches(token, '^[a-z]+$')
          AND token NOT IN ('street','road','avenue','lane','drive','apartment','suite','highway','roadway','near','opp','opposite','floor','plot','flat','shop','block','delhi','mumbai','kolkata','chennai','bangalore','hyderabad','pune','city','state','india','us','district','nagar','extn','colony')
    """)
    
    con.execute(f"""
        CREATE OR REPLACE TABLE {side}_addr_2tok_pairs AS
        SELECT a.entity_id, a.country_norm, a.token as tok1, b.token as tok2
        FROM {side}_distinct_addr_toks a
        JOIN {side}_distinct_addr_toks b 
          ON a.entity_id = b.entity_id AND a.country_norm = b.country_norm AND a.token < b.token
    """)
    t1 = time.time()
    cnt = con.execute(f"SELECT count(*) FROM {side}_addr_2tok_pairs").fetchone()[0]
    staging_times[f"{side}_addr_2tok_pairs"] = (cnt, t1 - t0)
    print(f"  {side}_addr_2tok_pairs: {cnt:,} rows created in {t1 - t0:.2f}s")

# Build target frequency tables
t0 = time.time()
con.execute("CREATE OR REPLACE TABLE r_faddr_num_3plus AS SELECT country_norm, clean_num, count(*) n FROM r_addr_num_3plus GROUP BY 1, 2")
con.execute("CREATE OR REPLACE TABLE r_faddr_2tok AS SELECT country_norm, tok1, tok2, count(*) n FROM r_addr_2tok_pairs GROUP BY 1, 2, 3")
print(f"  Target frequency tables created in {time.time()-t0:.2f}s")

print("\n" + "="*80)
print("2. RE-RUNNING THE 30 MISSED PAIRS FROM STEP 2")
print("="*80)

# Exact 30 pairs from Step 2
pairs_to_check = [
    # 20 India Pairs
    ("S1-735306980", "S2-321790693", "India"),
    ("S1-221372774", "S2-325888307", "India"),
    ("S1-81688360",  "S2-813716427", "India"),
    ("S1-770067609", "S2-138311142", "India"),
    ("S1-47297703",  "S2-727721217", "India"),
    ("S1-932603284", "S3-487989301", "India"),
    ("S1-30342402",  "S3-358665067", "India"),
    ("S1-839247418", "S3-479480465", "India"),
    ("S1-396989705", "S3-755200452", "India"),
    ("S1-225569858", "S3-492475419", "India"),
    ("S1-988884695", "S2-579929075", "India"),
    ("S1-790709500", "S3-629341955", "India"),
    ("S1-541353045", "S2-839220939", "India"),
    ("S1-927336112", "S3-26682161",  "India"),
    ("S1-280137467", "S2-584417759", "India"),
    ("S1-56740699",  "S3-282959405", "India"),
    ("S1-44074857",  "S3-983728611", "India"),
    ("S1-538024132", "S3-996268852", "India"),
    ("S1-846070758", "S2-693280067", "India"),
    ("S1-940940575", "S2-816490266", "India"),
    # 10 US Pairs
    ("S1-870795558", "S3-446471096", "US"),
    ("S1-999705251", "S3-164200749", "US"),
    ("S1-864173263", "S3-315922939", "US"),
    ("S1-678431767", "S3-397724962", "US"),
    ("S1-794199162", "S3-386251118", "US"),
    ("S1-915182780", "S3-832889280", "US"),
    ("S1-192032968", "S3-707893421", "US"),
    ("S1-585222429", "S3-916224900", "US"),
    ("S1-479723547", "S3-945148586", "US"),
    ("S1-986129293", "S3-79188579",  "US")
]

table_rows = []
for idx, (s1_id, t_id, cntry) in enumerate(pairs_to_check, 1):
    # Check Rule 9a (Numeric)
    r9a = con.execute(f"""
        SELECT 1 FROM l_addr_num_3plus a
        JOIN r_addr_num_3plus b USING (country_norm, clean_num)
        JOIN r_faddr_num_3plus rf USING (country_norm, clean_num)
        WHERE a.entity_id = '{s1_id}' AND b.entity_id = '{t_id}' AND rf.n <= 80
    """).fetchone()

    # Check Rule 9b (2-Token)
    r9b = con.execute(f"""
        SELECT 1 FROM l_addr_2tok_pairs a
        JOIN r_addr_2tok_pairs b USING (country_norm, tok1, tok2)
        JOIN r_faddr_2tok rf USING (country_norm, tok1, tok2)
        WHERE a.entity_id = '{s1_id}' AND b.entity_id = '{t_id}' AND rf.n <= 50
    """).fetchone()

    if r9a and r9b:
        status, rule = "YES", "Rule 9a (Num 3+) & 9b (2-Tok)"
    elif r9a:
        status, rule = "YES", "Rule 9a (Addr Num 3+)"
    elif r9b:
        status, rule = "YES", "Rule 9b (Addr 2-Tok)"
    else:
        status, rule = "NO", "None (Missed)"

    table_rows.append({
        "No": idx,
        "Country": cntry,
        "Source1 ID": s1_id,
        "Target ID": t_id,
        "Present in Candidates": status,
        "Caught By Rule": rule
    })

df_step2_eval = pd.DataFrame(table_rows)
print(df_step2_eval.to_string(index=False))

print("\n" + "="*80)
print("3. FULL TRAINING SET BLOCKING RECALL (BEFORE VS AFTER RULE 9)")
print("="*80)

t0 = time.time()
query_full = '''
WITH full_truth AS (
    SELECT 
        tr.source1_entity_id,
        tr.candidate_entity_id,
        s1.name_norm AS s1_name,
        s1.address_norm AS s1_addr,
        s1.country_norm AS s1_country,
        t.name_norm AS t_name,
        t.address_norm AS t_addr,
        t.country_norm AS t_country
    FROM truth tr
    JOIN s1 ON tr.source1_entity_id = s1.entity_id
    JOIN target t ON tr.candidate_entity_id = t.entity_id
),
labeled_rules AS (
    SELECT 
        ft.*,
        -- Existing Rules
        (ft.s1_country = ft.t_country AND ft.s1_name <> '' AND ft.s1_name = ft.t_name) AS r_name_exact,
        (ft.s1_country = ft.t_country AND ft.s1_addr <> '' AND ft.s1_addr = ft.t_addr) AS r_addr_exact,
        (ft.s1_name <> '' AND ft.s1_name = ft.t_name) AS r_name_global,
        (ft.s1_addr <> '' AND ft.s1_addr = ft.t_addr) AS r_addr_global,
        (ft.s1_country = ft.t_country AND length(ft.s1_name) >= 3 AND length(ft.s1_addr) >= 4 
         AND substr(split_part(ft.s1_name,' ',1),1,3) = substr(split_part(ft.t_name,' ',1),1,3)
         AND substr(split_part(ft.s1_addr,' ',1),1,4) = substr(split_part(ft.t_addr,' ',1),1,4)) AS r_combo,
        (length(ft.s1_name) >= 4 AND length(ft.s1_addr) >= 4 
         AND substr(split_part(ft.s1_name,' ',1),1,4) = substr(split_part(ft.t_name,' ',1),1,4)
         AND substr(split_part(ft.s1_addr,' ',1),1,4) = substr(split_part(ft.t_addr,' ',1),1,4)) AS r_combo_g,
        (ft.s1_country = ft.t_country 
         AND regexp_extract(ft.s1_name, '[a-z]{5,}', 0) <> '' 
         AND regexp_extract(ft.s1_name, '[a-z]{5,}', 0) = regexp_extract(ft.t_name, '[a-z]{5,}', 0)) AS r_rare_name,
        (ft.s1_country = ft.t_country 
         AND regexp_extract(ft.s1_name, '^[^ ]+ ([a-z]{4,})', 1) NOT IN ('','inc','llc','ltd','corp','pvt','company','limited','group','services')
         AND regexp_extract(ft.s1_name, '^[^ ]+ ([a-z]{4,})', 1) = regexp_extract(ft.t_name, '^[^ ]+ ([a-z]{4,})', 1)) AS r_second_word,
        (ft.s1_country = ft.t_country 
         AND regexp_extract(ft.s1_addr, '[a-z]{4,}', 0) <> '' 
         AND regexp_extract(ft.s1_addr, '[a-z]{4,}', 0) = regexp_extract(ft.t_addr, '[a-z]{4,}', 0)) AS r_addr_word,
        (ft.s1_country = ft.t_country
         AND ltrim(regexp_extract(ft.s1_addr, '([0-9]{1,6})', 1), '0') <> ''
         AND ltrim(regexp_extract(ft.s1_addr, '([0-9]{1,6})', 1), '0') = ltrim(regexp_extract(ft.t_addr, '([0-9]{1,6})', 1), '0')
         AND substr(replace(ft.s1_name, ' ', ''), 1, 4) = substr(replace(ft.t_name, ' ', ''), 1, 4)) AS r_rule_4a,
        (ft.s1_country = ft.t_country
         AND ft.t_name = '' AND ft.s1_addr <> '' AND ft.t_addr <> ''
         AND ltrim(regexp_extract(ft.s1_addr, '([0-9]{1,6})', 1), '0') <> ''
         AND ltrim(regexp_extract(ft.s1_addr, '([0-9]{1,6})', 1), '0') = ltrim(regexp_extract(ft.t_addr, '([0-9]{1,6})', 1), '0')) AS r_rule_1b
    FROM full_truth ft
),
joined_new_rules AS (
    SELECT 
        lr.*,
        (r9a.entity_id IS NOT NULL) AS r_9a,
        (r9b.entity_id IS NOT NULL) AS r_9b
    FROM labeled_rules lr
    LEFT JOIN (
        SELECT a.entity_id, b.entity_id AS target_id
        FROM l_addr_num_3plus a
        JOIN r_addr_num_3plus b USING (country_norm, clean_num)
        JOIN r_faddr_num_3plus rf USING (country_norm, clean_num)
        WHERE rf.n <= 80
    ) r9a ON lr.source1_entity_id = r9a.entity_id AND lr.candidate_entity_id = r9a.target_id
    LEFT JOIN (
        SELECT a.entity_id, b.entity_id AS target_id
        FROM l_addr_2tok_pairs a
        JOIN r_addr_2tok_pairs b USING (country_norm, tok1, tok2)
        JOIN r_faddr_2tok rf USING (country_norm, tok1, tok2)
        WHERE rf.n <= 50
    ) r9b ON lr.source1_entity_id = r9b.entity_id AND lr.candidate_entity_id = r9b.target_id
)
SELECT 
    s1_country AS country,
    count(*) AS total_ground_truth_pairs,
    sum(CASE WHEN (r_name_exact OR r_addr_exact OR r_name_global OR r_addr_global OR r_combo OR r_combo_g OR r_rare_name OR r_second_word OR r_addr_word OR r_rule_4a OR r_rule_1b) THEN 1 ELSE 0 END) AS before_blocked_pairs,
    round(sum(CASE WHEN (r_name_exact OR r_addr_exact OR r_name_global OR r_addr_global OR r_combo OR r_combo_g OR r_rare_name OR r_second_word OR r_addr_word OR r_rule_4a OR r_rule_1b) THEN 1 ELSE 0 END) * 100.0 / count(*), 4) AS before_recall_pct,
    sum(CASE WHEN (r_name_exact OR r_addr_exact OR r_name_global OR r_addr_global OR r_combo OR r_combo_g OR r_rare_name OR r_second_word OR r_addr_word OR r_rule_4a OR r_rule_1b OR r_9a OR r_9b) THEN 1 ELSE 0 END) AS after_blocked_pairs,
    round(sum(CASE WHEN (r_name_exact OR r_addr_exact OR r_name_global OR r_addr_global OR r_combo OR r_combo_g OR r_rare_name OR r_second_word OR r_addr_word OR r_rule_4a OR r_rule_1b OR r_9a OR r_9b) THEN 1 ELSE 0 END) * 100.0 / count(*), 4) AS after_recall_pct
FROM joined_new_rules
GROUP BY s1_country
ORDER BY s1_country
'''

res_full = con.execute(query_full).df()
print(res_full.to_string(index=False))

tot_gt = res_full['total_ground_truth_pairs'].sum()
tot_before = res_full['before_blocked_pairs'].sum()
tot_after = res_full['after_blocked_pairs'].sum()
print(f"\nOverall Before: {tot_before:,} / {tot_gt:,} ({tot_before/tot_gt*100:.4f}%)")
print(f"Overall After:  {tot_after:,} / {tot_gt:,} ({tot_after/tot_gt*100:.4f}%)")
print(f"Incremental True Pairs Recovered: +{int(tot_after - tot_before):,}")
print(f"Query execution time: {time.time()-t0:.2f}s")

con.close()
