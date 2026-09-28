import duckdb, time

con = duckdb.connect('tmp/experiment_blocking.duckdb', read_only=True)
t0 = time.time()

query = '''
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
        (r9a.source1_entity_id IS NOT NULL) AS r_9a,
        (r9b.source1_entity_id IS NOT NULL) AS r_9b
    FROM labeled_rules lr
    LEFT JOIN (
        SELECT DISTINCT a.entity_id AS source1_entity_id, b.entity_id AS candidate_entity_id
        FROM l_addr_num_3plus a
        JOIN r_addr_num_3plus b USING (country_norm, clean_num)
        JOIN r_faddr_num_3plus rf USING (country_norm, clean_num)
        WHERE rf.n <= 80
    ) r9a ON lr.source1_entity_id = r9a.source1_entity_id AND lr.candidate_entity_id = r9a.candidate_entity_id
    LEFT JOIN (
        SELECT DISTINCT a.entity_id AS source1_entity_id, b.entity_id AS candidate_entity_id
        FROM l_addr_2tok_pairs a
        JOIN r_addr_2tok_pairs b USING (country_norm, tok1, tok2)
        JOIN r_faddr_2tok rf USING (country_norm, tok1, tok2)
        WHERE rf.n <= 50
    ) r9b ON lr.source1_entity_id = r9b.source1_entity_id AND lr.candidate_entity_id = r9b.candidate_entity_id
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

res = con.execute(query).df()
print("Full Training Set Blocking Recall Check (Apples-to-Apples):")
print(res.to_string(index=False))

tot_gt = res['total_ground_truth_pairs'].sum()
tot_before = res['before_blocked_pairs'].sum()
tot_after = res['after_blocked_pairs'].sum()
print(f"\nTotal Ground Truth Pairs: {tot_gt:,}")
print(f"Total Blocked Before:     {int(tot_before):,} ({tot_before/tot_gt*100:.4f}%)")
print(f"Total Blocked After:      {int(tot_after):,} ({tot_after/tot_gt*100:.4f}%)")
print(f"Incremental Recovered:    +{int(tot_after - tot_before):,}")
print(f"Done in {time.time()-t0:.2f}s.")
con.close()
