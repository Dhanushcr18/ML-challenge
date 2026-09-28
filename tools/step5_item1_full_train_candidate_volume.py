import duckdb
import time

con = duckdb.connect('tmp/experiment_blocking.duckdb')
t0 = time.time()

cand_query = """
CREATE OR REPLACE TABLE full_train_candidates AS
SELECT DISTINCT source1_entity_id, candidate_entity_id FROM (
  SELECT source1_entity_id, candidate_entity_id FROM (
    SELECT source1_entity_id, candidate_entity_id,
           row_number() OVER (PARTITION BY source1_entity_id ORDER BY evidence DESC, candidate_entity_id) as rn
    FROM (
      SELECT source1_entity_id, candidate_entity_id, sum(evidence) as evidence FROM (
        -- 1. Exact Name + Country
        SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id, 10 evidence
        FROM s1 a JOIN target b USING (country_norm, name_norm)
        JOIN full_lfn af ON af.country_norm=a.country_norm AND af.name_norm=a.name_norm
        JOIN rfn bf ON bf.country_norm=b.country_norm AND bf.name_norm=b.name_norm
        WHERE af.n <= 300 AND bf.n <= 300

        UNION ALL
        -- 2. Exact Name (global)
        SELECT a.entity_id, b.entity_id, 9
        FROM s1 a JOIN target b USING (name_norm)
        JOIN full_lgn af ON af.name_norm=a.name_norm
        JOIN rgn bf ON bf.name_norm=b.name_norm
        WHERE af.n <= 80 AND bf.n <= 80

        UNION ALL
        -- 3. Exact Address + Country
        SELECT a.entity_id, b.entity_id, 8
        FROM s1 a JOIN target b USING (country_norm, address_norm)
        JOIN full_lfa af ON af.country_norm=a.country_norm AND af.address_norm=a.address_norm
        JOIN rfa bf ON bf.country_norm=b.country_norm AND bf.address_norm=b.address_norm
        WHERE af.n <= 200 AND bf.n <= 200

        UNION ALL
        -- 4. Exact Address (global)
        SELECT a.entity_id, b.entity_id, 7
        FROM s1 a JOIN target b USING (address_norm)
        JOIN full_lga af ON af.address_norm=a.address_norm
        JOIN rga bf ON bf.address_norm=b.address_norm
        WHERE af.n <= 80 AND bf.n <= 80

        UNION ALL
        -- 5. Combo: Name Prefix (3) + Address Prefix (4) within Country
        SELECT a.entity_id, b.entity_id, 6
        FROM s1 a JOIN target b
          ON a.country_norm=b.country_norm
          AND substr(split_part(a.name_norm,' ',1),1,3) = substr(split_part(b.name_norm,' ',1),1,3)
          AND substr(split_part(a.address_norm,' ',1),1,4) = substr(split_part(b.address_norm,' ',1),1,4)
        JOIN full_lcombo af ON af.country_norm=a.country_norm AND af.np=substr(split_part(a.name_norm,' ',1),1,3) AND af.ap=substr(split_part(a.address_norm,' ',1),1,4)
        JOIN rcombo bf ON bf.country_norm=b.country_norm AND bf.np=substr(split_part(b.name_norm,' ',1),1,3) AND bf.ap=substr(split_part(b.address_norm,' ',1),1,4)
        WHERE af.n <= 200 AND bf.n <= 200

        UNION ALL
        -- 6. Combo Global (country-agnostic)
        SELECT a.entity_id, b.entity_id, 5
        FROM s1 a JOIN target b
          ON substr(split_part(a.name_norm,' ',1),1,4) = substr(split_part(b.name_norm,' ',1),1,4)
          AND substr(split_part(a.address_norm,' ',1),1,4) = substr(split_part(b.address_norm,' ',1),1,4)
        JOIN full_lcombo_g af ON af.np=substr(split_part(a.name_norm,' ',1),1,4) AND af.ap=substr(split_part(a.address_norm,' ',1),1,4)
        JOIN rcombo_g bf ON bf.np=substr(split_part(b.name_norm,' ',1),1,4) AND bf.ap=substr(split_part(b.address_norm,' ',1),1,4)
        WHERE af.n <= 100 AND bf.n <= 100

        UNION ALL
        -- 7. Name Prefix (4 chars) within Country
        SELECT a.entity_id, b.entity_id, 4
        FROM s1 a JOIN target b
          ON a.country_norm=b.country_norm AND substr(split_part(a.name_norm,' ',1),1,4) = substr(split_part(b.name_norm,' ',1),1,4)
        JOIN full_lfp af ON af.country_norm=a.country_norm AND af.prefix=substr(split_part(a.name_norm,' ',1),1,4)
        JOIN rfp bf ON bf.country_norm=b.country_norm AND bf.prefix=substr(split_part(b.name_norm,' ',1),1,4)
        WHERE af.n <= 120 AND bf.n <= 120

        UNION ALL
        -- 8. First 5-letter name token within Country
        SELECT a.entity_id, b.entity_id, 4
        FROM full_lnw a JOIN rnw b USING (country_norm, token)
        JOIN full_lnf af USING (country_norm, token)
        JOIN rnf bf USING (country_norm, token)
        WHERE af.n <= 250 AND bf.n <= 250

        UNION ALL
        -- 9. Second 4-letter name token within Country
        SELECT a.entity_id, b.entity_id, 3
        FROM full_ln2 a JOIN rn2 b USING (country_norm, token)
        JOIN full_ln2f af USING (country_norm, token)
        JOIN rn2f bf USING (country_norm, token)
        WHERE af.n <= 250 AND bf.n <= 250

        UNION ALL
        -- 10. Address 4-letter token within Country
        SELECT a.entity_id, b.entity_id, 2
        FROM full_law a JOIN raw b USING (country_norm, token)
        JOIN full_lafw af USING (country_norm, token)
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
  FROM full_l_name_tok2_num a
  JOIN t_name_tok2_num b USING (country_norm, name_tok2, clean_num)
  JOIN t_fname_tok2_num f USING (country_norm, name_tok2, clean_num)
  WHERE f.n <= 100

  UNION ALL
  -- Rule 4A: R_folded_p4_num
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM full_l_name_folded a
  JOIN t_name_folded b USING (country_norm, fold_p4, clean_num)
  JOIN t_ffold_p4_num f USING (country_norm, fold_p4, clean_num)
  WHERE f.n <= 100

  UNION ALL
  -- Rule 1B: R_empty_num_tok1
  SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
  FROM full_l_multi_num_tok a
  JOIN t_multi_num_tok b USING (country_norm, clean_num, addr_tok1)
  JOIN target tgt ON tgt.entity_id = b.entity_id
  JOIN t_fmulti_num_tok f USING (country_norm, clean_num, addr_tok1)
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

print("Executing candidate generation query on full training set (s1: 2,206,821 rows)...")
t_cand = time.time()
con.execute(cand_query)
print(f"Candidate table created in {time.time()-t_cand:.2f}s")

# Compute stats
total_cands = con.execute("SELECT count(*) FROM full_train_candidates").fetchone()[0]
distinct_s1 = con.execute("SELECT count(DISTINCT source1_entity_id) FROM full_train_candidates").fetchone()[0]
total_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]

avg_per_s1_total = total_cands / total_s1
avg_per_s1_active = total_cands / distinct_s1

print("\n" + "="*60)
print("FULL-SCALE TRAINING CANDIDATE VOLUME METRICS (POST-RULE 9)")
print("="*60)
print(f"Total S1 Entities (Full Train):   {total_s1:,}")
print(f"Total Candidate Rows:             {total_cands:,}")
print(f"Distinct S1 Entities with Cands:  {distinct_s1:,} ({distinct_s1/total_s1*100:.2f}%)")
print(f"Average Candidates per Total S1:  {avg_per_s1_total:.4f}")
print(f"Average Candidates per Active S1: {avg_per_s1_active:.4f}")
print("="*60)
print(f"Total Elapsed Time: {time.time()-t0:.2f}s")
con.close()
