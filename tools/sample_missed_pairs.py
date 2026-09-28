import sys, os, random
import duckdb
import pandas as pd

sys.path.insert(0, '.')
from src.preprocessing import normalize_name, normalize_address, _base

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

con = duckdb.connect('tmp/experiment_blocking.duckdb', read_only=True)

query = '''
WITH full_truth AS (
    SELECT 
        tr.source1_entity_id,
        tr.candidate_entity_id,
        s1.business_name AS s1_raw_name,
        s1.business_address AS s1_raw_addr,
        s1.country AS s1_raw_country,
        s1.name_norm AS s1_name,
        s1.address_norm AS s1_addr,
        s1.country_norm AS s1_country,
        t.business_name AS t_raw_name,
        t.business_address AS t_raw_addr,
        t.country AS t_raw_country,
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
)
SELECT *
FROM labeled_rules
WHERE NOT (r_name_exact OR r_addr_exact OR r_name_global OR r_addr_global OR r_combo OR r_combo_g OR r_rare_name OR r_second_word OR r_addr_word OR r_rule_4a OR r_rule_1b)
'''

print("Querying missed true pairs from DuckDB...")
df_missed = con.execute(query).df()
con.close()

print(f"Total missed true pairs extracted: {len(df_missed):,} (India: {(df_missed['s1_country']=='india').sum():,}, US: {(df_missed['s1_country']=='us').sum():,})")

# Sample 20 India and 10 US missed pairs
sample_in = df_missed[df_missed['s1_country'] == 'india'].sample(n=20, random_state=42).reset_index(drop=True)
sample_us = df_missed[df_missed['s1_country'] == 'us'].sample(n=10, random_state=42).reset_index(drop=True)

def print_pair(idx, row, country_label):
    print("=" * 80)
    print(f"[{country_label} PAIR #{idx+1}]")
    print(f"  Source1 ID: {row['source1_entity_id']}  <--->  Target ID: {row['candidate_entity_id']}")
    print("-" * 80)
    print("  RAW SOURCE1:")
    print(f"    business_name:    {repr(row['s1_raw_name'])}")
    print(f"    business_address: {repr(row['s1_raw_addr'])}")
    print(f"    country:          {repr(row['s1_raw_country'])}")
    print("  RAW TARGET (Source2/3):")
    print(f"    business_name:    {repr(row['t_raw_name'])}")
    print(f"    business_address: {repr(row['t_raw_addr'])}")
    print(f"    country:          {repr(row['t_raw_country'])}")
    print("-" * 80)
    print("  NORMALIZED (via src.preprocessing):")
    print(f"    S1 name_norm:     {repr(normalize_name(row['s1_raw_name']))}")
    print(f"    T  name_norm:     {repr(normalize_name(row['t_raw_name']))}")
    print(f"    S1 country_norm:  {repr(_base(row['s1_raw_country']))}")
    print(f"    T  country_norm:  {repr(_base(row['t_raw_country']))}")
    print()

print("\n" + "#" * 80)
print("INDIA MISSED TRUE MATCH PAIRS (20 EXAMPLES)")
print("#" * 80)
for i, r in sample_in.iterrows():
    print_pair(i, r, "INDIA")

print("\n" + "#" * 80)
print("US MISSED TRUE MATCH PAIRS (10 EXAMPLES)")
print("#" * 80)
for i, r in sample_us.iterrows():
    print_pair(i, r, "US")
