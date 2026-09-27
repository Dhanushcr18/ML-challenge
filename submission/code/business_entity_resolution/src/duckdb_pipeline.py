"""Disk-backed pipeline for the multi-million-row challenge files.

DuckDB stages the TSVs and candidate joins on disk. Only bounded feature batches and
a reproducible Source-1 training sample are held in Python memory.
"""
import logging
import time as _time
from pathlib import Path
import gc
import uuid
import duckdb
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from .config import TRAIN_DIR, TEST_DIR, OUTPUT_DIR, MODEL_DIR, RANDOM_STATE, THRESHOLDS
from .features import _text_features
from .evaluation import tune_threshold

LOG = logging.getLogger(__name__)
MEMORY_LIMIT = "6GB"
S1_TRAIN_SAMPLE = 30_000
FEATURE_BATCH = 40_000
INFERENCE_SOURCE_BATCH = 1_000
MAX_CANDIDATES_PER_ENTITY = 40
PRODUCTION_THRESHOLD = 0.97


def _norm(expr, address=False):
    x = f"lower(trim(regexp_replace(regexp_replace(coalesce({expr}, ''), '[^[:alnum:] ]', ' ', 'g'), '\\s+', ' ', 'g')))"
    if address:
        for a, b in (("street","st"),("road","rd"),("avenue","ave"),("boulevard","blvd"),("lane","ln"),("drive","dr"),("apartment","apt"),("suite","ste"),("highway","hwy")):
            x = f"regexp_replace({x}, '\\b{a}\\b', '{b}', 'g')"
    else:
        for a, b in (("corporation","corp"),("corporate","corp"),("private","pvt"),("limited","ltd")):
            x = f"regexp_replace({x}, '\\b{a}\\b', '{b}', 'g')"
    return x


def _quote(path): return "'" + str(path).replace("'", "''") + "'"


def _load_sources(con, folder, stem):
    for i in (1, 2, 3):
        path = folder / f"{stem}_source{i}.tsv"
        name = f"s{i}"
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT entity_id, business_name, business_address, country, {_norm('business_name')} AS name_norm, {_norm('business_address', True)} AS address_norm, lower(trim(country)) AS country_norm FROM read_csv_auto({_quote(path)}, delim='\\t', all_varchar=true, header=true)")
        LOG.info("Staged %s rows from %s", con.execute(f"select count(*) from {name}").fetchone()[0], path.name)
    con.execute("CREATE OR REPLACE TABLE target AS SELECT * FROM s2 UNION ALL SELECT * FROM s3")


def _candidate_table(con, source1="s1", target="target", out="candidates"):
    for side, src in (("l", source1), ("r", target)):
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

        # Staging for Rule 5A (R_name_tok2_num) and Rule 4A (R_folded_p4_num)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_clean_num AS
            SELECT entity_id, country_norm,
                   ltrim(regexp_extract(address_norm, '([0-9]{{1,6}})', 1), '0') as clean_num
            FROM {src}
            WHERE address_norm <> '' AND regexp_extract(address_norm, '([0-9]{{1,6}})', 1) <> ''
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_name_toks_raw AS
            SELECT entity_id, country_norm, unnest(string_split(name_norm, ' ')) as token
            FROM {src}
            WHERE name_norm <> ''
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_name_tok2_num AS
            SELECT a.entity_id, a.country_norm, a.token as name_tok2, b.clean_num
            FROM (
              SELECT entity_id, country_norm, token,
                     row_number() OVER (PARTITION BY entity_id ORDER BY length(token) DESC, token) as rn
              FROM {side}_name_toks_raw
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
        # Staging for Rule 1B (R_empty_num_tok1)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_all_nums AS
            SELECT entity_id, country_norm, unnest(regexp_extract_all(address_norm, '[0-9]{{1,6}}')) as num
            FROM {src}
            WHERE address_norm <> ''
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_distinct_nums AS
            SELECT DISTINCT entity_id, country_norm, ltrim(num, '0') as clean_num
            FROM {side}_all_nums
            WHERE ltrim(num, '0') <> ''
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_addr_toks_raw AS
            SELECT entity_id, country_norm, unnest(string_split(address_norm, ' ')) as token
            FROM {src}
            WHERE address_norm <> ''
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE {side}_multi_num_tok AS
            SELECT n.entity_id, n.country_norm, n.clean_num, t.token as addr_tok1
            FROM {side}_distinct_nums n
            JOIN (
              SELECT entity_id, country_norm, token,
                     row_number() OVER (PARTITION BY entity_id ORDER BY length(token) DESC, token) as rn
              FROM {side}_addr_toks_raw
              WHERE length(token) >= 5 AND token NOT IN ('st','rd','ave','blvd','ln','dr','apt','ste','hwy','near','opp','opposite','floor','plot','flat','shop','no','block','road','street','avenue','lane','drive','delhi','mumbai','kolkata','chennai','bangalore','hyderabad','pune','city','state','india','us','district','nagar','extn','colony')
            ) t ON t.entity_id = n.entity_id AND t.country_norm = n.country_norm AND t.rn = 1
        """)

    # Target frequency tables for the 3 validated rules
    con.execute("CREATE OR REPLACE TABLE r_fname_tok2_num AS SELECT country_norm, name_tok2, clean_num, count(*) n FROM r_name_tok2_num GROUP BY 1, 2, 3")
    con.execute("CREATE OR REPLACE TABLE r_ffold_p4_num AS SELECT country_norm, fold_p4, clean_num, count(*) n FROM r_name_folded GROUP BY 1, 2, 3")
    con.execute("CREATE OR REPLACE TABLE r_fmulti_num_tok AS SELECT country_norm, clean_num, addr_tok1, count(*) n FROM r_multi_num_tok GROUP BY 1, 2, 3")

    con.execute(f"""CREATE OR REPLACE TABLE {out} AS
      SELECT DISTINCT source1_entity_id, candidate_entity_id FROM (
        SELECT source1_entity_id, candidate_entity_id FROM (
          SELECT source1_entity_id, candidate_entity_id,
                 row_number() OVER (PARTITION BY source1_entity_id ORDER BY evidence DESC, candidate_entity_id) as rn
          FROM (
            SELECT source1_entity_id, candidate_entity_id, sum(evidence) as evidence FROM (
              -- 1. Exact Name + Country
              SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id, 10 evidence
              FROM {source1} a JOIN {target} b USING (country_norm, name_norm)
              JOIN lfn af ON af.country_norm=a.country_norm AND af.name_norm=a.name_norm
              JOIN rfn bf ON bf.country_norm=b.country_norm AND bf.name_norm=b.name_norm
              WHERE af.n <= 300 AND bf.n <= 300

              UNION ALL
              -- 2. Exact Name (country-agnostic)
              SELECT a.entity_id, b.entity_id, 9
              FROM {source1} a JOIN {target} b USING (name_norm)
              JOIN lgn af ON af.name_norm=a.name_norm
              JOIN rgn bf ON bf.name_norm=b.name_norm
              WHERE af.n <= 80 AND bf.n <= 80

              UNION ALL
              -- 3. Exact Address + Country
              SELECT a.entity_id, b.entity_id, 8
              FROM {source1} a JOIN {target} b USING (country_norm, address_norm)
              JOIN lfa af ON af.country_norm=a.country_norm AND af.address_norm=a.address_norm
              JOIN rfa bf ON bf.country_norm=b.country_norm AND bf.address_norm=b.address_norm
              WHERE af.n <= 200 AND bf.n <= 200

              UNION ALL
              -- 4. Exact Address (country-agnostic)
              SELECT a.entity_id, b.entity_id, 7
              FROM {source1} a JOIN {target} b USING (address_norm)
              JOIN lga af ON af.address_norm=a.address_norm
              JOIN rga bf ON bf.address_norm=b.address_norm
              WHERE af.n <= 80 AND bf.n <= 80

              UNION ALL
              -- 5. Combo: Name Prefix (3) + Address Prefix (4) within Country
              SELECT a.entity_id, b.entity_id, 6
              FROM {source1} a JOIN {target} b
                ON a.country_norm=b.country_norm
                AND substr(split_part(a.name_norm,' ',1),1,3) = substr(split_part(b.name_norm,' ',1),1,3)
                AND substr(split_part(a.address_norm,' ',1),1,4) = substr(split_part(b.address_norm,' ',1),1,4)
              JOIN lcombo af ON af.country_norm=a.country_norm AND af.np=substr(split_part(a.name_norm,' ',1),1,3) AND af.ap=substr(split_part(a.address_norm,' ',1),1,4)
              JOIN rcombo bf ON bf.country_norm=b.country_norm AND bf.np=substr(split_part(b.name_norm,' ',1),1,3) AND bf.ap=substr(split_part(b.address_norm,' ',1),1,4)
              WHERE af.n <= 200 AND bf.n <= 200

              UNION ALL
              -- 6. Combo Global (country-agnostic): Name Prefix (4) + Address Prefix (4)
              SELECT a.entity_id, b.entity_id, 5
              FROM {source1} a JOIN {target} b
                ON substr(split_part(a.name_norm,' ',1),1,4) = substr(split_part(b.name_norm,' ',1),1,4)
                AND substr(split_part(a.address_norm,' ',1),1,4) = substr(split_part(b.address_norm,' ',1),1,4)
              JOIN lcombo_g af ON af.np=substr(split_part(a.name_norm,' ',1),1,4) AND af.ap=substr(split_part(a.address_norm,' ',1),1,4)
              JOIN rcombo_g bf ON bf.np=substr(split_part(b.name_norm,' ',1),1,4) AND bf.ap=substr(split_part(b.address_norm,' ',1),1,4)
              WHERE af.n <= 100 AND bf.n <= 100

              UNION ALL
              -- 7. Name Prefix (4 chars) within Country
              SELECT a.entity_id, b.entity_id, 4
              FROM {source1} a JOIN {target} b
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
        WHERE rn <= {MAX_CANDIDATES_PER_ENTITY}

        UNION ALL
        -- Rule 5A: R_name_tok2_num (second name token >= 5 chars, freq <= 100 + clean address number)
        SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
        FROM l_name_tok2_num a
        JOIN r_name_tok2_num b USING (country_norm, name_tok2, clean_num)
        JOIN r_fname_tok2_num f USING (country_norm, name_tok2, clean_num)
        WHERE f.n <= 100

        UNION ALL
        -- Rule 4A: R_folded_p4_num (folded 4-char name prefix + clean address number, freq <= 100)
        SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
        FROM l_name_folded a
        JOIN r_name_folded b USING (country_norm, fold_p4, clean_num)
        JOIN r_ffold_p4_num f USING (country_norm, fold_p4, clean_num)
        WHERE f.n <= 100

        UNION ALL
        -- Rule 1B: R_empty_num_tok1 (empty target name + clean address number set overlap + addr_tok1, freq <= 50)
        SELECT a.entity_id source1_entity_id, b.entity_id candidate_entity_id
        FROM l_multi_num_tok a
        JOIN r_multi_num_tok b USING (country_norm, clean_num, addr_tok1)
        JOIN {target} tgt ON tgt.entity_id = b.entity_id
        JOIN r_fmulti_num_tok f USING (country_norm, clean_num, addr_tok1)
        WHERE (tgt.name_norm IS NULL OR tgt.name_norm = '')
          AND f.n <= 50
      ) all_cands""")
    count = con.execute(f"select count(*) from {out}").fetchone()[0]
    LOG.info("Disk-backed blocking produced %s candidate pairs", count)
    return count


def _feature_rows(con, pairs, s1="s1", target="target", after_source_id=None):
    cols = {r[1] for r in con.execute(f"PRAGMA table_info('{pairs}')").fetchall()}
    label_select = ", p.label" if "label" in cols else ""
    resume_filter = " WHERE p.source1_entity_id > ?" if after_source_id is not None else ""
    order = " ORDER BY p.source1_entity_id, p.candidate_entity_id" if after_source_id is not None else ""
    cur = con.cursor()
    query = f"""SELECT p.source1_entity_id, p.candidate_entity_id,
      a.name_norm an, b.name_norm bn, a.address_norm aa, b.address_norm ba, a.country_norm ac, b.country_norm bc{label_select}
      FROM {pairs} p JOIN {s1} a ON a.entity_id=p.source1_entity_id JOIN {target} b ON b.entity_id=p.candidate_entity_id{resume_filter}{order}"""
    return cur.execute(query, [after_source_id] if after_source_id is not None else [])


def _write_id_pairs(con, path, col_name, source1="s1", pairs="candidates"):
    """Write grouped IDs processed in source batches to avoid full-join OOM.

    The original single-query LEFT JOIN over all S1 x candidates pairs (up to
    50 M rows) caused DuckDB to OOM before streaming the first chunk.  We now
    iterate over Source-1 entity IDs in batches of INFERENCE_SOURCE_BATCH and
    run a bounded query per batch.
    """
    s1_ids = [r[0] for r in con.execute(
        f"SELECT entity_id FROM {source1} ORDER BY entity_id").fetchall()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"source1_entity_id\t{col_name}\n")
        for batch_start in range(0, len(s1_ids), INFERENCE_SOURCE_BATCH):
            batch = s1_ids[batch_start: batch_start + INFERENCE_SOURCE_BATCH]
            batch_df = pd.DataFrame({"entity_id": batch})
            con.register("_write_batch", batch_df)
            q_sql = (
                f"SELECT s.entity_id source1_entity_id, c.candidate_entity_id"
                f" FROM _write_batch s"
                f" LEFT JOIN {pairs} c ON c.source1_entity_id = s.entity_id"
                f" ORDER BY s.entity_id, c.candidate_entity_id"
            )
            rows = con.execute(q_sql).fetchall()
            con.unregister("_write_batch")
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


def _materialize_features(con, pairs, s1="s1", target="target", cap_negatives=False):
    has_label = "label" in {r[1] for r in con.execute(f"PRAGMA table_info('{pairs}')").fetchall()}
    query = _feature_rows(con, pairs, s1, target)
    X = []; ids = []; labels = []
    while True:
        df = query.fetch_df_chunk(FEATURE_BATCH)
        if df is None or df.empty:
            break
        for row in df.itertuples(index=False, name=None):
            nf = _text_features(row[2], row[3], "name")
            af = _text_features(row[4], row[5], "address")
            nf["country_exact"] = int(bool(row[6]) and row[6] == row[7])
            nf["combined_similarity"] = (nf["name_token_jaccard"] + af["address_token_jaccard"]) / 2
            X.append({**nf, **af})
            ids.append((row[0], row[1]))
            if has_label:
                labels.append(row[8])
        if len(X) % 100000 < FEATURE_BATCH:
            LOG.info("Built features for %d pairs", len(X))
    return pd.DataFrame(X).fillna(0), ids, (np.asarray(labels, dtype=np.int8) if labels else None)


def _set_temp(con, root):
    temp = root / "output" / "duckdb_tmp" / uuid.uuid4().hex
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory={_quote(temp)}")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")


SCORE_CUTS = (0.50, 0.70, 0.80, 0.90, 0.92, 0.95, 0.97, 0.99)


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _feature_frame(df, feature_columns):
    rows = []
    for row in df.itertuples(index=False, name=None):
        nf = _text_features(row[2], row[3], "name")
        af = _text_features(row[4], row[5], "address")
        nf["country_exact"] = int(bool(row[6]) and row[6] == row[7])
        nf["combined_similarity"] = (nf["name_token_jaccard"] + af["address_token_jaccard"]) / 2
        rows.append({**nf, **af})
    X = pd.DataFrame(rows).fillna(0)
    if feature_columns:
        X = X.reindex(columns=list(feature_columns), fill_value=0)
    return X


def _new_score_stats():
    return {"n": 0, "sum": 0.0, "min": None, "max": None, "cuts": {c: 0 for c in SCORE_CUTS}}


def _update_score_stats(stats, probs):
    if probs is None or len(probs) == 0:
        return
    stats["n"] += int(len(probs))
    stats["sum"] += float(np.sum(probs))
    pmin = float(np.min(probs))
    pmax = float(np.max(probs))
    stats["min"] = pmin if stats["min"] is None else min(stats["min"], pmin)
    stats["max"] = pmax if stats["max"] is None else max(stats["max"], pmax)
    for cut in SCORE_CUTS:
        stats["cuts"][cut] += int(np.sum(probs >= cut))


def _log_score_stats(stats, prefix="Score distribution"):
    n = stats["n"]
    mean = (stats["sum"] / n) if n else 0.0
    LOG.info("%s: n=%d min=%.6f max=%.6f mean=%.6f",
             prefix, n, stats["min"] or 0.0, stats["max"] or 0.0, mean)
    denom = max(1, n)
    for cut in SCORE_CUTS:
        LOG.info("  >= %.2f: %d (%.2f%%)", cut, stats["cuts"][cut], 100.0 * stats["cuts"][cut] / denom)




# Slow-batch threshold: if a batch of INFERENCE_SOURCE_BATCH entities takes longer
# than this, the next batch will be split into halves then quarters automatically.
_SLOW_BATCH_SECS = 120  # 2 minutes per batch is the warning level
_MIN_SPLIT_SIZE = 100   # never split below 100 S1 entities per sub-batch


def _score_one_batch(con, model, threshold, batch_ids, selected_table, feature_columns, stats, scores_handle):
    """Score one batch of S1 entity IDs and insert matches. Returns (candidates_scored, matched_count)."""
    batch_df = pd.DataFrame({"entity_id": [str(x) for x in batch_ids]})
    con.register("_score_batch", batch_df)
    q_sql = (
        "SELECT p.source1_entity_id, p.candidate_entity_id,"
        " a.name_norm an, b.name_norm bn, a.address_norm aa, b.address_norm ba,"
        " a.country_norm ac, b.country_norm bc"
        " FROM candidates p"
        " JOIN _score_batch sb ON CAST(sb.entity_id AS VARCHAR) = CAST(p.source1_entity_id AS VARCHAR)"
        " JOIN s1 a ON a.entity_id = p.source1_entity_id"
        " JOIN target b ON b.entity_id = p.candidate_entity_id"
    )
    df = con.execute(q_sql).df()
    con.unregister("_score_batch")
    matched = 0
    if not df.empty:
        X = _feature_frame(df, feature_columns)
        probs = model.predict_proba(X)[:, 1]
        _update_score_stats(stats, probs)
        if scores_handle is not None:
            for (s1id, cid), score in zip(
                df[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None), probs
            ):
                scores_handle.write(f"{s1id}\t{cid}\t{float(score):.8f}\n")
        keep_mask = np.asarray(probs >= threshold)
        kept = df.loc[keep_mask, ["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
        if not kept.empty:
            con.register("kept_df", kept)
            con.execute(
                f"INSERT OR IGNORE INTO {selected_table} "
                "SELECT CAST(source1_entity_id AS VARCHAR), CAST(candidate_entity_id AS VARCHAR) FROM kept_df"
            )
            con.unregister("kept_df")
            matched = len(kept)
    return len(df), matched


def _score_s1_ids(con, model, threshold, s1_ids, selected_table, feature_columns,
                  checkpoint_path=None, scores_handle=None):
    """Score candidates for an ordered list of Source-1 IDs and insert matches.

    Per-batch timing is logged for every batch.  If a batch exceeds _SLOW_BATCH_SECS,
    the next batch is automatically split into sub-batches of half size (down to
    _MIN_SPLIT_SIZE) so that a single dense S1 range cannot stall the whole run.

    Checkpoints are written atomically *after* each successful batch so that
    previously completed results are never lost on resume.
    """
    stats = _new_score_stats()
    total = 0
    total_s1 = len(s1_ids)
    batch_size = INFERENCE_SOURCE_BATCH  # may be halved adaptively for slow batches
    batch_num = 0
    run_start = _time.monotonic()

    batch_start = 0
    while batch_start < total_s1:
        batch_ids = s1_ids[batch_start: batch_start + batch_size]
        batch_num += 1
        t0 = _time.monotonic()
        candidates_in_batch, matched_in_batch = _score_one_batch(
            con, model, threshold, batch_ids, selected_table, feature_columns, stats, scores_handle
        )
        elapsed = _time.monotonic() - t0
        total += candidates_in_batch

        # Write checkpoint atomically AFTER successful batch commit
        if checkpoint_path is not None:
            _atomic_write_text(checkpoint_path, str(batch_ids[-1]))

        sel_so_far = con.execute(f"SELECT count(*) FROM {selected_table}").fetchone()[0]
        wall_elapsed = _time.monotonic() - run_start
        LOG.info(
            "Batch %d | S1 %s..%s (%d entities) | %d candidates | %d matched | "
            "batch_time=%.1fs | total_scored=%d | total_selected=%d | wall=%.0fs",
            batch_num,
            batch_ids[0], batch_ids[-1], len(batch_ids),
            candidates_in_batch, matched_in_batch,
            elapsed, total, sel_so_far, wall_elapsed,
        )

        # Adaptive slow-batch split: if this batch was slow, halve the batch size
        if elapsed > _SLOW_BATCH_SECS and batch_size > _MIN_SPLIT_SIZE:
            new_size = max(_MIN_SPLIT_SIZE, batch_size // 2)
            LOG.warning(
                "Batch took %.1fs (> %ds threshold); reducing batch size %d -> %d for subsequent batches",
                elapsed, _SLOW_BATCH_SECS, batch_size, new_size,
            )
            batch_size = new_size

        # Periodic DuckDB checkpoint to flush WAL (every 50 batches)
        if batch_num % 50 == 0:
            con.execute("CHECKPOINT")

        batch_start += len(batch_ids)

    con.execute("CHECKPOINT")
    return stats, total


def _score_and_export(con, model, best, test_candidate_count):
    """Score all candidate pairs and write submission files.

    BUG FIX: the original implementation used a single streaming query over the
    full candidates JOIN s1 JOIN target (up to 50 M rows).  DuckDB must
    materialise the entire join result before it can stream chunks, so on large
    test sets the very first fetch_df_chunk call raises OutOfMemoryException,
    the except-less loop silently exits, and `selected` stays empty.

    Fix: iterate over Source-1 entity IDs in batches of INFERENCE_SOURCE_BATCH,
    run a bounded per-batch query (at most INFERENCE_SOURCE_BATCH *
    MAX_CANDIDATES_PER_ENTITY rows), and score each batch independently.  This
    keeps per-batch memory proportional to INFERENCE_SOURCE_BATCH, not to the
    total dataset size.
    """
    candidate_path = OUTPUT_DIR / 'candidate_pairs.tsv'
    matching_path = OUTPUT_DIR / 'matching_results.tsv'
    checkpoint_path = OUTPUT_DIR / 'inference_checkpoint.txt'
    resume = checkpoint_path.exists() and 'selected' in {r[0] for r in con.execute('show tables').fetchall()}
    if not resume:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if candidate_path.exists():
            candidate_path.unlink()
        if matching_path.exists():
            matching_path.unlink()
    LOG.info("Writing candidate_pairs.tsv...")
    if not candidate_path.exists() or candidate_path.stat().st_size == 0:
        _write_id_pairs(con, candidate_path, "candidate_entity_ids", "s1", "candidates")
    if not resume:
        con.execute("CREATE OR REPLACE TABLE selected(source1_entity_id VARCHAR, candidate_entity_id VARCHAR, PRIMARY KEY(source1_entity_id, candidate_entity_id))")

    last_source_id = checkpoint_path.read_text(encoding='utf-8').strip() if resume else None
    if last_source_id:
        ckpt_ok = con.execute("SELECT count(*) FROM s1 WHERE entity_id = ?", [last_source_id]).fetchone()[0]
        if not ckpt_ok:
            LOG.warning("Checkpoint id %s is not in s1; starting from the beginning of remaining work without that resume point", last_source_id)
            last_source_id = None
    if last_source_id:
        s1_ids = [r[0] for r in con.execute(
            "SELECT entity_id FROM s1 WHERE entity_id > ? ORDER BY entity_id",
            [last_source_id]).fetchall()]
        LOG.info("Resuming from checkpoint after source_id=%s, %d S1 entities remaining", last_source_id, len(s1_ids))
    else:
        s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM s1 ORDER BY entity_id").fetchall()]

    feature_columns = best.get("feature_columns") or list(getattr(model, "feature_names_in_", []) or [])
    LOG.info("Scoring candidate pairs with XGBoost threshold %.3f over %d S1 entities in source-batches of %d...",
             best['threshold'], len(s1_ids), INFERENCE_SOURCE_BATCH)
    stats, total = _score_s1_ids(
        con, model, best["threshold"], s1_ids, "selected", feature_columns,
        checkpoint_path=checkpoint_path,
    )
    _log_score_stats(stats, prefix="Full-inference score distribution (this run)")
    con.execute("CHECKPOINT")

    match_count = con.execute("SELECT count(*) FROM selected").fetchone()[0]
    qualifying = stats["cuts"].get(0.92, 0)
    # Use the actual threshold bucket if present, else count from the nearest logged cut.
    for cut in SCORE_CUTS:
        if abs(cut - float(best["threshold"])) < 1e-9:
            qualifying = stats["cuts"][cut]
            break
    if match_count == 0 and qualifying > 0:
        raise RuntimeError(
            f"Refusing to write matching_results.tsv from an empty selected table: "
            f"{qualifying} scored pairs were >= threshold {best['threshold']} in this run"
        )

    LOG.info("Writing matching_results.tsv from selected (%d rows) on the same connection...", match_count)
    _write_id_pairs(con, matching_path, "matched_entity_ids", "s1", "selected")
    distinct_matches = con.execute("SELECT count(DISTINCT source1_entity_id) FROM selected").fetchone()[0]
    total_test_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    LOG.info("Wrote submission: %d test candidates, %d scored pairs, %d matched pairs across %d entities (%d singletons), threshold %.3f",
             test_candidate_count, total, match_count, distinct_matches, total_test_s1 - distinct_matches, best['threshold'])
    # Preserve checkpoint until validation passes per safety requirement
    LOG.info("Preserving inference checkpoint at %s until validation passes", checkpoint_path)


def _resume_from_training_tables(root, con):
    """Resume a stopped run after staged train candidates exist in DuckDB."""
    LOG.info("Resuming from staged training tables in output/pipeline.duckdb")
    Xtr, ids_tr, ytr = _materialize_features(con, "train_ids")
    Xv, ids_v, yv = _materialize_features(con, "valid_ids")
    if ytr is None or yv is None or len(set(ytr)) < 2:
        raise RuntimeError("Staged training tables do not contain both labels")
    model = XGBClassifier(n_estimators=350, max_depth=7, learning_rate=.05, min_child_weight=2, subsample=.85, colsample_bytree=.9, reg_lambda=1, objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_jobs=4, random_state=RANDOM_STATE)
    pos = max(1, int((ytr == 0).sum()) / max(1, int((ytr == 1).sum())))
    model.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, pos, 1.0))
    pv = model.predict_proba(Xv)[:, 1]
    best, table = tune_threshold(yv, pv, THRESHOLDS)
    LOG.info("Validation result=%s all_thresholds=%s", best, table)
    joblib.dump({"model": model, "threshold": best['threshold'], "feature_columns": list(Xtr.columns)}, MODEL_DIR / 'matcher.joblib', compress=3)
    del Xtr, Xv, ids_tr, ids_v, ytr, yv, pv; gc.collect()
    con.execute("DROP TABLE IF EXISTS sampled_s1; DROP TABLE IF EXISTS split_group; DROP TABLE IF EXISTS training; DROP TABLE IF EXISTS train_ids; DROP TABLE IF EXISTS valid_ids; DROP TABLE IF EXISTS labeled; DROP TABLE IF EXISTS candidates; DROP TABLE IF EXISTS s1; DROP TABLE IF EXISTS all_s1")
    _load_sources(con, TEST_DIR, "test")
    test_candidate_count = _candidate_table(con)
    _score_and_export(con, model, best, test_candidate_count)
    con.close()
    return best


def run_disk_backed_pipeline(root: Path):
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    db = OUTPUT_DIR / "pipeline.duckdb"
    if db.exists():
        check = duckdb.connect(str(db), read_only=True)
        tables = {r[0] for r in check.execute("SHOW TABLES").fetchall()}
        check.close()
        if {"all_s1", "candidates", "train_ids", "valid_ids", "target"}.issubset(tables):
            con = duckdb.connect(str(db))
            _set_temp(con, root)
            return _resume_from_training_tables(root, con)
        if {"s1", "s2", "s3", "target", "candidates", "selected"}.issubset(tables) and (MODEL_DIR / "matcher.joblib").exists():
            con = duckdb.connect(str(db))
            _set_temp(con, root)
            payload = joblib.load(MODEL_DIR / "matcher.joblib")
            best = {"threshold": PRODUCTION_THRESHOLD}
            candidate_count = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
            _score_and_export(con, payload["model"], best, candidate_count)
            con.close()
            return best
        if {"s1", "s2", "s3", "target", "candidates"}.issubset(tables) and (MODEL_DIR / "matcher.joblib").exists():
            con = duckdb.connect(str(db))
            _set_temp(con, root)
            payload = joblib.load(MODEL_DIR / "matcher.joblib")
            best = {"threshold": PRODUCTION_THRESHOLD}
            candidate_count = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
            _score_and_export(con, payload["model"], best, candidate_count)
            con.close()
            return best
    if db.exists():
        db.unlink()
    wal = Path(str(db) + ".wal")
    if wal.exists():
        wal.unlink()
    con = duckdb.connect(str(db))
    _set_temp(con, root)
    _load_sources(con, TRAIN_DIR, "train")
    con.execute("CREATE OR REPLACE TABLE truth AS SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) candidate_entity_id FROM read_csv_auto(?,delim='\\t',all_varchar=true,header=true)", [str(TRAIN_DIR / 'train_ground_truth.tsv')])
    con.execute(f"CREATE OR REPLACE TABLE sampled_s1 AS SELECT * FROM s1 ORDER BY hash(entity_id) LIMIT {S1_TRAIN_SAMPLE}")
    con.execute("CREATE OR REPLACE TABLE split_group AS SELECT entity_id, row_number() OVER(ORDER BY entity_id)%5 split FROM sampled_s1")
    con.execute("CREATE OR REPLACE TABLE all_s1 AS SELECT * FROM s1")
    con.execute("CREATE OR REPLACE TABLE s1 AS SELECT * FROM sampled_s1")
    n = _candidate_table(con)
    con.execute("CREATE OR REPLACE TABLE labeled AS SELECT c.*, CASE WHEN t.candidate_entity_id IS NULL THEN 0 ELSE 1 END AS label FROM candidates c LEFT JOIN truth t USING(source1_entity_id,candidate_entity_id)")
    candidate_positive = con.execute("SELECT count(*) FROM labeled WHERE label=1").fetchone()[0]
    sampled_truth = con.execute("SELECT count(*) FROM truth t JOIN sampled_s1 s ON s.entity_id=t.source1_entity_id").fetchone()[0]
    LOG.info("Training blocking recall on sampled entities: %d/%d = %.4f", candidate_positive, sampled_truth, candidate_positive / max(1, sampled_truth))
    con.execute("CREATE OR REPLACE TABLE training AS SELECT * EXCLUDE(rn) FROM (SELECT *, row_number() OVER(PARTITION BY source1_entity_id,label ORDER BY hash(candidate_entity_id)) rn FROM labeled) WHERE label=1 OR rn<=15")
    con.execute("CREATE OR REPLACE TABLE train_ids AS SELECT t.* FROM training t JOIN split_group g ON g.entity_id=t.source1_entity_id WHERE g.split<>0")
    con.execute("CREATE OR REPLACE TABLE valid_ids AS SELECT * EXCLUDE(rn) FROM (SELECT l.*, row_number() OVER(PARTITION BY source1_entity_id,label ORDER BY hash(candidate_entity_id)) rn FROM labeled l JOIN split_group g ON g.entity_id=l.source1_entity_id WHERE g.split=0) WHERE label=1 OR rn<=25")
    LOG.info("Sampled pair counts: training=%s validation=%s columns=%s", con.execute("select count(*) from train_ids").fetchone()[0], con.execute("select count(*) from valid_ids").fetchone()[0], [r[1] for r in con.execute("pragma table_info('train_ids')").fetchall()])
    Xtr, ids_tr, ytr = _materialize_features(con, "train_ids")
    Xv, ids_v, yv = _materialize_features(con, "valid_ids")
    model = XGBClassifier(n_estimators=350, max_depth=7, learning_rate=.05, min_child_weight=2, subsample=.85, colsample_bytree=.9, reg_lambda=1, objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_jobs=4, random_state=RANDOM_STATE)
    pos = max(1, int((ytr == 0).sum()) / max(1, int((ytr == 1).sum())))
    model.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, pos, 1.0))
    pv = model.predict_proba(Xv)[:, 1]
    best, table = tune_threshold(yv, pv, THRESHOLDS)
    LOG.info("Validation result=%s all_thresholds=%s", best, table)
    validation_ids = [r[0] for r in con.execute("SELECT entity_id FROM split_group WHERE split=0").fetchall()]
    pred_any = {sid: False for sid in validation_ids}
    for (sid, _), score in zip(ids_v, pv):
        if score >= best['threshold']:
            pred_any[sid] = True
    true_singletons = con.execute("SELECT count(*) FROM split_group g WHERE g.split=0 AND NOT EXISTS (SELECT 1 FROM truth t WHERE t.source1_entity_id=g.entity_id)").fetchone()[0]
    LOG.info("Validation singleton entities: true=%d predicted among candidates=%d", true_singletons, sum(not x for x in pred_any.values()))
    joblib.dump({"model": model, "threshold": best['threshold'], "feature_columns": list(Xtr.columns)}, MODEL_DIR / 'matcher.joblib', compress=3)
    del Xtr, Xv, ids_tr, ids_v, ytr, yv, pv; gc.collect()
    con.execute("DROP TABLE IF EXISTS sampled_s1; DROP TABLE IF EXISTS split_group; DROP TABLE IF EXISTS training; DROP TABLE IF EXISTS train_ids; DROP TABLE IF EXISTS valid_ids; DROP TABLE IF EXISTS labeled; DROP TABLE IF EXISTS candidates; DROP TABLE IF EXISTS s1; DROP TABLE IF EXISTS all_s1")
    # Re-stage test inputs, generate and score candidates in feature batches.
    _load_sources(con, TEST_DIR, "test")
    test_candidate_count = _candidate_table(con)
    _score_and_export(con, model, best, test_candidate_count)
    con.close()
    return best
