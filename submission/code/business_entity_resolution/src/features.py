"""Fast pairwise lexical feature extraction without external lookups."""
import re
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s): return set(_TOK.findall(s or ""))


def _chargrams(s, n=3):
    s = f"  {s}  "
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}


def _sim(a, b):
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))


def _tfidf_cosine(a, b):
    """Smoothed pair-local TF-IDF cosine; avoids building a global dense matrix."""
    at, bt = _tokens(a), _tokens(b)
    vocab=at | bt
    if not vocab: return 0.0
    # With the two records as the local corpus, shared terms have lower IDF.
    idf={t: 1.0 + np.log(3.0/(1.0 + int(t in at) + int(t in bt))) for t in vocab}
    dot=sum(idf[t]**2 for t in at & bt)
    na=np.sqrt(sum(idf[t]**2 for t in at)); nb=np.sqrt(sum(idf[t]**2 for t in bt))
    return float(dot/(na*nb)) if na and nb else 0.0


def _edit(a, b):
    if not a or not b: return float(a == b)
    return Levenshtein.normalized_similarity(a,b)


def _text_features(a, b, prefix):
    at, bt = _tokens(a), _tokens(b)
    return {
        f"{prefix}_exact": int(bool(a) and a == b),
        f"{prefix}_char_jaccard": _sim(_chargrams(a), _chargrams(b)),
        f"{prefix}_token_jaccard": _sim(at, bt),
        f"{prefix}_token_containment": len(at & bt) / max(1, min(len(at), len(bt))),
        f"{prefix}_tfidf_cosine": _tfidf_cosine(a,b),
        f"{prefix}_number_overlap": _sim({t for t in at if any(c.isdigit() for c in t)}, {t for t in bt if any(c.isdigit() for c in t)}),
        f"{prefix}_edit": _edit(a, b),
        f"{prefix}_length_delta": abs(len(a)-len(b)) / max(1, max(len(a),len(b))),
        f"{prefix}_missing_a": int(not a), f"{prefix}_missing_b": int(not b),
    }


def make_features(candidates: pd.DataFrame, source1: pd.DataFrame, source2: pd.DataFrame,
                  source3: pd.DataFrame) -> pd.DataFrame:
    from .preprocessing import preprocess
    left = preprocess(source1).set_index("entity_id", drop=False)
    right = pd.concat([source2, source3], ignore_index=True)
    right = preprocess(right).set_index("entity_id", drop=False)
    rows = []
    for s1id, tid in candidates[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None):
        a, b = left.loc[s1id], right.loc[tid]
        nf = _text_features(a.business_name_normalized, b.business_name_normalized, "name")
        af = _text_features(a.business_address_normalized, b.business_address_normalized, "address")
        nf["country_exact"] = int(bool(a.country_normalized) and a.country_normalized == b.country_normalized)
        nf["combined_similarity"] = (nf["name_token_jaccard"] + af["address_token_jaccard"]) / 2
        rows.append({**nf, **af})
    return pd.DataFrame(rows, index=candidates.index).replace([np.inf, -np.inf], 0).fillna(0)


def feature_columns():
    return list(_text_features("", "", "name")) + list(_text_features("", "", "address")) + ["country_exact", "combined_similarity"]
