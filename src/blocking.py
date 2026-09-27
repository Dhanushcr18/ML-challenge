"""Multi-pass inverted-index blocking with deterministic posting caps."""
from collections import defaultdict
import logging
import re
import pandas as pd
from .config import MAX_KEY_POSTINGS, MAX_CANDIDATES_PER_ENTITY
from .preprocessing import preprocess

LOG = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9]+")


def _keys(row):
    name, addr, country = row.business_name_normalized, row.business_address_normalized, row.country_normalized
    toks = _WORD.findall(name)
    atoks = _WORD.findall(addr)
    result = set()
    if not country: return result
    if name: result.add(("exact_name", country, name))
    if addr: result.add(("exact_addr", country, addr))
    for token in toks:
        if len(token) >= 2: result.add(("name_tok", country, token))
    if toks: result.add(("name_prefix", country, toks[0][:4]))
    for token in atoks:
        if len(token) >= 3: result.add(("addr_tok", country, token))
    if len(name) >= 3:
        for i in range(max(1, len(name)-2)):
            result.add(("ngram", country, name[i:i+3]))
    if toks and atoks:
        result.add(("combo", country, toks[0][:4], next((t for t in atoks if any(c.isdigit() for c in t)), atoks[0])))
    return result


def generate_candidates(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame,
                        max_per_entity: int = MAX_CANDIDATES_PER_ENTITY,
                        max_postings: int = MAX_KEY_POSTINGS) -> pd.DataFrame:
    s1, s2, s3 = (preprocess(x) for x in (source1, source2, source3))
    targets = pd.concat([s2, s3], ignore_index=True)
    index = defaultdict(list)
    for pos, row in enumerate(targets.itertuples(index=False)):
        for key in _keys(row):
            # Cap common keys to control memory/candidate explosion.
            bucket = index[key]
            if len(bucket) < max_postings: bucket.append(pos)
    pairs = set()
    target_ids = targets.entity_id.to_numpy()
    for row in s1.itertuples(index=False):
        found = {}
        for key in _keys(row):
            weight={"exact_name":8,"exact_addr":7,"combo":6,"name_prefix":5,"name_tok":4,"addr_tok":3,"ngram":2}.get(key[0],1)
            for pos in index.get(key, ()):
                found[pos]=found.get(pos,0)+weight
        # Keep the strongest multi-block evidence when an entity has many candidates.
        if len(found) > max_per_entity:
            found=sorted(found,key=lambda p:(-found[p],p))[:max_per_entity]
        pairs.update((row.entity_id, target_ids[p]) for p in found)
    result = pd.DataFrame(sorted(pairs), columns=["source1_entity_id", "candidate_entity_id"])
    LOG.info("Generated %s candidates from %s possible pairs (reduction %.6f)", len(result),
             len(s1) * len(targets), 1-len(result)/max(1, len(s1)*len(targets)))
    return result
