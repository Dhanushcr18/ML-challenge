"""Apply a trained model and write submission artifacts."""
import logging
import pandas as pd
from .features import make_features

LOG=logging.getLogger(__name__)


def predict_candidates(candidates, sources, model, threshold, batch_size=50_000):
    pieces=[]
    for start in range(0,len(candidates),batch_size):
        batch=candidates.iloc[start:start+batch_size]
        features=make_features(batch,*sources)
        scores=model.predict_proba(features)[:,1]
        pieces.append(pd.DataFrame({"source1_entity_id":batch.source1_entity_id.to_numpy(),"candidate_entity_id":batch.candidate_entity_id.to_numpy(),"confidence":scores}))
        LOG.info("Scored %d/%d candidate pairs",min(start+batch_size,len(candidates)),len(candidates))
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=["source1_entity_id","candidate_entity_id","confidence"])


def write_outputs(source1, candidate_ids, scored, output_dir, threshold):
    output_dir.mkdir(parents=True,exist_ok=True)
    matched=scored[scored.confidence >= threshold]
    by_id=matched.groupby("source1_entity_id").candidate_entity_id.apply(lambda x: ",".join(sorted(set(x))))
    rows=pd.DataFrame({"source1_entity_id":source1.entity_id.astype(str)})
    rows["matched_entity_ids"]=rows.source1_entity_id.map(by_id).fillna("")
    rows.to_csv(output_dir/"matching_results.tsv",sep="\t",index=False)
    # Save exactly the candidate universe sent to the model, including zero-candidate entities.
    grouped=candidate_ids.groupby("source1_entity_id").candidate_entity_id.apply(lambda x: ",".join(sorted(set(x))))
    pairs=pd.DataFrame({"source1_entity_id":source1.entity_id.astype(str)})
    pairs["candidate_entity_ids"]=pairs.source1_entity_id.map(grouped).fillna("")
    pairs.to_csv(output_dir/"candidate_pairs.tsv",sep="\t",index=False)
