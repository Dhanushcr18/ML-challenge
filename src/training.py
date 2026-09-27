"""Candidate-labeled training and source-entity holdout validation."""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from .config import RANDOM_STATE, VALIDATION_FRACTION, MAX_TRAIN_NEGATIVES_PER_ENTITY
from .features import make_features
from .evaluation import tune_threshold

LOG = logging.getLogger(__name__)


def parse_truth(truth):
    pairs=set()
    for sid, values in truth[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False, name=None):
        for tid in str(values).split(","):
            if tid: pairs.add((sid, tid))
    return pairs


def label_candidates(candidates, truth_pairs):
    labels=np.fromiter(((a,b) in truth_pairs for a,b in candidates[["source1_entity_id","candidate_entity_id"]].itertuples(index=False,name=None)), dtype=np.int8)
    return labels


def _balanced_candidates(candidates, labels):
    data=candidates.copy(); data["label"]=labels
    pos=data[data.label==1]
    neg=data[data.label==0]
    neg=neg.groupby("source1_entity_id", sort=False, group_keys=False).head(MAX_TRAIN_NEGATIVES_PER_ENTITY)
    return pd.concat([pos,neg],ignore_index=True)


def train_validate(candidates, labels, sources, output_dir):
    y=np.asarray(labels,dtype=np.int8)
    groups=candidates.source1_entity_id.to_numpy()
    splitter=GroupShuffleSplit(n_splits=1,test_size=VALIDATION_FRACTION,random_state=RANDOM_STATE)
    tr,va=next(splitter.split(candidates,y,groups))
    train_candidates=candidates.iloc[tr].copy(); train_candidates["label"]=y[tr]
    train_candidates=_balanced_candidates(train_candidates,train_candidates.label.to_numpy())
    validation_candidates=candidates.iloc[va]
    validation_labels=y[va]
    LOG.info("Training pairs=%d positives=%d negatives=%d; validation pairs=%d positives=%d",len(train_candidates),int(train_candidates.label.sum()),int((train_candidates.label==0).sum()),len(validation_candidates),int(validation_labels.sum()))
    train_features=make_features(train_candidates,sources[0],sources[1],sources[2])
    validation_features=make_features(validation_candidates,sources[0],sources[1],sources[2])
    model=XGBClassifier(n_estimators=400,max_depth=7,learning_rate=.05,min_child_weight=2,
                        subsample=.85,colsample_bytree=.9,reg_lambda=1.0,
                        objective="binary:logistic",eval_metric="logloss",
                        tree_method="hist",n_jobs=-1,random_state=RANDOM_STATE)
    pos_weight=max(1.0,float((train_candidates.label==0).sum())/max(1,int(train_candidates.label.sum())))
    weights=np.where(train_candidates.label.to_numpy()==1,pos_weight,1.0)
    model.fit(train_features,train_candidates.label.to_numpy(),sample_weight=weights)
    probs=model.predict_proba(validation_features)[:,1]
    best, table=tune_threshold(validation_labels,probs)
    LOG.info("Validation: %s thresholds=%s",best,table)
    # Report singleton behavior per held-out Source 1 entity, alongside pair metrics.
    val_s1=validation_candidates.source1_entity_id.to_numpy()
    pred=probs>=best["threshold"]
    true_by={sid:False for sid in pd.unique(val_s1)}
    pred_by=true_by.copy()
    for sid, truth_value, pred_value in zip(val_s1,validation_labels,pred):
        true_by[sid] = true_by[sid] or bool(truth_value)
        pred_by[sid] = pred_by[sid] or bool(pred_value)
    LOG.info("Validation singleton entities: true=%d predicted=%d",sum(not x for x in true_by.values()),sum(not x for x in pred_by.values()))
    all_candidates=candidates.copy(); all_candidates["label"]=y
    refit=_balanced_candidates(all_candidates,y)
    refit_features=make_features(refit,sources[0],sources[1],sources[2])
    weights=np.where(refit.label.to_numpy()==1,max(1.0,float((refit.label==0).sum())/max(1,int(refit.label.sum()))),1.0)
    model.fit(refit_features,refit.label.to_numpy(),sample_weight=weights)
    output_dir.mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump({"model":model,"threshold":best["threshold"],"feature_columns":list(refit_features.columns)},output_dir/"matcher.joblib",compress=3)
    return model,best,table
