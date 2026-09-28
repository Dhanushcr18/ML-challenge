import sys
sys.path.insert(0, '.')
import time
import duckdb
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, precision_score, recall_score, fbeta_score
from src.features import _text_features

t0 = time.time()
print("="*60)
print("SCORING VAL_SPLIT0 WITH EXISTING CLASSIFIER (UNCHANGED MODEL)")
print("="*60)

con = duckdb.connect('tmp/experiment_blocking.duckdb', read_only=True)

# 1. Load candidates for val_s1 (both before and after)
print("1. Loading candidate pairs for val_s1 (6,000 entities)...")

# New candidate pool (Rule 9 included)
cands_after_df = con.execute("""
    SELECT c.source1_entity_id, c.candidate_entity_id,
           s.name_norm AS an, t.name_norm AS bn,
           s.address_norm AS aa, t.address_norm AS ba,
           s.country_norm AS ac, t.country_norm AS bc,
           (tr.source1_entity_id IS NOT NULL) AS is_true
    FROM full_train_candidates c
    JOIN val_s1 v ON c.source1_entity_id = v.entity_id
    JOIN s1 s ON c.source1_entity_id = s.entity_id
    JOIN target t ON c.candidate_entity_id = t.entity_id
    LEFT JOIN truth tr ON c.source1_entity_id = tr.source1_entity_id AND c.candidate_entity_id = tr.candidate_entity_id
""").df()

# Baseline candidate pool (before Rule 9)
cands_before_df = con.execute("""
    SELECT c.source1_entity_id, c.candidate_entity_id,
           s.name_norm AS an, t.name_norm AS bn,
           s.address_norm AS aa, t.address_norm AS ba,
           s.country_norm AS ac, t.country_norm AS bc,
           (tr.source1_entity_id IS NOT NULL) AS is_true
    FROM candidates_base c
    JOIN val_s1 v ON c.source1_entity_id = v.entity_id
    JOIN s1 s ON c.source1_entity_id = s.entity_id
    JOIN target t ON c.candidate_entity_id = t.entity_id
    LEFT JOIN truth tr ON c.source1_entity_id = tr.source1_entity_id AND c.candidate_entity_id = tr.candidate_entity_id
""").df()

total_gt = con.execute("SELECT count(*) FROM truth JOIN val_s1 ON truth.source1_entity_id = val_s1.entity_id").fetchone()[0]
con.close()

print(f"   Candidates BEFORE Rule 9: {len(cands_before_df):,} pairs")
print(f"   Candidates AFTER  Rule 9: {len(cands_after_df):,} pairs")
print(f"   Total Ground Truth Pairs: {total_gt:,}")

# 2. Load model
print("2. Loading existing classifier models/matcher.joblib...")
model_dict = joblib.load('models/matcher.joblib')
model = model_dict['model']
feature_cols = model_dict['feature_columns']
default_thresh = model_dict.get('threshold', 0.92)
print(f"   Model type: {type(model).__name__}")
print(f"   Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"   Internal default threshold: {default_thresh}")

# 3. Extract features
def compute_features(df):
    t_f = time.time()
    feats = []
    for row in df.itertuples(index=False):
        f = {}
        # Name features
        f.update(_text_features(row.an, row.bn, "name"))
        # Country exact
        f["country_exact"] = int(row.ac == row.bc)
        # Combined similarity
        f["combined_similarity"] = 0.5 * (f["name_char_jaccard"] + f.get("name_token_jaccard", 0.0))
        # Address features
        f.update(_text_features(row.aa, row.ba, "address"))
        feats.append(f)
    feat_df = pd.DataFrame(feats)[feature_cols]
    return feat_df

print("3. Extracting features for candidate pools...")
X_before = compute_features(cands_before_df)
X_after = compute_features(cands_after_df)

# 4. Score candidates
print("4. Predicting scores with existing classifier...")
cands_before_df['score'] = model.predict_proba(X_before)[:, 1]
cands_after_df['score'] = model.predict_proba(X_after)[:, 1]

# 5. Evaluate and generate classification reports
for name, df in [("BEFORE Rule 9 (166k pool)", cands_before_df), ("AFTER Rule 9 (385k pool)", cands_after_df)]:
    print("\n" + "="*70)
    print(f"EVALUATION: {name}")
    print("="*70)
    
    for thresh in [0.92, 0.970]:
        y_pred = (df['score'] >= thresh).astype(int)
        y_true = df['is_true'].astype(int)
        
        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & (1 - y_true)).sum())
        fn_cand = int(((1 - y_pred) & y_true).sum())
        fn_gt = int(total_gt - tp)
        
        cand_p = tp / max(1, tp + fp)
        cand_r = tp / max(1, tp + fn_cand)
        cand_f05 = (1 + 0.5**2) * cand_p * cand_r / max(1e-9, (0.5**2 * cand_p) + cand_r)
        
        gt_r = tp / total_gt
        gt_f05 = (1 + 0.5**2) * cand_p * gt_r / max(1e-9, (0.5**2 * cand_p) + gt_r)
        
        print(f"\n--- Threshold = {thresh:.3f} ---")
        print(f"Candidate-level Confusion: TP={tp:,}, FP={fp:,}, FN={fn_cand:,}")
        print(f"Ground-truth-level:        TP={tp:,}, FP={fp:,}, FN={fn_gt:,} (out of {total_gt:,} total GT)")
        print(f"Precision:                 {cand_p:.4f} ({cand_p*100:.2f}%)")
        print(f"Recall (vs Candidates):    {cand_r:.4f} ({cand_r*100:.2f}%)")
        print(f"Recall (vs Full GT):       {gt_r:.4f} ({gt_r*100:.2f}%)")
        print(f"Candidate F0.5:            {cand_f05:.4f}")
        print(f"End-to-End Ground-Truth F0.5: {gt_f05:.4f}")
        
        print("\nScikit-learn classification_report (on Candidate Pool):")
        print(classification_report(y_true, y_pred, target_names=["Non-Match", "Match"], digits=4))

print("\n" + "="*70)
print(f"Done in {time.time()-t0:.2f}s")
