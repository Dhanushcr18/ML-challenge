import sys
sys.path.insert(0, '.')
import time
import logging
import duckdb
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
from src.features import _text_features
from src.evaluation import tune_threshold, f05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("step6_retrain")

t0 = time.time()
print("="*75)
print("STEP 6: RETRAIN ON POST-RULE-9 CANDIDATES & FINE-GRID THRESHOLD TUNING")
print("="*75)

con = duckdb.connect('tmp/experiment_blocking.duckdb', read_only=True)

# 1. Prepare Training & Validation Candidate Datasets
print("\n1. Preparing training & validation datasets from full_train_candidates...")
t_prep = time.time()

# Train split: split_group WHERE split != 0 (24,000 S1 entities)
# Keep all positives (label=1) and up to 15 negatives per entity (balanced sampling)
train_cands_df = con.execute("""
    WITH split_cands AS (
        SELECT c.source1_entity_id, c.candidate_entity_id,
               s.name_norm AS an, t.name_norm AS bn,
               s.address_norm AS aa, t.address_norm AS ba,
               s.country_norm AS ac, t.country_norm AS bc,
               CASE WHEN tr.source1_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
        FROM full_train_candidates c
        JOIN split_group g ON c.source1_entity_id = g.entity_id
        JOIN s1 s ON c.source1_entity_id = s.entity_id
        JOIN target t ON c.candidate_entity_id = t.entity_id
        LEFT JOIN truth tr ON c.source1_entity_id = tr.source1_entity_id AND c.candidate_entity_id = tr.candidate_entity_id
        WHERE g.split <> 0
    ),
    ranked_train AS (
        SELECT *,
               row_number() OVER (PARTITION BY source1_entity_id, label ORDER BY hash(candidate_entity_id)) as rn
        FROM split_cands
    )
    SELECT source1_entity_id, candidate_entity_id, an, bn, aa, ba, ac, bc, label
    FROM ranked_train
    WHERE label = 1 OR rn <= 15
""").df()

# Val split: split_group WHERE split == 0 (6,000 S1 entities, exactly val_s1)
# Keep ALL candidates to reflect real full-candidate evaluation
val_cands_df = con.execute("""
    SELECT c.source1_entity_id, c.candidate_entity_id,
           s.name_norm AS an, t.name_norm AS bn,
           s.address_norm AS aa, t.address_norm AS ba,
           s.country_norm AS ac, t.country_norm AS bc,
           CASE WHEN tr.source1_entity_id IS NOT NULL THEN 1 ELSE 0 END AS label
    FROM full_train_candidates c
    JOIN split_group g ON c.source1_entity_id = g.entity_id
    JOIN s1 s ON c.source1_entity_id = s.entity_id
    JOIN target t ON c.candidate_entity_id = t.entity_id
    LEFT JOIN truth tr ON c.source1_entity_id = tr.source1_entity_id AND c.candidate_entity_id = tr.candidate_entity_id
    WHERE g.split = 0
""").df()

total_gt = con.execute("SELECT count(*) FROM truth JOIN split_group g ON truth.source1_entity_id = g.entity_id WHERE g.split = 0").fetchone()[0]
con.close()

n_train = len(train_cands_df)
n_train_pos = int(train_cands_df['label'].sum())
n_train_neg = n_train - n_train_pos

n_val = len(val_cands_df)
n_val_pos = int(val_cands_df['label'].sum())
n_val_neg = n_val - n_val_pos

print(f"   Training Pairs:   {n_train:,} (Positives={n_train_pos:,}, Negatives={n_train_neg:,}, Ratio={n_train_neg/n_train_pos:.2f}:1)")
print(f"   Validation Pairs: {n_val:,} (Positives={n_val_pos:,}, Negatives={n_val_neg:,})")
print(f"   Held-out Total Ground Truth: {total_gt:,} true pairs across 6,000 S1 entities")
print(f"   Dataset prepared in {time.time()-t_prep:.2f}s")

# 2. Extract Features
feature_cols = [
    'name_exact', 'name_char_jaccard', 'name_token_jaccard', 'name_token_containment',
    'name_tfidf_cosine', 'name_number_overlap', 'name_edit', 'name_length_delta',
    'name_missing_a', 'name_missing_b', 'country_exact', 'combined_similarity',
    'address_exact', 'address_char_jaccard', 'address_token_jaccard', 'address_token_containment',
    'address_tfidf_cosine', 'address_number_overlap', 'address_edit', 'address_length_delta',
    'address_missing_a', 'address_missing_b'
]

def featurize_df(df):
    rows = []
    for row in df.itertuples(index=False):
        nf = _text_features(row.an, row.bn, "name")
        af = _text_features(row.aa, row.ba, "address")
        nf["country_exact"] = int(bool(row.ac) and row.ac == row.bc)
        nf["combined_similarity"] = 0.5 * (nf["name_char_jaccard"] + nf.get("name_token_jaccard", 0.0))
        rows.append({**nf, **af})
    return pd.DataFrame(rows)[feature_cols]

print("\n2. Featurizing training and validation sets...")
t_feat = time.time()
X_train = featurize_df(train_cands_df)
y_train = train_cands_df['label'].to_numpy(dtype=np.int8)

X_val = featurize_df(val_cands_df)
y_val = val_cands_df['label'].to_numpy(dtype=np.int8)
print(f"   Featurization completed in {time.time()-t_feat:.2f}s")

# 3. Retrain Classifier
print("\n3. Training XGBClassifier on post-Rule-9 training candidate pool...")
t_train = time.time()
pos_weight = float(n_train_neg) / max(1, n_train_pos)
sample_weights = np.where(y_train == 1, pos_weight, 1.0)

model = XGBClassifier(
    n_estimators=350,
    max_depth=7,
    learning_rate=0.05,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)
train_duration = time.time() - t_train
print(f"   Model fitting completed in {train_duration:.2f}s")

# Save retrained model
joblib.dump({
    "model": model,
    "feature_columns": feature_cols
}, "tmp/retrained_rule9_model.joblib", compress=3)
print("   Saved retrained model to tmp/retrained_rule9_model.joblib")

# 4. Predict probabilities on validation split
print("\n4. Predicting on held-out validation split...")
p_val = model.predict_proba(X_val)[:, 1]

# 5. Fine-Grid Threshold Sweep (0.80 to 0.99 in steps of 0.01)
print("\n5. Running tune_threshold on fine grid (0.80 to 0.99, step=0.01)...")
fine_grid = [round(x, 2) for x in np.arange(0.80, 0.995, 0.01)]

table_rows = []
for t in fine_grid:
    pred = p_val >= t
    tp = int((pred & (y_val == 1)).sum())
    fp = int((pred & (y_val == 0)).sum())
    fn_cand = int((~pred & (y_val == 1)).sum())
    
    p = tp / max(1, tp + fp)
    r_cand = tp / max(1, tp + fn_cand)
    f05_cand = f05(y_val, pred)
    
    r_gt = tp / total_gt
    f05_gt = (1 + 0.5**2) * p * r_gt / max(1e-9, (0.5**2 * p) + r_gt)
    
    table_rows.append({
        "threshold": t,
        "precision": p,
        "recall_cand": r_cand,
        "f0.5_cand": f05_cand,
        "recall_gt": r_gt,
        "f0.5_gt": f05_gt,
        "tp": tp,
        "fp": fp
    })

tuning_df = pd.DataFrame(table_rows)

print("\n" + "="*85)
print("THRESHOLD TUNING GRID TABLE (FINE GRID 0.80 - 0.99):")
print("="*85)
print(tuning_df.to_string(index=False, formatters={
    "threshold": "{:.2f}".format,
    "precision": "{:.4f}".format,
    "recall_cand": "{:.4f}".format,
    "f0.5_cand": "{:.4f}".format,
    "recall_gt": "{:.4f}".format,
    "f0.5_gt": "{:.4f}".format,
    "tp": "{:,}".format,
    "fp": "{:,}".format
}))

best_cand_row = tuning_df.loc[tuning_df['f0.5_cand'].idxmax()]
best_gt_row = tuning_df.loc[tuning_df['f0.5_gt'].idxmax()]

print("\n" + "-"*85)
print(f"Best Candidate F0.5:   Threshold={best_cand_row['threshold']:.2f} | Cand F0.5={best_cand_row['f0.5_cand']:.4f} | GT F0.5={best_cand_row['f0.5_gt']:.4f}")
print(f"Best End-to-End GT F0.5: Threshold={best_gt_row['threshold']:.2f} | GT F0.5={best_gt_row['f0.5_gt']:.4f} | Precision={best_gt_row['precision']:.4f} | GT Recall={best_gt_row['recall_gt']:.4f}")
print("="*85)

# 6. Final Evaluation at Tuned Optimal Threshold
opt_thresh = float(best_gt_row['threshold'])
y_pred_opt = (p_val >= opt_thresh).astype(int)

tp_opt = int((y_pred_opt & y_val).sum())
fp_opt = int((y_pred_opt & (1 - y_val)).sum())
fn_cand_opt = int(((1 - y_pred_opt) & y_val).sum())
fn_gt_opt = int(total_gt - tp_opt)

cand_p_opt = tp_opt / max(1, tp_opt + fp_opt)
cand_r_opt = tp_opt / max(1, tp_opt + fn_cand_opt)
cand_f05_opt = (1 + 0.5**2) * cand_p_opt * cand_r_opt / max(1e-9, (0.5**2 * cand_p_opt) + cand_r_opt)

gt_r_opt = tp_opt / total_gt
gt_f05_opt = (1 + 0.5**2) * cand_p_opt * gt_r_opt / max(1e-9, (0.5**2 * cand_p_opt) + gt_r_opt)

print("\n" + "="*75)
print(f"FINAL METRICS AT NEWLY-TUNED OPTIMAL THRESHOLD (Threshold = {opt_thresh:.2f})")
print("="*75)
print(f"Candidate-level Confusion: TP={tp_opt:,}, FP={fp_opt:,}, FN={fn_cand_opt:,}")
print(f"Ground-truth-level:        TP={tp_opt:,}, FP={fp_opt:,}, FN={fn_gt_opt:,} (out of {total_gt:,} total GT)")
print(f"Precision:                 {cand_p_opt:.4f} ({cand_p_opt*100:.2f}%)")
print(f"Recall (vs Candidates):    {cand_r_opt:.4f} ({cand_r_opt*100:.2f}%)")
print(f"Recall (vs Full GT):       {gt_r_opt:.4f} ({gt_r_opt*100:.2f}%)")
print(f"Candidate F0.5:            {cand_f05_opt:.4f}")
print(f"End-to-End Ground-Truth F0.5: {gt_f05_opt:.4f}")
print("\nScikit-learn classification_report (on Candidate Pool):")
print(classification_report(y_val, y_pred_opt, target_names=["Non-Match", "Match"], digits=4))

# Also print comparison with baseline 0.970 on retrained model
print("\n" + "-"*75)
print("METRICS AT STANDARD PRODUCTION THRESHOLD (Threshold = 0.97)")
print("-"*75)
y_pred_97 = (p_val >= 0.97).astype(int)
tp_97 = int((y_pred_97 & y_val).sum())
fp_97 = int((y_pred_97 & (1 - y_val)).sum())
cand_p_97 = tp_97 / max(1, tp_97 + fp_97)
gt_r_97 = tp_97 / total_gt
gt_f05_97 = (1 + 0.5**2) * cand_p_97 * gt_r_97 / max(1e-9, (0.5**2 * cand_p_97) + gt_r_97)

print(f"At Threshold 0.97: TP={tp_97:,}, FP={fp_97:,}, Precision={cand_p_97:.4f}, GT Recall={gt_r_97:.4f}, GT F0.5={gt_f05_97:.4f}")
print("="*75)
print(f"Step 6 completed in {time.time()-t0:.2f}s")
