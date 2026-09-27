# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Entity Resolution Pioneers  
**Team Members:** Participant  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present an enterprise-scale, memory-efficient Business Entity Resolution system designed to match noisy records from Source 2 and Source 3 against a deduplicated reference Source 1. Our approach combines a high-precision 31-rule multi-blocking engine with an Apache-2.0 licensed XGBoost gradient-boosted decision tree classifier, optimized specifically for the precision-oriented Macro $F_{0.5}$ metric. The system processes over 11.7 million records in batches under strict memory bounds using DuckDB, achieving a validated Macro $F_{0.5}$ of 0.8867 (Macro Precision 0.9401, Macro Recall 0.7867) on held-out validation data.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across training and test sets, we identified several structural noise patterns:
- **Missing / Empty Fields:** Numerous Source 2 and Source 3 entities lack business names or display landmark-only addresses.
- **Address Perturbations:** Variations in street numbering (e.g., OCR digit flips, transpositions), abbreviations (St vs. Street, Rd vs. Road), landmark references ("Near SBI ATM"), and missing postal codes.
- **Name Perturbations:** Legal suffix variants (Corp, Corporation, Pvt Ltd, LLC), phonetic spelling differences, multi-token order transpositions, and character-level typos.
- **Cross-Source Asymmetry:** Source 1 is deduplicated, while Source 2 and Source 3 contribute partial fragments. Country is an open string field (including US, India, and test-only France).

### 2.2 Solution Strategy
**Approach Type:** Multi-Rule High-Precision Blocking + Gradient Boosted Decision Tree (XGBoost) Pairwise Classifier.  
**Core Innovation:** 
1. **Targeted Multi-Blocking with Frequency Capping:** A 31-rule blocking strategy combining exact tokens, address number + distinct tokens, folded 4-character name prefixes, and empty-name fallback rules. Rare token frequencies are strictly capped ($\le 50-100$) to prevent Cartesian explosions while securing $>93\%$ candidate recall.
2. **Chunked, Out-of-Core Processing:** DuckDB-backed chunked execution ensuring constant memory overhead ($< 2$ GB RAM) across 1.73M Source 1 entities and 9.97M target entities.
3. **Threshold Tuning for Macro $F_{0.5}$:** High-confidence decision threshold ($0.97$) strictly penalizing false merges on singletons and multi-matches alike.

---

## 3. Candidate Generation (Blocking)
To reduce the comparison space ($1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairs) to a computationally tractable candidate pool, we deployed 31 high-precision blocking rules:
- **Blocking keys used:**
  - Exact normalized name + country
  - Exact normalized address + country
  - Clean address number + distinctive address token / street token (frequency bounded)
  - 4-character folded name prefix + clean address number (Rule 4A)
  - Second name token ($\text{len} \ge 5$) + clean address number (Rule 5A)
  - Empty target name + address number set overlap + address token 1 (Rule 1B)
  - Numeric edit distance $\le 1$ and $+/- 1$ street number matching with distinctive tokens
- **Candidate pairs generated:** 67,000,632 pairs across 1,732,544 test $S_1$ entities (average ~38.7 candidates/entity, capped at top 180 per entity).
- **How true matches were preserved:** Multi-pass disjunctive blocking ensures that if an entity's name is corrupted or omitted, address-based and fuzzy-number rules capture the match, achieving a verified 93.14% validation blocking recall ceiling.

---

## 4. Matching Model

**Features used:**
- **Name features:** Character 3-gram Jaccard, Token Jaccard, Token Containment, Levenshtein edit similarity, length differences, exact match indicator.
- **Address features:** Clean address number match, numeric token overlap, token Jaccard, address token containment, edit similarity, missing postal/address indicators.
- **Global & Cross-Field:** TF-IDF pair cosine similarity, country equality indicator, combined name-address similarity.

**Model type:** XGBoost Classifier (tree-based gradient boosting, Apache-2.0 license, $< 8\text{B}$ parameters, fully compliant with challenge constraints).  
**Threshold selection method:** Grid search on held-out validation Source 1 entities directly maximizing Macro $F_{0.5}$. The optimal threshold was selected at $0.97$.

---

## 5. Results & Error Analysis

- **Validation Metrics (Held-out S1 split):**
  - **Macro $F_{0.5}$:** **0.8867**
  - **Macro Precision:** 0.9401
  - **Macro Recall:** 0.7867
  - **Global $F_{0.5}$:** 0.9209
  - **Blocking Recall:** 93.14%
  - **False Positives:** 586 (out of 20,989 true pairs)
- **Common false positives (wrong merges):** Co-located distinct businesses sharing identical shopping mall or complex addresses with truncated or generic names (e.g., "Cafe" or "Pharmacy" inside the same municipal building).
- **Common false negatives (missed matches):** Severe simultaneous corruption of both name and address (e.g., misspelled trade name combined with landmark-only address lacking street numbers or common tokens).

---

## 6. Conclusion
By pairing an expressive, selectivity-controlled 31-rule blocking pipeline with a heavily regularized XGBoost classifier evaluated at threshold 0.97, our solution achieves state-of-the-art performance (Macro $F_{0.5} = 0.8867$) while scaling smoothly to 11.7M records on commodity hardware.

---

## Appendix

### A. Code Artefacts
All runnable code is located under `code/business_entity_resolution/src/`:
- `duckdb_pipeline.py`: Production out-of-core blocking, feature scoring, and inference pipeline.
- `blocking.py`: Multi-rule blocking logic, inverted index generators, and frequency filtering.
- `features.py`: Pairwise similarity feature extractors (n-gram, token, fuzzy, numeric).
- `training.py`: Model training and threshold selection using held-out group cross-validation.
- `preprocessing.py`: String normalization, token extraction, and address parsing.
- `run_production_pipeline.py`: End-to-end runner executing the pipeline on test datasets.
- `config.py` & `data_loader.py`: Schema validation, file paths, and configuration parameters.

### B. Reproducibility
To regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
pip install -r requirements.txt
python run_production_pipeline.py
```
Validation check:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
