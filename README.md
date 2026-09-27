# Business Entity Resolution

This repository resolves records in a deduplicated Source 1 against noisy Source 2 and Source 3 using only the supplied challenge TSVs. There are no external identity lookups or data enrichment calls. Country is treated as an open string field.

## Architecture

`src/data_loader.py` reads the pandas utility path with `pd.read_csv(..., sep="\t")`, inspects missing values in chunks, and validates the required schema and source ID prefixes. The production runner uses `src/duckdb_pipeline.py` to stage TSV data on disk under a 2 GB DuckDB memory limit, so it does not load all source files into pandas at once. It creates normalized name, address, and country columns while retaining original values. Normalization lowercases, standardizes punctuation and whitespace, and maps common legal suffix and address variants without dropping useful tokens.

`blocking.py` provides the in-memory utility implementation. The production runner uses DuckDB joins for exact name, exact address, country/name prefix, and country/address-number blocks; overly common keys are filtered and a per-entity cap bounds the final candidate set. All candidates passed to the classifier are written to `output/candidate_pairs.tsv`; predicted matches are selected only from this set.

`features.py` calculates name and address exactness, character-trigram and token Jaccard/containment, pair-local TF-IDF cosine, numeric-token overlap, normalized edit similarity, length differences, missing indicators, country equality, and combined similarity. `training.py` labels blocked pairs from the provided ground truth, caps sampled negatives per source entity, holds out whole Source 1 entities for validation, and trains an XGBoost model (Apache-2.0; no pretrained or external model data). The threshold is selected by validation F0.5. The refit model and threshold are stored in `models/matcher.joblib`.

## Metric and validation

For precision `P` and recall `R`, F0.5 is `(1.25 * P * R) / (0.25 * P + R)`. Candidate recall is reported because the classifier cannot recover pairs excluded during blocking. Thresholds are compared on a held-out group of Source 1 entities, reducing leakage between train and validation pairs. Entities with no output matches are reported as predicted singletons and receive an empty `matched_entity_ids` value.

## Run

Install requirements in a Python environment, ensure the supplied files exist under `dataset/train` and `dataset/test`, and have several GB of free disk space for DuckDB temporary data, then run:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The run produces exactly two TSV files in `output/`. `matching_results.tsv` has one row per test Source 1 entity and comma-separated S2/S3 IDs (or an empty cell). `candidate_pairs.tsv` lists the final candidate IDs actually scored for each Source 1 entity. Model parameters and selected threshold are stored in `models/matcher.joblib`.

Validate with the supplied official checker:

```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

The pipeline logs dataset sizes, candidate counts and reduction, training class counts, validation precision/recall/F0.5, threshold, predicted matches, and singleton count. Large input files are read as strings to avoid mixed-type inference; blocking controls candidate volume. Results depend on the configured posting and per-entity caps, which trade some recall for bounded runtime and memory.
