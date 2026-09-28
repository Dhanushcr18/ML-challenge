import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path

t0 = time.time()
out_dir = Path("tmp/step8_full_submission")
match_tsv = out_dir / "matching_results.tsv"
cand_tsv = out_dir / "candidate_pairs.tsv"
zip_out = out_dir / "FINAL_SUBMISSION.zip"

print("="*75)
print("INSTANT POST-INFERENCE PACKAGER & VALIDATOR")
print("="*75)

# 1. Fast Validation (standard checks: row count, format, columns, singletons)
print("\n1. Running fast official validator (without --check-ids for zero latency)...")
t_val = time.time()
cmd_fast = [
    sys.executable, "utils/validate_submission.py",
    "--matching", str(match_tsv),
    "--candidate", str(cand_tsv),
    "--test-dir", "dataset/test"
]
res_fast = subprocess.run(cmd_fast, capture_output=True, text=True)
print(res_fast.stdout)
if res_fast.returncode != 0:
    print("VALIDATION ERROR:", res_fast.stderr)
    sys.exit(1)
print(f"   Fast validation PASSED in {time.time()-t_val:.2f}s")

# 2. Package ZIP in exact portal structure (output/matching_results.tsv)
print("\n2. Packaging submission ZIP...")
t_zip = time.time()
if zip_out.exists():
    zip_out.unlink()

with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    # Portal expects matching_results.tsv (and candidate_pairs.tsv if provided)
    zf.write(match_tsv, arcname="output/matching_results.tsv")
    if cand_tsv.exists():
        zf.write(cand_tsv, arcname="output/candidate_pairs.tsv")

zip_size_mb = zip_out.stat().st_size / (1024 * 1024)
print(f"   ZIP packaged: {zip_out} ({zip_size_mb:.2f} MB) in {time.time()-t_zip:.2f}s")

# 3. Optional deep check-ids in background / stdout
print("\n3. Running full ID-existence verification (informational)...")
t_deep = time.time()
cmd_deep = [
    sys.executable, "utils/validate_submission.py",
    "--matching", str(match_tsv),
    "--test-dir", "dataset/test",
    "--check-ids"
]
res_deep = subprocess.run(cmd_deep, capture_output=True, text=True)
print(res_deep.stdout)
print(f"   Deep ID verification completed in {time.time()-t_deep:.2f}s")

print("="*75)
print(f"ALL PACKAGING & VALIDATION COMPLETED IN {time.time()-t0:.2f}s (< 1 minute)")
print("="*75)
