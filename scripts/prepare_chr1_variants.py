"""
Prepare chr1 variant list for Kaggle scoring (Experiment 1: replication).

Queries gnomAD v4 GraphQL API for chromosome 1 SNVs in batches, applying
the same AFR/EUR selection criteria as chr22.  Saves the variant TSV to
data/gnomad/gnomad_af_chr1.tsv so the Kaggle scoring kernel can consume it.

Run locally before submitting to Kaggle.  Expected runtime: 10-30 min
depending on API rate limits.

Usage:
    python scripts/prepare_chr1_variants.py
"""

import json
import time
from pathlib import Path

import requests

BASE = Path(__file__).parent.parent
OUT  = BASE / "data/gnomad/gnomad_af_chr1.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)

GNOMAD_URL = "https://gnomad.broadinstitute.org/api"
CHROM = "1"
# Chr1 length ~248 Mbp; query in 5 Mbp windows
WINDOW_SIZE = 5_000_000
CHR1_LENGTH = 248_956_422

# Same criteria as chr22
AFR_MIN = 0.10; AFR_EUR_MAX = 0.02; FOLD_MIN = 5.0
EUR_MIN = 0.10; EUR_AFR_MAX = 0.02

QUERY = """
query VariantsByRegion($chrom: String!, $start: Int!, $stop: Int!) {
  region(chrom: $chrom, start: $start, stop: $stop, reference_genome: GRCh38) {
    variants(dataset: gnomad_r4) {
      variant_id
      pos
      ref
      alt
      genome {
        af
        populations {
          id
          af
        }
      }
    }
  }
}
"""

def query_window(chrom, start, stop, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(
                GNOMAD_URL,
                json={"query": QUERY, "variables": {"chrom": chrom, "start": start, "stop": stop}},
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None

def extract_af(variant):
    try:
        pops = variant["genome"]["populations"]
        af_afr = next((p["af"] for p in pops if p["id"] == "afr"), None)
        af_eur = next((p["af"] for p in pops if p["id"] in ("nfe", "eur")), None)
        if af_afr is None or af_eur is None:
            return None, None
        return float(af_afr), float(af_eur)
    except (TypeError, KeyError):
        return None, None

print(f"Querying gnomAD v4 chr{CHROM} in {WINDOW_SIZE//1_000_000} Mbp windows...")
print(f"Output: {OUT}")

afr_rows = []
eur_rows = []
total_snvs = 0

start = 1
while start < CHR1_LENGTH:
    stop = min(start + WINDOW_SIZE - 1, CHR1_LENGTH)
    print(f"  Window {start:,} - {stop:,}", end=" ", flush=True)

    data = query_window(CHROM, start, stop)
    if data is None or "data" not in data or data["data"]["region"] is None:
        print("SKIP (no data)")
        start = stop + 1
        continue

    variants = data["data"]["region"]["variants"]
    print(f"({len(variants):,} variants)", end=" ")

    for v in variants:
        # SNVs only
        if len(v.get("ref", "")) != 1 or len(v.get("alt", "")) != 1:
            continue
        total_snvs += 1
        af_afr, af_eur = extract_af(v)
        if af_afr is None:
            continue

        pos = int(v["pos"])
        ref = v["ref"]; alt = v["alt"]

        # AFR-enriched
        if (af_afr >= AFR_MIN and af_eur <= AFR_EUR_MAX and
                af_eur > 0 and (af_afr / af_eur) >= FOLD_MIN):
            afr_rows.append((CHROM, pos, ref, alt, af_afr, af_eur))

        # EUR-enriched
        elif af_eur >= EUR_MIN and af_afr <= EUR_AFR_MAX:
            eur_rows.append((CHROM, pos, ref, alt, af_afr, af_eur))

    print(f"-> AFR: {len(afr_rows):,}  EUR: {len(eur_rows):,}")
    start = stop + 1
    time.sleep(0.5)   # polite rate-limiting

print(f"\nTotal SNVs seen: {total_snvs:,}")
print(f"AFR-enriched: {len(afr_rows):,}  EUR-enriched: {len(eur_rows):,}")

# Subsample AFR to 7,684 to match chr22 (fixed seed for reproducibility)
import random
random.seed(42)
if len(afr_rows) > 7684:
    afr_rows = random.sample(afr_rows, 7684)
    print(f"AFR subsampled to 7,684 (seed=42)")

# Subsample EUR to 509 if more are available
if len(eur_rows) > 509:
    eur_rows = random.sample(eur_rows, 509)
    print(f"EUR subsampled to 509 (seed=42)")

all_rows = [("CHROM","POS","REF","ALT","AF_afr","AF_eur","vtype")]
for r in afr_rows:
    all_rows.append((*r, "ref_minor"))
for r in eur_rows:
    all_rows.append((*r, "control"))

with open(OUT, "w") as f:
    for row in all_rows:
        f.write("\t".join(str(x) for x in row) + "\n")

print(f"Saved {len(all_rows)-1} variants -> {OUT}")
print("\nNext step: upload data/gnomad/gnomad_af_chr1.tsv to Kaggle and run")
print("kaggle_kernel/run_chr1_scoring.py with the NT models.")
