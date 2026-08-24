"""
Fetch gnomAD v4 allele frequencies via the public REST/GraphQL API.

Replaces the need to download the full gnomAD VCF (~5-10 GB per chromosome).
Queries windowed regions of a chromosome, filters for ancestry-stratified AF,
and writes compact TSVs used by variant_filter.py.

Output columns: CHROM  POS  REF  ALT  AF  AF_afr  AF_eur  AF_amr  AF_eas

Usage:
    python fetch_gnomad_api.py --chrom chr22 --n 2000   # fetch up to 2000 variants
    python fetch_gnomad_api.py --chrom chr22             # fetch all

Created: 2026-08-19
"""

import argparse
import json
import ssl
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import GNOMAD_DIR, AUTOSOMES, log

API_URL = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r4"

# chr22 usable range (post-centromere): 10,700,000 – 50,818,468
CHR_RANGES = {
    "chr22": (10_700_000, 50_818_468),
    "chr21": (5_011_000, 46_709_983),
    "chr19": (60_000, 58_617_616),
    "chr17": (160_000, 83_257_441),
    "chr11": (200_000, 135_086_622),
    "chr1":  (900_000, 248_946_422),
}
WINDOW_SIZE = 200_000   # 200 kb per API call (gnomAD limit ~500 kb)
RATE_LIMIT_SLEEP = 0.5  # seconds between requests

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


GQL_REGION = """
query RegionVariants($chrom: String!, $start: Int!, $stop: Int!, $dataset: DatasetId!) {
  region(chrom: $chrom, start: $start, stop: $stop, reference_genome: GRCh38) {
    variants(dataset: $dataset) {
      variant_id
      ref
      alt
      genome {
        af
        populations {
          id
          ac
          an
        }
      }
    }
  }
}
"""


def _query_api(chrom_bare: str, start: int, stop: int, retries: int = 3) -> list[dict]:
    """Run one GraphQL region query. Returns list of variant dicts."""
    payload = json.dumps({
        "query": GQL_REGION,
        "variables": {"chrom": chrom_bare, "start": start, "stop": stop, "dataset": DATASET}
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, context=_ssl_ctx, timeout=60) as resp:
                data = json.loads(resp.read())
            if "errors" in data:
                log.warning(f"API error {chrom_bare}:{start}-{stop}: {data['errors']}")
                return []
            return data.get("data", {}).get("region", {}).get("variants", []) or []
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"API attempt {attempt+1} failed ({e}), retry in {wait}s")
            time.sleep(wait)
    return []


def _pop_af(populations: list[dict], pop_id: str) -> float:
    """Compute AF for a specific population from gnomAD ac/an counts."""
    for p in populations:
        if p["id"] == pop_id:
            ac = p.get("ac") or 0
            an = p.get("an") or 0
            return ac / an if an > 0 else 0.0
    return 0.0


def fetch_chrom(chrom: str, max_variants: int = 0, prefilter: bool = True) -> Path:
    """
    Fetch gnomAD AF data for one chromosome via API.
    Writes GNOMAD_DIR/gnomad_af_{chrom}.tsv.

    prefilter=True (default): only write variants with AF_afr>=0.05 OR AF_eur>=0.05.
    This reduces file size 20-50x and enables full-chromosome fetches.

    Returns path to output file.
    """
    chrom_bare = chrom.replace("chr", "")
    out_path = GNOMAD_DIR / f"gnomad_af_{chrom}.tsv"

    if out_path.exists() and out_path.stat().st_size > 1000:
        log.info(f"Already exists: {out_path} ({out_path.stat().st_size // 1024} KB)")
        return out_path

    start_bp, end_bp = CHR_RANGES.get(chrom, (60_000, 50_000_000))
    total_variants = 0
    done = False
    windows = range(start_bp, end_bp, WINDOW_SIZE)

    log.info(f"Fetching gnomAD AF for {chrom}: {len(windows)} windows of {WINDOW_SIZE//1000}kb")

    with open(out_path, "w") as fh:
        fh.write("CHROM\tPOS\tREF\tALT\tAF\tAF_afr\tAF_eur\tAF_amr\tAF_eas\n")

        for win_start in windows:
            win_end = min(win_start + WINDOW_SIZE, end_bp)
            variants = _query_api(chrom_bare, win_start, win_end)

            for v in variants:
                if not v.get("genome"):
                    continue
                genome = v["genome"]
                af_global = genome.get("af") or 0.0
                pops = genome.get("populations") or []

                ref = v.get("ref", "")
                alt = v.get("alt", "")
                # SNVs only
                if len(ref) != 1 or len(alt) != 1 or ref == alt:
                    continue
                if any(b not in "ACGT" for b in (ref, alt)):
                    continue

                # Parse variant_id: "22-12345678-A-G"
                parts = v["variant_id"].split("-")
                if len(parts) != 4:
                    continue
                pos = int(parts[1])

                af_afr = _pop_af(pops, "afr")
                af_eur = _pop_af(pops, "nfe")   # gnomAD v4 "nfe" = non-Finnish European
                af_amr = _pop_af(pops, "amr")
                af_eas = _pop_af(pops, "eas")

                # Pre-filter: skip rare-in-all-populations variants
                if prefilter and af_afr < 0.05 and af_eur < 0.05:
                    continue

                fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t"
                         f"{af_global:.6f}\t{af_afr:.6f}\t{af_eur:.6f}\t"
                         f"{af_amr:.6f}\t{af_eas:.6f}\n")
                total_variants += 1

                if max_variants > 0 and total_variants >= max_variants:
                    log.info(f"  Reached max_variants={max_variants} mid-window, stopping.")
                    done = True
                    break

            if total_variants % 5000 == 0 and total_variants > 0:
                log.info(f"  {chrom}: {total_variants:,} variants so far...")

            if done:
                break

            time.sleep(RATE_LIMIT_SLEEP)

    log.info(f"Fetched {total_variants:,} SNVs for {chrom} → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", default="chr22", help="Chromosome (e.g. chr22)")
    parser.add_argument("--n", type=int, default=0,
                        help="Max variants to fetch (0 = all)")
    parser.add_argument("--all-autosomes", action="store_true",
                        help="Fetch all autosomes in CHR_RANGES")
    args = parser.parse_args()

    if args.all_autosomes:
        for chrom in CHR_RANGES:
            fetch_chrom(chrom, max_variants=args.n)
    else:
        fetch_chrom(args.chrom, max_variants=args.n)


if __name__ == "__main__":
    main()
