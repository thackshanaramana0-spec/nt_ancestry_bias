"""
Experiment 1 — Step 1: Filter gnomAD AF tables for reference-minor variants.

A "reference-minor variant" is one where:
  - AF_afr > AFR_MIN  (common in African populations)
  - AF_eur < EUR_MAX  (rare in European populations)
  - AF     < GLOBAL_MAX (reference allele is globally minor)
  - SNV only (already filtered in AF table)

These are variants where the hg38 reference allele is NOT the typical allele
in African populations — the strongest test of reference allele memorization.

Also produces a CONTROL set (European-common, African-rare) for sanity checking.

Usage:
    python variant_filter.py --chrom chr22             # filter one chrom
    python variant_filter.py --all                     # filter chr1-chr22
    python variant_filter.py --chrom chr22 --n 5000   # limit to 5000 variants

Output:
    data/results/ref_minor_variants.{chrom}.tsv
    data/results/eur_common_variants.{chrom}.tsv  (control)

Created: 2026-08-18
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import GNOMAD_DIR, RESULTS, log

# ── Thresholds ────────────────────────────────────────────────────────────────

AFR_MIN    = 0.10   # AF_afr must be >= this (substantially common in Africans)
EUR_MAX    = 0.02   # AF_eur must be <= this (rare in Europeans)
FOLD_MIN   = 5.0    # AF_afr / AF_eur fold-enrichment (avoid zero-denominator edge cases)
# Note: no global AF cutoff — the fold enrichment captures the signal

# Control group: European-common, African-rare (sanity check — should show NO bias)
EUR_MIN_CTRL = 0.10
AFR_MAX_CTRL = 0.02

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

# ── Core filter ───────────────────────────────────────────────────────────────

def filter_chrom(chrom: str, n_limit: int = None) -> tuple[int, int]:
    """
    Filter AF table for one chromosome.
    Accepts either:
      - data/gnomad/gnomad_af_{chrom}.tsv      (from fetch_gnomad_api.py, uncompressed)
      - data/gnomad/af_table.{chrom}.tsv.gz    (from download_data.py, gzipped VCF extract)
    Returns (n_ref_minor, n_control) written.
    """
    # Prefer API-fetched TSV (no download needed), fall back to VCF-extracted gz
    af_table_plain = GNOMAD_DIR / f"gnomad_af_{chrom}.tsv"
    af_table_gz    = GNOMAD_DIR / f"af_table.{chrom}.tsv.gz"

    if af_table_plain.exists() and af_table_plain.stat().st_size > 1000:
        af_table = af_table_plain
        use_gz   = False
    elif af_table_gz.exists():
        af_table = af_table_gz
        use_gz   = True
    else:
        log.warning(f"No AF table for {chrom}. Run: python fetch_gnomad_api.py --chrom {chrom}")
        return 0, 0

    out_ref   = RESULTS / f"ref_minor_variants.{chrom}.tsv"
    out_ctrl  = RESULTS / f"eur_common_variants.{chrom}.tsv"

    n_ref = 0
    n_ctrl = 0
    n_total = 0

    header = "CHROM\tPOS\tREF\tALT\tAF\tAF_afr\tAF_eur\n"

    open_fn = gzip.open if use_gz else open
    with open_fn(af_table, "rt") as fh, \
         open(out_ref, "w") as fh_ref, \
         open(out_ctrl, "w") as fh_ctrl:

        fh_ref.write(header)
        fh_ctrl.write(header)

        next(fh)  # skip header
        for line in fh:
            n_total += 1
            parts = line.rstrip().split("\t")
            if len(parts) < 9:
                continue

            chrom_col, pos, ref, alt, af, af_afr, af_eur, af_amr, af_eas = parts[:9]

            # Parse floats safely
            try:
                af_f     = float(af)     if af     != "." else None
                af_afr_f = float(af_afr) if af_afr != "." else None
                af_eur_f = float(af_eur) if af_eur != "." else None
            except ValueError:
                continue

            if af_f is None or af_afr_f is None or af_eur_f is None:
                continue

            row = f"{chrom_col}\t{pos}\t{ref}\t{alt}\t{af_f:.4f}\t{af_afr_f:.4f}\t{af_eur_f:.4f}\n"

            # Reference-minor: substantially enriched in Africans vs Europeans
            fold = af_afr_f / af_eur_f if af_eur_f > 0 else float("inf")
            if (af_afr_f >= AFR_MIN and af_eur_f <= EUR_MAX and fold >= FOLD_MIN):
                if n_limit is None or n_ref < n_limit:
                    fh_ref.write(row)
                    n_ref += 1

            # Control: European-common, African-rare (symmetric criteria)
            elif (af_eur_f >= EUR_MIN_CTRL and af_afr_f <= AFR_MAX_CTRL):
                if n_limit is None or n_ctrl < n_limit:
                    fh_ctrl.write(row)
                    n_ctrl += 1

            if n_total % 1_000_000 == 0:
                log.info(f"  {chrom}: scanned {n_total:,} | ref_minor={n_ref:,} | ctrl={n_ctrl:,}")

            # Early exit if both limits reached
            if n_limit and n_ref >= n_limit and n_ctrl >= n_limit:
                break

    log.info(
        f"{chrom}: scanned {n_total:,} total → "
        f"{n_ref:,} ref-minor variants, {n_ctrl:,} control variants"
    )
    return n_ref, n_ctrl


def filter_all(n_limit: int = None):
    """Filter all autosomes and concatenate results."""
    totals = {"ref_minor": 0, "control": 0}

    for chrom in AUTOSOMES:
        n_ref, n_ctrl = filter_chrom(chrom, n_limit=n_limit)
        totals["ref_minor"] += n_ref
        totals["control"]   += n_ctrl

    # Concatenate into single files
    for variant_type in ["ref_minor_variants", "eur_common_variants"]:
        merged = RESULTS / f"{variant_type}.all.tsv"
        with open(merged, "w") as fh_out:
            first = True
            for chrom in AUTOSOMES:
                per_chrom = RESULTS / f"{variant_type}.{chrom}.tsv"
                if not per_chrom.exists():
                    continue
                with open(per_chrom) as fh_in:
                    for i, line in enumerate(fh_in):
                        if i == 0 and not first:
                            continue  # skip header for non-first file
                        fh_out.write(line)
                first = False
        log.info(f"Merged → {merged}")

    log.info(
        f"Total: {totals['ref_minor']:,} reference-minor variants, "
        f"{totals['control']:,} control variants"
    )
    return totals


def summarize(chrom: str):
    """Print summary statistics of the filtered variant files."""
    for label, fname in [
        ("Reference-minor", f"ref_minor_variants.{chrom}.tsv"),
        ("Control (EUR-common)", f"eur_common_variants.{chrom}.tsv"),
    ]:
        path = RESULTS / fname
        if not path.exists():
            print(f"  {label}: NOT FOUND ({path})")
            continue
        n = sum(1 for _ in open(path)) - 1  # subtract header
        print(f"  {label} ({chrom}): {n:,} variants")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Filter gnomAD for reference-minor variants")
    parser.add_argument("--chrom", help="Chromosome to filter (e.g. chr22)")
    parser.add_argument("--all", action="store_true", help="Filter all autosomes")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to N variants per type (for testing)")
    parser.add_argument("--summarize", metavar="CHROM", help="Print summary for a chromosome")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return

    if args.all:
        filter_all(n_limit=args.n)
    elif args.chrom:
        chrom = args.chrom if args.chrom.startswith("chr") else f"chr{args.chrom}"
        filter_chrom(chrom, n_limit=args.n)
    else:
        parser.print_help()
        print("\nExample: python variant_filter.py --chrom chr22 --n 1000")


if __name__ == "__main__":
    main()
