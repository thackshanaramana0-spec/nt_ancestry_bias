"""
Shared utilities for Paper D experiments.
Reference allele memorization in genomic foundation models.
Created: 2026-08-18
"""

import gzip
import os
import re
import json
import logging
import numpy as np
import torch
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data"
RESULTS = DATA / "results"
FIGURES = ROOT / "figures"

GNOMAD_DIR   = DATA / "gnomad"
CLINVAR_VCF  = DATA / "clinvar" / "clinvar.vcf.gz"
HG38_DIR     = DATA / "hg38"
HPRC_DIR     = DATA / "hprc"

for d in [RESULTS, FIGURES, GNOMAD_DIR, DATA/"clinvar", HG38_DIR, HPRC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    "NT-hg38": {
        "hf_id": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "type": "masked",
        "tokenizer_type": "kmer6",
        "description": "500M params, trained on hg38 only"
    },
    "NT-1000G": {
        "hf_id": "InstaDeepAI/nucleotide-transformer-500m-1000g",
        "type": "masked",
        "tokenizer_type": "kmer6",
        "description": "500M params, trained on 3202 diverse genomes"
    },
    "DNABERT2": {
        "hf_id": "zhihan1996/DNABERT-2-117M",
        "type": "masked",
        "tokenizer_type": "bpe",
        "description": "117M params, BPE tokenization, multi-species"
    },
    "HyenaDNA": {
        "hf_id": "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "type": "causal",
        "tokenizer_type": "char",
        "description": "14M params, hg38 only, character-level"
    },
    "Caduceus": {
        "hf_id": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        "type": "masked",
        "tokenizer_type": "char",
        "description": "hg38 only, Mamba-based"
    },
}

# ── FASTA utilities ──────────────────────────────────────────────────────────

_fasta_cache: dict = {}

def load_fasta_chrom(chrom: str) -> str:
    """Load one chromosome from per-chromosome hg38 FASTA files."""
    if chrom in _fasta_cache:
        return _fasta_cache[chrom]

    # try uncompressed first, then gzipped
    for fname in [f"{chrom}.fa", f"{chrom}.fa.gz", f"chr{chrom}.fa", f"chr{chrom}.fa.gz"]:
        fpath = HG38_DIR / fname
        if fpath.exists():
            log.info(f"Loading {fpath}")
            if str(fpath).endswith(".gz"):
                with gzip.open(fpath, "rt") as fh:
                    rec = next(SeqIO.parse(fh, "fasta"))
            else:
                rec = next(SeqIO.parse(str(fpath), "fasta"))
            seq = str(rec.seq).upper()
            _fasta_cache[chrom] = seq
            return seq

    raise FileNotFoundError(
        f"hg38 FASTA for {chrom} not found in {HG38_DIR}. "
        f"Download with: download_data.py --hg38 {chrom}"
    )

def extract_window(chrom: str, pos: int, window: int = 512) -> tuple[str, int]:
    """
    Extract a sequence window from hg38 centered on pos (1-based).
    Returns (sequence, variant_offset_in_sequence).
    """
    seq = load_fasta_chrom(chrom)
    half = window // 2
    start = max(0, pos - 1 - half)       # convert 1-based to 0-based
    end   = min(len(seq), pos - 1 + half)
    window_seq = seq[start:end]
    variant_offset = (pos - 1) - start
    return window_seq, variant_offset


# ── GFM model loading ────────────────────────────────────────────────────────

_model_cache: dict = {}

def load_model(model_name: str):
    """Load a GFM and its tokenizer, with caching."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

    cfg = MODELS[model_name]
    hf_id = cfg["hf_id"]
    log.info(f"Loading {model_name} from {hf_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if cfg["type"] == "masked":
            model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
    except ImportError as e:
        # triton / Flash Attention not available on Windows — skip model
        raise RuntimeError(
            f"{model_name} requires a package unavailable on Windows ({e}). "
            f"Skip this model or run on Linux/WSL."
        ) from e

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    log.info(f"{model_name} loaded on {device}")

    _model_cache[model_name] = (model, tokenizer, cfg, device)
    return _model_cache[model_name]


# ── VCF parsing ──────────────────────────────────────────────────────────────

def open_vcf(path: Path):
    """Open a plain or gzipped VCF file."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")

def parse_info_field(info: str) -> dict:
    """Parse VCF INFO field into a dict. Values are strings."""
    result = {}
    for item in info.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            result[k] = v
        else:
            result[item] = True
    return result

def is_snv(ref: str, alt: str) -> bool:
    """Return True only for simple single-nucleotide variants."""
    return (
        len(ref) == 1 and len(alt) == 1
        and ref in "ACGT" and alt in "ACGT"
        and ref != alt
    )

# ── Result I/O ───────────────────────────────────────────────────────────────

def save_results(data: list[dict], name: str):
    """Save a list of result dicts as JSON lines."""
    out = RESULTS / f"{name}.jsonl"
    with open(out, "w") as fh:
        for row in data:
            fh.write(json.dumps(row) + "\n")
    log.info(f"Saved {len(data)} rows to {out}")

def load_results(name: str) -> list[dict]:
    """Load JSON lines result file."""
    path = RESULTS / f"{name}.jsonl"
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]

# ── Misc ─────────────────────────────────────────────────────────────────────

NUCLEOTIDES = ["A", "C", "G", "T"]
AUTOSOMES   = [f"chr{i}" for i in range(1, 23)]

def complement(base: str) -> str:
    return {"A": "T", "T": "A", "C": "G", "G": "C"}.get(base, base)

def check_paths():
    """Print status of all expected data files."""
    items = [
        ("hg38 chr22", HG38_DIR / "chr22.fa.gz"),
        ("gnomAD chr22", GNOMAD_DIR / "gnomad.genomes.v4.1.sites.chr22.vcf.bgz"),
        ("ClinVar", CLINVAR_VCF),
        ("HPRC VCF", HPRC_DIR / "hprc_pangenome.vcf.gz"),
    ]
    for label, path in items:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {label}: {path}")
