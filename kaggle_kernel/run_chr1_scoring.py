"""
Kaggle scoring kernel for Experiment 1: chr1 replication.

Identical pipeline to chr22 but reads from gnomad_af_chr1.tsv and
chr1.fa (GRCh38).  Upload both files as Kaggle dataset inputs.

Input files expected in /kaggle/input/:
  gnomad_af_chr1.tsv          (from prepare_chr1_variants.py)
  Homo_sapiens.GRCh38.dna.chromosome.1.fa  (Ensembl GRCh38 chr1 FASTA)
    OR chr1.fa  (UCSC hg38 chr1)

Output: /kaggle/working/chr1_full_scored.jsonl
"""

import json, gc, os, math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR = Path("/kaggle/input")
WORK_DIR  = Path("/kaggle/working")
WORK_DIR.mkdir(exist_ok=True)

VARIANT_TSV = INPUT_DIR / "gnomad_af_chr1.tsv"
# Try common FASTA names
for fname in ["chr1.fa", "chr1.fa.gz", "Homo_sapiens.GRCh38.dna.chromosome.1.fa"]:
    CHR1_FA = INPUT_DIR / fname
    if CHR1_FA.exists():
        break

OUT_JSONL = WORK_DIR / "chr1_full_scored.jsonl"

MODELS = {
    "NT-hg38":  "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "NT-1000G": "InstaDeepAI/nucleotide-transformer-500m-1000g",
}
CONTEXT_TOKENS = 512
KMER = 6

# ── Load chr1 sequence ────────────────────────────────────────────────────────
print("Loading chr1 sequence...")
import gzip

def load_fasta(path):
    seq = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.append(line.strip().upper())
    return "".join(seq)

chr1_seq = load_fasta(CHR1_FA)
print(f"chr1 length: {len(chr1_seq):,} bp")

# ── Load variants ─────────────────────────────────────────────────────────────
variants = []
with open(VARIANT_TSV) as f:
    header = f.readline().strip().split("\t")
    for line in f:
        parts = line.strip().split("\t")
        d = dict(zip(header, parts))
        variants.append({
            "chrom": d["CHROM"],
            "pos":   int(d["POS"]),
            "ref":   d["REF"],
            "alt":   d["ALT"],
            "af_afr": float(d["AF_afr"]),
            "af_eur": float(d["AF_eur"]),
            "vtype": d["vtype"],
        })
print(f"Loaded {len(variants):,} variants from TSV")

def get_kmer_at(seq, pos0, kmer=6):
    """Return the 6-mer that contains position pos0 (0-based)."""
    start = (pos0 // kmer) * kmer
    return start, seq[start:start+kmer]

def score_variant(model, tokenizer, seq, pos1, ref, alt, device):
    pos0 = pos1 - 1
    if pos0 < 0 or pos0 >= len(seq):
        return None, "out_of_range"
    if seq[pos0] != ref:
        return None, f"ref_mismatch:{seq[pos0]}!={ref}"

    km_start, ref_kmer = get_kmer_at(seq, pos0)
    alt_kmer = ref_kmer[:pos0 - km_start] + alt + ref_kmer[pos0 - km_start + 1:]
    if len(alt_kmer) != 6:
        return None, "kmer_len_error"

    # Centre CONTEXT_TOKENS window on the k-mer
    half = (CONTEXT_TOKENS * KMER) // 2
    ctx_start = max(0, km_start - half)
    ctx_end   = min(len(seq), ctx_start + CONTEXT_TOKENS * KMER)
    ctx_start = max(0, ctx_end - CONTEXT_TOKENS * KMER)
    ctx_seq   = seq[ctx_start:ctx_end]

    # Replace masked k-mer in context
    km_local = km_start - ctx_start
    mask_seq  = ctx_seq[:km_local] + "[MASK]" + ctx_seq[km_local + KMER:]

    toks = tokenizer(mask_seq, return_tensors="pt").to(device)
    mask_idx = (toks.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_idx) == 0:
        return None, "no_mask_token"
    mask_pos = mask_idx[0].item()

    with torch.no_grad():
        logits = model(**toks).logits[0, mask_pos]
        log_probs = torch.log_softmax(logits, dim=-1)

    ref_id = tokenizer.convert_tokens_to_ids(ref_kmer)
    alt_id = tokenizer.convert_tokens_to_ids(alt_kmer)
    if ref_id == tokenizer.unk_token_id or alt_id == tokenizer.unk_token_id:
        return None, "unk_kmer"

    score = float(log_probs[ref_id].item() - log_probs[alt_id].item())
    return score, "ok"

# ── Score ─────────────────────────────────────────────────────────────────────
device = torch.device("cpu")
with open(OUT_JSONL, "w") as fout:
    for model_name, model_id in MODELS.items():
        print(f"\nLoading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
        model.to(device)

        for i, v in enumerate(variants):
            if i % 500 == 0:
                print(f"  {i}/{len(variants)}", flush=True)
            score, status = score_variant(
                model, tokenizer, chr1_seq, v["pos"], v["ref"], v["alt"], device
            )
            rec = {
                "model": model_name,
                "chrom": v["chrom"],
                "pos":   v["pos"],
                "ref":   v["ref"],
                "alt":   v["alt"],
                "vtype": v["vtype"],
                "af_afr": v["af_afr"],
                "af_eur": v["af_eur"],
                "bias_score": score,
                "status": status,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

        del model
        gc.collect()
        print(f"{model_name} done.")

print(f"\nAll done. Output: {OUT_JSONL}")
