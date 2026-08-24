"""
Paper D — Full chr22 NT scoring
Runs ALL variants on CPU (no GPU cap). 7684 ref-minor + 509 control.
~90 min on Kaggle CPU. No torch reinstall needed.
"""

import os, glob, json, gzip, math, time
import urllib.request, ssl
from datetime import datetime
import numpy as np
import torch
from scipy import stats as sp

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("Installing packages...")
os.system("pip install -q transformers==4.40.0 accelerate biopython scipy")

from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO

log(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")

# Use GPU if available and compatible, otherwise CPU — no variant cap either way
DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        major, _ = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        # Try a small tensor to confirm GPU actually works
        _ = torch.zeros(1).cuda()
        DEVICE = "cuda"
        log(f"GPU OK: {name} sm_{major}x")
    except Exception as e:
        log(f"GPU init failed ({e}) — using CPU")

log(f"Device: {DEVICE}")

# ── Variants (ALL of them, regardless of device) ──────────────────────────────
variant_path = "/kaggle/input/paperd-variants/variants.json"
if not os.path.exists(variant_path):
    found = glob.glob("/kaggle/input/**/variants.json", recursive=True)
    variant_path = found[0] if found else None
    if not variant_path:
        raise FileNotFoundError("variants.json not found")

with open(variant_path) as f:
    raw = json.load(f)

ref_minor = raw["ref_minor"]   # all 7,684
control   = raw["control"]     # all 509
log(f"Ref-minor: {len(ref_minor)}  Control: {len(control)}")

# ── hg38 chr22 ────────────────────────────────────────────────────────────────
HG38_PATH = "/kaggle/working/chr22.fa.gz"
HG38_URL  = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

if not os.path.exists(HG38_PATH):
    log("Downloading hg38 chr22...")
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(HG38_URL, HG38_PATH)

log("Parsing chr22...")
with gzip.open(HG38_PATH, "rt") as fh:
    CHR22 = str(next(SeqIO.parse(fh, "fasta")).seq).upper()
log(f"chr22: {len(CHR22):,} bp")

KMER   = 6
WINDOW = 512

def extract_window(pos):
    half  = WINDOW // 2
    start = max(0, pos - 1 - half)
    end   = min(len(CHR22), start + WINDOW)
    return CHR22[start:end], (pos - 1) - start

def score_nt(seq, offset, ref, alt, model, tokenizer):
    kmers = [seq[i:i+KMER] for i in range(0, len(seq) - KMER + 1, KMER)]
    ki = offset // KMER
    pi = offset % KMER
    if ki >= len(kmers) or len(kmers[ki]) != KMER:
        return float("nan"), "bad_kmer"
    rk = kmers[ki]
    if rk[pi] != ref:
        return float("nan"), "ref_mismatch"
    ak = rk[:pi] + alt + rk[pi+1:]
    rid = tokenizer.convert_tokens_to_ids(rk)
    aid = tokenizer.convert_tokens_to_ids(ak)
    if rid == tokenizer.unk_token_id or aid == tokenizer.unk_token_id:
        return float("nan"), "unk_token"
    enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=WINDOW)
    inp = enc["input_ids"].clone()
    mp  = ki + 1
    if mp >= inp.shape[1]:
        return float("nan"), "mask_oob"
    inp[0, mp] = tokenizer.mask_token_id
    with torch.no_grad():
        logits = model(inp.to(DEVICE)).logits
    lp = torch.log_softmax(logits[0, mp], dim=-1)
    return (lp[rid] - lp[aid]).item(), "ok"

# ── Main loop ─────────────────────────────────────────────────────────────────
MODELS = {
    "NT-hg38":  "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "NT-1000G": "InstaDeepAI/nucleotide-transformer-500m-1000g",
}

all_results = []

for model_name, hf_id in MODELS.items():
    log(f"\n{'='*60}")
    log(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model     = AutoModelForMaskedLM.from_pretrained(
        hf_id, trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    log(f"  Loaded on {next(model.parameters()).device}")

    for vtype, variants in [("ref_minor", ref_minor), ("control", control)]:
        log(f"  Scoring {len(variants)} {vtype}...")
        t0 = time.time()
        n_ok = n_err = 0

        for i, v in enumerate(variants):
            seq, offset = extract_window(v["pos"])
            if offset < 0 or offset >= len(seq):
                all_results.append({**v, "model": model_name, "vtype": vtype,
                                    "bias_score": None, "status": "oob"})
                n_err += 1
            else:
                try:
                    bs, status = score_nt(seq, offset, v["ref"], v["alt"], model, tokenizer)
                    if math.isnan(bs):
                        bs = None; n_err += 1
                    else:
                        n_ok += 1
                    all_results.append({**v, "model": model_name, "vtype": vtype,
                                        "bias_score": bs, "status": status})
                except Exception as e:
                    all_results.append({**v, "model": model_name, "vtype": vtype,
                                        "bias_score": None, "status": f"error:{e}"})
                    n_err += 1

            if (i+1) % 500 == 0 or (i+1) == len(variants):
                elapsed = time.time() - t0
                rate = (i+1)/elapsed if elapsed > 0 else 0
                eta  = (len(variants)-i-1)/rate if rate > 0 else 0
                log(f"    {i+1}/{len(variants)} | {rate:.1f}/s | ETA {eta/60:.0f}m | ok={n_ok}")

        ckpt = f"/kaggle/working/ckpt_{model_name.replace('-','_')}_{vtype}.jsonl"
        with open(ckpt, "w") as f:
            for r in [x for x in all_results if x["model"]==model_name and x["vtype"]==vtype]:
                f.write(json.dumps(r) + "\n")
        log(f"  Checkpoint saved: {ckpt}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ── Save ──────────────────────────────────────────────────────────────────────
out = "/kaggle/working/e1_full_chr22.jsonl"
with open(out, "w") as f:
    for r in all_results:
        f.write(json.dumps(r) + "\n")
log(f"\nSaved {len(all_results)} rows -> {out}")

# ── Summary ───────────────────────────────────────────────────────────────────
log("\n" + "="*60)
log("FINAL RESULTS SUMMARY")
log("="*60)

res_table = {}
for mn in MODELS:
    res_table[mn] = {}
    for vt in ["ref_minor", "control"]:
        sc = [r["bias_score"] for r in all_results
              if r["model"]==mn and r["vtype"]==vt
              and r.get("status")=="ok" and r["bias_score"] is not None]
        if sc:
            t, p = sp.ttest_1samp(sc, 0)
            pct = sum(1 for s in sc if s > 0)/len(sc)*100
            res_table[mn][vt] = sc
            log(f"{mn} | {vt}: N={len(sc)} mean={np.mean(sc):+.4f} sd={np.std(sc):.3f} "
                f"%>0={pct:.1f}% t={t:.3f} p={p:.3e}")

for vt in ["ref_minor", "control"]:
    h38  = res_table.get("NT-hg38",  {}).get(vt, [])
    g1kg = res_table.get("NT-1000G", {}).get(vt, [])
    if h38 and g1kg:
        t, p = sp.ttest_ind(h38, g1kg)
        d = (np.mean(h38)-np.mean(g1kg))/np.sqrt((np.var(h38)+np.var(g1kg))/2)
        red = (np.mean(h38)-np.mean(g1kg))/np.mean(h38)*100 if np.mean(h38)>0 else float("nan")
        log(f"\nNT-hg38 vs NT-1000G [{vt}]: t={t:.3f} p={p:.4e} d={d:.4f} reduction={red:.1f}%")

log("DONE.")
