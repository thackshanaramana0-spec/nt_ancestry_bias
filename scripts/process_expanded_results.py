"""
Paper D — process expanded results (N_EUR=6830).
Run after downloading e1_full_chr22.jsonl from the expanded Kaggle kernel.

Usage:
    python scripts/process_expanded_results.py --input <path_to_expanded.jsonl>

Downloads kernel output if --download flag is passed:
    python scripts/process_expanded_results.py --download
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp
from scipy.stats import chi2_contingency

BASE    = Path(__file__).parent.parent
FIG_DIR = BASE / "figures"
OUT_DIR = BASE / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_REF = "thackshana/paper-d-expanded-eur-chr22-nt-scoring"

def download_results():
    print(f"Downloading output from {KERNEL_REF} ...")
    out_dir = BASE / "kaggle_kernel/output_expanded"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_REF, "-p", str(out_dir)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(1)
    jsonl_files = list(out_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl file found in output. Check kernel is complete.")
        sys.exit(1)
    return jsonl_files[0]


def load_records(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("bias_score") is not None:
                records.append(r)
    return records


def get(records, model, vtype):
    return [r for r in records if r["model"] == model and r["vtype"] == vtype]


def full_analysis(records, label="expanded"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                          "axes.spines.top": False, "axes.spines.right": False})
    C_AFR = "#1f77b4"; C_EUR = "#d62728"

    afr_h38 = get(records, "NT-hg38",  "ref_minor")
    eur_h38 = get(records, "NT-hg38",  "control")
    afr_1kg = get(records, "NT-1000G", "ref_minor")
    eur_1kg = get(records, "NT-1000G", "control")

    print(f"\n{'='*60}")
    print(f"EXPANDED ANALYSIS ({label})")
    print(f"N: AFR={len(afr_1kg)}, EUR={len(eur_1kg)}")
    print(f"{'='*60}")

    stats = {}

    # 1. Descriptives
    print("\n--- Descriptives ---")
    for lbl, grp, ak in [("AFR_hg38", afr_h38,"af_afr"),("EUR_hg38",eur_h38,"af_eur"),
                          ("AFR_1000G",afr_1kg,"af_afr"),("EUR_1000G",eur_1kg,"af_eur")]:
        sc = np.array([r["bias_score"] for r in grp])
        af = np.array([r[ak] for r in grp])
        t1, p1 = sp.ttest_1samp(sc, 0)
        stats[lbl] = {"n": len(sc), "mean": float(np.mean(sc)), "sd": float(np.std(sc)),
                      "pct_pos": float(np.mean(sc > 0)), "p_vs0": float(p1),
                      "af_mean": float(np.mean(af))}
        print(f"{lbl}: N={len(sc)} mean={np.mean(sc):+.4f} sd={np.std(sc):.4f} "
              f"%pos={100*np.mean(sc>0):.1f}% p={p1:.2e} AF_mean={np.mean(af):.4f}")

    # 2. Between-group (main result)
    print("\n--- Between-group AFR vs EUR ---")
    for model, afr, eur in [("NT-hg38", afr_h38, eur_h38), ("NT-1000G", afr_1kg, eur_1kg)]:
        sc_a = np.array([r["bias_score"] for r in afr])
        sc_e = np.array([r["bias_score"] for r in eur])
        t, p = sp.ttest_ind(sc_a, sc_e, equal_var=False)
        diff = float(np.mean(sc_e) - np.mean(sc_a))
        d = diff / float(np.sqrt((np.var(sc_a) + np.var(sc_e)) / 2))
        key = f"between_{model.replace('-','_')}"
        stats[key] = {"diff": diff, "t": float(t), "p": float(p), "d": float(d)}
        print(f"{model}: EUR-AFR={diff:+.4f} t={t:.3f} p={p:.3e} d={d:.4f}")

    # 3. Between-model (diversity correction)
    print("\n--- Between-model hg38 vs 1000G ---")
    for vtype, h38, kg in [("AFR", afr_h38, afr_1kg), ("EUR", eur_h38, eur_1kg)]:
        sc_h = np.array([r["bias_score"] for r in h38])
        sc_k = np.array([r["bias_score"] for r in kg])
        t, p = sp.ttest_ind(sc_h, sc_k, equal_var=False)
        diff = float(np.mean(sc_h) - np.mean(sc_k))
        pct = diff / float(np.mean(sc_h)) * 100
        stats[f"between_model_{vtype}"] = {"reduction_pct": float(pct), "p": float(p)}
        print(f"{vtype}: reduction={pct:.1f}% p={p:.3e}")

    # 4. Within-group calibration
    print("\n--- Within-group calibration ---")
    for lbl, grp, ak in [("AFR_hg38",afr_h38,"af_afr"),("AFR_1000G",afr_1kg,"af_afr"),
                          ("EUR_hg38",eur_h38,"af_eur"),("EUR_1000G",eur_1kg,"af_eur")]:
        sc = np.array([r["bias_score"] for r in grp])
        af = np.array([r[ak] for r in grp])
        r_p, p_p = sp.pearsonr(af, sc)
        stats[f"calib_{lbl}"] = {"r": float(r_p), "p": float(p_p)}
        print(f"{lbl}: r={r_p:+.4f} p={p_p:.3e}")

    # Fisher z-test
    print("\n--- Fisher z-test ---")
    for short, model in [("hg38","NT-hg38"),("1000G","NT-1000G")]:
        r_a = stats[f"calib_AFR_{short}"]["r"]
        r_e = stats[f"calib_EUR_{short}"]["r"]
        n_a = stats[f"AFR_{short}"]["n"]
        n_e = stats[f"EUR_{short}"]["n"]
        z_a = np.arctanh(r_a); z_e = np.arctanh(r_e)
        z_diff = (z_a - z_e) / np.sqrt(1/(n_a-3) + 1/(n_e-3))
        p_f = float(2 * sp.norm.sf(abs(z_diff)))
        stats[f"fisher_{short}"] = {"r_AFR": r_a, "r_EUR": r_e, "p": p_f}
        print(f"{model}: r_AFR={r_a:+.4f} r_EUR={r_e:+.4f} Fisher p={p_f:.3e}")

    # 5. Bootstrap CI
    print("\n--- Bootstrap CI (NT-1000G) ---")
    sc_a = np.array([r["bias_score"] for r in afr_1kg])
    sc_e = np.array([r["bias_score"] for r in eur_1kg])
    rng = np.random.default_rng(42)
    boot = [float(np.mean(rng.choice(sc_e,len(sc_e),True)) -
                  np.mean(rng.choice(sc_a,len(sc_a),True))) for _ in range(10000)]
    ci_lo, ci_hi = float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))
    stats["bootstrap"] = {"mean": float(np.mean(boot)), "ci95": [ci_lo, ci_hi], "all_positive": bool(ci_lo>0)}
    print(f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  all_positive={ci_lo>0}")

    # 6. Negative bias (clinical downstream)
    print("\n--- Negative bias (false flagging) ---")
    pct_a = float(np.mean(sc_a < 0))
    pct_e = float(np.mean(sc_e < 0))
    or_nb = (pct_a/(1-pct_a)) / (pct_e/(1-pct_e))
    table = [[int(np.sum(sc_a<0)),int(np.sum(sc_a>=0))],
             [int(np.sum(sc_e<0)),int(np.sum(sc_e>=0))]]
    chi2_v, p_chi, _, _ = chi2_contingency(table)
    stats["negative_bias"] = {"pct_AFR": pct_a, "pct_EUR": pct_e,
                               "OR": float(or_nb), "p": float(p_chi)}
    print(f"AFR={100*pct_a:.1f}% EUR={100*pct_e:.1f}% OR={or_nb:.2f}x p={p_chi:.3e}")

    # Save stats
    out_path = OUT_DIR / f"paper_d_stats_{label}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved -> {out_path}")

    # Figure 1: Violin
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [(np.array([r["bias_score"] for r in afr_h38]), C_AFR, "AFR\nNT-hg38"),
              (np.array([r["bias_score"] for r in eur_h38]), C_EUR, "EUR\nNT-hg38"),
              (np.array([r["bias_score"] for r in afr_1kg]), C_AFR, "AFR\nNT-1000G"),
              (np.array([r["bias_score"] for r in eur_1kg]), C_EUR, "EUR\nNT-1000G")]
    positions = [1, 2, 4, 5]
    for (data, col, lbl), pos in zip(groups, positions):
        vp = ax.violinplot(data, positions=[pos], showmedians=False, showextrema=False, widths=0.8)
        for body in vp["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.45); body.set_edgecolor(col)
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        ax.plot([pos]*2, [q25, q75], color=col, lw=4, solid_capstyle="round")
        ax.scatter([pos], [q50], color="white", s=30, zorder=5, linewidths=1.5, edgecolors=col)
        ax.scatter([pos], [np.mean(data)], color=col, marker="D", s=40, zorder=6)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

    def bracket(x1, x2, y, text):
        h = 0.12
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", lw=1.0)
        ax.text((x1+x2)/2, y+h+0.05, text, ha="center", va="bottom", fontsize=9)

    p1 = stats["between_NT_hg38"]["p"]; p2 = stats["between_NT_1000G"]["p"]
    bracket(1, 2, 7.5, f"p={p1:.1e}"); bracket(4, 5, 7.5, f"p={p2:.1e}")
    p3 = stats["between_model_AFR"]["p"]
    bracket(1, 4, 9.2, f"AFR reduction p={p3:.2f}")

    ci_lo2, ci_hi2 = stats["bootstrap"]["ci95"]
    ax.text(4.5, -7.5,
            f"EUR-AFR (NT-1000G): {stats['bootstrap']['mean']:+.3f}\n95% CI [{ci_lo2:+.3f}, {ci_hi2:+.3f}]",
            fontsize=8, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6", edgecolor="#e0a000", alpha=0.9))

    ax.set_xticks(positions)
    ax.set_xticklabels([lbl for _, _, lbl in groups], fontsize=10)
    ax.set_ylabel("Bias score  [log P(ref k-mer) - log P(alt k-mer)]", fontsize=10)
    ax.set_title(f"Reference allele bias — expanded analysis (N_EUR={len(eur_1kg):,})", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 6); ax.set_ylim(-9, 12.5)

    handles = [mpatches.Patch(color=C_AFR, alpha=0.6, label=f"AFR (N={len(afr_1kg):,})"),
               mpatches.Patch(color=C_EUR, alpha=0.6, label=f"EUR (N={len(eur_1kg):,})"),
               Line2D([0],[0], color="black", lw=1, ls="--", label="Zero bias"),
               Line2D([0],[0], marker="D", color="gray", markersize=7, ls="", label="Mean")]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig_path = FIG_DIR / f"fig1_violin_{label}.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to expanded .jsonl file")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle first")
    args = parser.parse_args()

    if args.download:
        jsonl_path = download_results()
    elif args.input:
        jsonl_path = Path(args.input)
    else:
        # Try default expanded output location
        default = BASE / "kaggle_kernel/output_expanded/e1_full_chr22.jsonl"
        if default.exists():
            jsonl_path = default
        else:
            print("Usage: --input <file.jsonl>  OR  --download")
            sys.exit(1)

    records = load_records(jsonl_path)
    print(f"Loaded {len(records)} OK records from {jsonl_path}")
    full_analysis(records, label="expanded_2026-08-23")


if __name__ == "__main__":
    main()
