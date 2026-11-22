"""Figures, LaTeX table fragments and a numeric trace for the Month 2 report.

Run from the repository root:

    python month2_sources/make_figures.py

Inputs are the copies of published tables in ``month2_sources/data`` (see
``month2_sources/SOURCES.md`` for the exact commits).  Outputs:

* ``figures/month2/*.pdf`` / ``*.png``  -- figures included by ``main.tex``
* ``tables/month2/*.tex``              -- table bodies included by ``main.tex``
* ``month2_sources/NUMBERS.md``        -- every number quoted in the text, with its source

No new evaluation is performed: every value is read or averaged from the
published aggregate tables.  Averaging order follows the published packages:
per-frame RMSE -> mean over frames of a condition -> mean over conditions ->
mean over the three observation replicas -> mean (and min/max) over the three
training seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "month2_sources" / "data"
FIG = ROOT / "figures" / "month2"
TAB = ROOT / "tables" / "month2"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

FAM = {"sparse_unet": "U-Net", "adapted_glu": "GLU"}
COL = {"sparse_unet": "#1f5fa8", "adapted_glu": "#d9731a"}
CONFIGS = [("sparse_unet", True), ("sparse_unet", False), ("adapted_glu", True), ("adapted_glu", False)]
VALC = ["0NH3_1.05.h5", "20NH3_1.1.h5", "40NH3_0.9.h5"]
VALLAB = {"0NH3_1.05.h5": "0% NH$_3$, $\\phi$=1.05", "20NH3_1.1.h5": "20% NH$_3$, $\\phi$=1.10",
          "40NH3_0.9.h5": "40% NH$_3$, $\\phi$=0.90"}
KS = [0, 1, 4, 9, 16, 32]

TRACE: list[str] = []


def cfg_label(fam, p):
    return f"{FAM[fam]}, {'with' if p else 'without'} $p$"


def cfg_plain(fam, p):
    return f"{FAM[fam]} {'with' if p else 'without'} p"


def note(key, value, source):
    TRACE.append(f"| {key} | {value} | {source} |")


def fmt(x, d=2):
    return f"{x:.{d}f}"


def boolcol(df, col="p_conditioning"):
    if df[col].dtype == object or str(df[col].dtype).startswith("str"):
        df[col] = df[col].astype(str).map({"True": True, "False": False})
    return df


# ---------------------------------------------------------------- loading
s7 = boolcol(pd.read_csv(DATA / "free_s7/global_by_setting.csv"))
s7pc = boolcol(pd.read_csv(DATA / "free_s7/per_condition.csv"))
s7cost = pd.read_csv(DATA / "free_s7/training_costs_by_run.csv")
s7eval = pd.read_csv(DATA / "free_s7/eval_costs_by_task.csv")
s7runs = boolcol(pd.read_csv(DATA / "free_s7/run_matrix.csv"))
base = boolcol(pd.read_csv(DATA / "free_s7/baselines_c149_macro.csv"))
swap = boolcol(pd.read_csv(DATA / "free_s7/swap_neural_by_run.csv"))
sw = boolcol(pd.read_csv(DATA / "k_sweep/by_seed.csv"))
swpc = boolcol(pd.read_csv(DATA / "k_sweep/per_condition_val_best_primary.csv"))
jr = boolcol(pd.read_csv(DATA / "joint/joint_by_run_replica_K.csv"))
js = boolcol(pd.read_csv(DATA / "joint/joint_by_seed_K.csv"))
ja = boolcol(pd.read_csv(DATA / "joint/joint_across_seeds_K.csv"))
jep = pd.read_csv(DATA / "joint/assisted_checkpoint_epochs.csv")
gain = boolcol(pd.read_csv(DATA / "assisted_v2/gain_vs_K0_by_run.csv"))
ag = boolcol(pd.read_csv(DATA / "assisted_v2/global_by_setting.csv"))
apc = boolcol(pd.read_csv(DATA / "assisted_v2/per_condition_val_best.csv"))
acost = pd.read_csv(DATA / "assisted_v2/training_costs_by_run.csv")
aeval = pd.read_csv(DATA / "assisted_v2/eval_costs_by_task.csv")
acomp = boolcol(pd.read_csv(DATA / "assisted_v2/completeness.csv"))
o5 = pd.read_csv(DATA / "data_quantity/global_by_setting.csv")
anchors = json.load(open(DATA / "anchors/month2_temperature_anchors_nested_v1.json"))

ROLE = {"val": "val_eval", "seen": "seen_train_eval", "ref": "train_fit"}

# =============================================================== Free (S7)
s7b5 = s7[(s7.checkpoint_kind == "best") & (np.isclose(s7.coverage, 0.05))]


def s7stat(role, fam, p, col, cov=0.05, ck="best"):
    x = s7[(s7.checkpoint_kind == ck) & np.isclose(s7.coverage, cov) & (s7.role == role)
           & (s7.family == fam) & (s7.p_conditioning == p)]
    assert len(x) == 3, (role, fam, p, col, len(x))
    return x[col].mean(), x[col].min(), x[col].max(), x


# Table: free comparison
rows = []
for fam, p in CONFIGS:
    r = {}
    for role in ("val_eval", "seen_train_eval"):
        m, lo, hi, _ = s7stat(role, fam, p, "rmse_K")
        r[role] = (m, lo, hi)
        for c in ("centered_rmse_K", "abs_bias_K", "mean_signed_error_K", "jet_rmse_K", "high_rmse_K", "high_iou"):
            r[(role, c)] = s7stat(role, fam, p, c)[0]
    r["ref"] = s7stat("train_fit", fam, p, "rmse_K")[0]
    rows.append(((fam, p), r))
    lab = cfg_plain(fam, p)
    note(f"free VAL RMSE {lab} (full field, 5%, best)", f"{fmt(r['val_eval'][0])} [{fmt(r['val_eval'][1])}-{fmt(r['val_eval'][2])}]", "free_s7/global_by_setting.csv")
    note(f"free seen RMSE {lab}", f"{fmt(r['seen_train_eval'][0])} [{fmt(r['seen_train_eval'][1])}-{fmt(r['seen_train_eval'][2])}]", "free_s7/global_by_setting.csv")
    for c in ("centered_rmse_K", "abs_bias_K", "mean_signed_error_K", "jet_rmse_K", "high_rmse_K", "high_iou"):
        note(f"free VAL {c} {lab}", fmt(r[('val_eval', c)], 3), "free_s7/global_by_setting.csv")
        note(f"free seen {c} {lab}", fmt(r[('seen_train_eval', c)], 3), "free_s7/global_by_setting.csv")
    note(f"free reference760 RMSE {lab}", fmt(r["ref"]), "free_s7/global_by_setting.csv")

with open(TAB / "free_summary.tex", "w") as f:
    for (fam, p), r in rows:
        v = r["val_eval"]
        s = r["seen_train_eval"]
        f.write(
            f"{FAM[fam]} & {'yes' if p else 'no'} & {v[0]:.1f} {{\\scriptsize[{v[1]:.1f}--{v[2]:.1f}]}} & "
            f"{r[('val_eval','centered_rmse_K')]:.1f} & {r[('val_eval','abs_bias_K')]:.1f} & "
            f"{r[('val_eval','mean_signed_error_K')]:+.1f} & {r[('val_eval','jet_rmse_K')]:.0f} & "
            f"{r[('val_eval','high_iou')]:.2f} & "
            f"{s[0]:.1f} {{\\scriptsize[{s[1]:.1f}--{s[2]:.1f}]}} & {r[('seen_train_eval','centered_rmse_K')]:.1f} \\\\\n"
        )
    f.write("\\midrule\n")
    bl = {("GappyPOD_free", True): "GappyPOD ($R{=}32$)", ("GappyPOD_free", False): "GappyPOD ($R{=}32$)",
          ("ParamConditionedMean", True): "Parametric mean", ("GlobalTrainMean", False): "Global mean"}
    for (meth, p), lab in bl.items():
        v = base[(base.method == meth) & (base.p_conditioning == p) & (base.role == "val_eval") & np.isclose(base.site_fraction, 0.05)]
        s = base[(base.method == meth) & (base.p_conditioning == p) & (base.role == "seen_train_eval") & np.isclose(base.site_fraction, 0.05)]
        assert len(v) == 1 and len(s) == 1
        v, s = v.iloc[0], s.iloc[0]
        f.write(f"{lab} & {'yes' if p else 'no'} & {v.rmse_K:.1f} & {v.centered_rmse_K:.1f} & {v.abs_bias_K:.1f} & -- & -- & -- & {s.rmse_K:.1f} & {s.centered_rmse_K:.1f} \\\\\n")
        note(f"baseline C149 VAL RMSE {meth} p={p}", fmt(v.rmse_K), "free_s7/baselines_c149_macro.csv")
        note(f"baseline C149 seen RMSE {meth} p={p}", fmt(s.rmse_K), "free_s7/baselines_c149_macro.csv")

# Coverage
for fam, p in CONFIGS:
    for cov in (0.01, 0.2):
        for role in ("val_eval", "seen_train_eval"):
            m, lo, hi, _ = s7stat(role, fam, p, "rmse_K", cov=cov)
            note(f"free {role} RMSE cov {cov} {cfg_plain(fam,p)}", f"{fmt(m)} [{fmt(lo)}-{fmt(hi)}]", "free_s7/global_by_setting.csv")
            m2 = s7stat(role, fam, p, "mean_signed_error_K", cov=cov)[0]
            note(f"free {role} signed bias cov {cov} {cfg_plain(fam,p)}", fmt(m2), "free_s7/global_by_setting.csv")
    for ck in ("last",):
        for role in ("val_eval", "seen_train_eval", "train_fit"):
            m = s7stat(role, fam, p, "rmse_K", ck=ck)[0]
            note(f"free {role} RMSE 5% {ck} {cfg_plain(fam,p)}", fmt(m), "free_s7/global_by_setting.csv")

# Per seed VAL table (appendix)
with open(TAB / "free_per_seed.tex", "w") as f:
    for fam, p in CONFIGS:
        for sd in (0, 1, 2):
            def g(role, col, cov=0.05):
                x = s7[(s7.checkpoint_kind == "best") & np.isclose(s7.coverage, cov) & (s7.role == role)
                       & (s7.family == fam) & (s7.p_conditioning == p) & (s7.model_seed == sd)]
                assert len(x) == 1
                return x.iloc[0][col]
            ep = g("val_eval", "epoch")
            stop = s7runs[(s7runs.family == fam) & (s7runs.p_conditioning == p) & (s7runs.model_seed == sd)].iloc[0]
            f.write(f"{FAM[fam]} & {'yes' if p else 'no'} & {sd} & {int(ep)} & {int(stop.epochs_completed)} & "
                    f"{g('val_eval','rmse_K'):.2f} & {g('val_eval','centered_rmse_K'):.2f} & {g('val_eval','mean_signed_error_K'):+.2f} & "
                    f"{g('val_eval','high_iou'):.3f} & {g('seen_train_eval','rmse_K'):.2f} & {g('train_fit','rmse_K'):.2f} & "
                    f"{g('val_eval','rmse_K',0.01):.1f} & {g('val_eval','rmse_K',0.2):.1f} \\\\\n")
        if (fam, p) != CONFIGS[-1]:
            f.write("\\addlinespace[2pt]\n")

# Per-condition VAL (free, best, 5%) table rows and figure
pc5 = s7pc[(s7pc.checkpoint_kind == "best") & np.isclose(s7pc.coverage, 0.05) & (s7pc.role == "val_eval")]
with open(TAB / "free_per_condition.tex", "w") as f:
    for fam, p in CONFIGS:
        cells = []
        for c in VALC:
            x = pc5[(pc5.family == fam) & (pc5.p_conditioning == p) & (pc5.sim_id == c)]
            assert len(x) == 3
            cells.append(f"{x.rmse_K.mean():.1f} & {x.mean_signed_error_K.mean():+.1f} {{\\scriptsize[{x.mean_signed_error_K.min():+.0f},{x.mean_signed_error_K.max():+.0f}]}} & {x.jet_rmse_K.mean():.0f} & {x.high_iou.mean():.2f}")
            for col in ("rmse_K", "mean_signed_error_K", "centered_rmse_K", "jet_rmse_K", "high_iou"):
                note(f"free VAL {c} {col} {cfg_plain(fam,p)} mean[min,max]", f"{fmt(x[col].mean(),3)} [{fmt(x[col].min(),3)},{fmt(x[col].max(),3)}]", "free_s7/per_condition.csv")
        f.write(f"{FAM[fam]} & {'yes' if p else 'no'} & " + " & ".join(cells) + " \\\\\n")

# Snapshot swap
for fam, p in CONFIGS:
    for role in ("val_eval", "seen_train_eval"):
        x = swap[(swap.family == fam) & (swap.p_conditioning == p) & (swap.role == role)]
        note(f"swap {role} {cfg_plain(fam,p)} aligned/swapped/delta(min-max)",
             f"{fmt(x.aligned_macro_K.mean())}/{fmt(x.swapped_macro_K.mean())}/{fmt(x.delta_swap_minus_aligned_K.min())}-{fmt(x.delta_swap_minus_aligned_K.max())}",
             "free_s7/swap_neural_by_run.csv")
note("swap donor reference VAL / seen", f"{fmt(swap[swap.role=='val_eval'].donor_reference_macro_K.iloc[0])} / {fmt(swap[swap.role=='seen_train_eval'].donor_reference_macro_K.iloc[0])}", "free_s7/swap_neural_by_run.csv")

# ---------- Figure: free comparison
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
ax = axs[0]
xt = []
for i, (fam, p) in enumerate(CONFIGS):
    for j, role in enumerate(("val_eval", "seen_train_eval")):
        _, _, _, x = s7stat(role, fam, p, "rmse_K")
        xpos = i + (j - 0.5) * 0.32
        mk = "o" if role == "val_eval" else "s"
        ax.scatter([xpos] * 3, x.rmse_K, s=16, marker=mk, facecolor=COL[fam] if p else "white", edgecolor=COL[fam], zorder=3, lw=1)
        ax.hlines(x.rmse_K.mean(), xpos - 0.12, xpos + 0.12, color="k", lw=1.2, zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels([f"{FAM[f]}\n{'with' if p else 'without'} $p$" for f, p in CONFIGS])
ax.set_ylabel("macro RMSE (K)")
ax.set_title("(a) 5% velocity, best checkpoint")
ax.scatter([], [], marker="o", facecolor="grey", edgecolor="grey", s=16, label="VAL (3 held-out cond.)")
ax.scatter([], [], marker="s", facecolor="grey", edgecolor="grey", s=16, label="seen (later frames, TRAIN cond.)")
ax.legend(loc="upper left", frameon=False, handletextpad=0.2)
ax.set_ylim(36, 63)

ax = axs[1]
for fam, p in CONFIGS:
    x = s7b5[(s7b5.role == "val_eval") & (s7b5.family == fam) & (s7b5.p_conditioning == p)]
    ax.scatter(x.abs_bias_K, x.centered_rmse_K, s=20, marker="o", facecolor=COL[fam] if p else "white", edgecolor=COL[fam], lw=1, label=cfg_label(fam, p))
ax.set_xlabel("mean per-frame |bias| (K)")
ax.set_ylabel("centered RMSE (K)")
ax.set_title("(b) level vs structure, VAL")
ax.legend(frameon=False, loc="center left", fontsize=6.2)

ax = axs[2]
covs = [0.01, 0.05, 0.2]
for fam, p in CONFIGS:
    ms, los, his = [], [], []
    for cov in covs:
        m, lo, hi, _ = s7stat("val_eval", fam, p, "rmse_K", cov=cov)
        ms.append(m); los.append(lo); his.append(hi)
    xs = np.array([1, 5, 20]) * (1 + 0.04 * (CONFIGS.index((fam, p)) - 1.5))
    ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms], color=COL[fam], ls="-" if p else "--",
                marker="o", mfc=COL[fam] if p else "white", ms=4, lw=1, capsize=2)
ax.set_xscale("log")
ax.set_xticks([1, 5, 20]); ax.set_xticklabels(["1%", "5%\n(train)", "20%"])
ax.set_ylabel("VAL macro RMSE (K)")
ax.set_title("(c) observed-velocity fraction")
fig.tight_layout(w_pad=1.0)
fig.savefig(FIG / "free_comparison.pdf")
plt.close(fig)

# ---------- Figure: per-condition VAL (free)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
metrics = [("rmse_K", "RMSE (K)"), ("mean_signed_error_K", "signed bias, pred $-$ true (K)"), ("jet_rmse_K", "RMSE in jets, $T<1800$ K (K)")]
for ax, (col, lab) in zip(axs, metrics):
    for ci, c in enumerate(VALC):
        for k, (fam, p) in enumerate(CONFIGS):
            x = pc5[(pc5.family == fam) & (pc5.p_conditioning == p) & (pc5.sim_id == c)]
            xpos = ci + (k - 1.5) * 0.18
            ax.scatter([xpos] * 3, x[col], s=10, facecolor=COL[fam] if p else "white", edgecolor=COL[fam], lw=0.9, zorder=3)
            ax.hlines(x[col].mean(), xpos - 0.07, xpos + 0.07, color="k", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([VALLAB[c].replace(", ", "\n") for c in VALC])
    ax.set_ylabel(lab)
    if col == "mean_signed_error_K":
        ax.axhline(0, color="k", lw=0.6)
for fam, p in CONFIGS:
    axs[0].scatter([], [], s=12, facecolor=COL[fam] if p else "white", edgecolor=COL[fam], label=cfg_label(fam, p))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(w_pad=0.8)
fig.savefig(FIG / "free_val_conditions.pdf")
plt.close(fig)

# =============================================================== Data quantity (O5)
o5["pool"] = o5.run_id.str.extract(r"O-(B19|A38|C149)")[0]
o5["p_on"] = o5.run_id.str.contains("-on-")
o5["seed"] = o5.run_id.str.extract(r"-s(\d)$")[0].astype(int)
npool = {"B19": 19, "A38": 38, "C149": 149}
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.2), sharey=False)
for ax, role, ttl in ((axs[0], "seen_train_eval", "(a) seen: later frames of TRAIN conditions"), (axs[1], "val_eval", "(b) VAL: three held-out conditions")):
    for p_on in (True, False):
        for ck, ls in (("best", "-"), ("last", ":")):
            x = o5[(o5.role == role) & np.isclose(o5.coverage, 0.05) & (o5.checkpoint_kind == ck) & (o5.p_on == p_on) & (o5.seed == 0)].copy()
            x["n"] = x.pool.map(npool)
            x = x.sort_values("n")
            ax.plot(x.n, x.rmse_K, ls=ls, marker="o", color=COL["sparse_unet"] if p_on else "#6aa0d8",
                    mfc=COL["sparse_unet"] if p_on else "white", ms=4, lw=1,
                    label=f"{'with' if p_on else 'without'} $p$, {ck}")
            for _, r in x.iterrows():
                note(f"O5 {role} {r.run_id} {ck} RMSE", fmt(r.rmse_K), "data_quantity/global_by_setting.csv")
        x1 = o5[(o5.role == role) & np.isclose(o5.coverage, 0.05) & (o5.checkpoint_kind == "best") & (o5.p_on == p_on) & (o5.seed == 1)]
        ax.scatter([38] * len(x1), x1.rmse_K, marker="x", color=COL["sparse_unet"] if p_on else "#6aa0d8", s=22, zorder=4,
                   label=f"{'with' if p_on else 'without'} $p$, seed 1 (38 only)")
        for _, r in x1.iterrows():
            note(f"O5 {role} {r.run_id} best RMSE", fmt(r.rmse_K), "data_quantity/global_by_setting.csv")
    ax.set_xscale("log"); ax.set_xticks([19, 38, 149]); ax.set_xticklabels(["19", "38", "149"])
    ax.minorticks_off()
    ax.set_xlabel("training snapshots per TRAIN condition")
    ax.set_ylabel("macro RMSE (K)")
    ax.set_title(ttl)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=6.3)
fig.tight_layout(w_pad=1.2)
fig.savefig(FIG / "data_quantity.pdf")
plt.close(fig)

# =============================================================== K sweep and joint comparison
swp = sw[(sw.layout_kind == "primary")]


def sweep_curve(method, p, treat, role, ck="best"):
    cks = ["best", "fitted"] if ck == "best" else [ck]
    x = swp[(swp.method == method) & (swp.p_conditioning == p) & (swp.treatment == treat) & (swp.role == role) & swp.checkpoint.isin(cks)]
    per_seed = x.groupby(["model_seed", "K"]).common_rmse_K.mean()
    g = per_seed.groupby("K")
    return g.mean(), g.min(), g.max()


def joint_seed(fam, p, est, role, ck="best", dom="common"):
    x = js[(js.family == fam) & (js.p_conditioning == p) & (js.role == role) & (js.checkpoint == ck)]
    g = x.groupby("K")[f"{dom}_{est}"]
    return g.mean(), g.min(), g.max()


# Cross-check: joint free/median equal sweep free/median
for fam, p in CONFIGS:
    for role in ("val_eval", "seen_train_eval"):
        for est in ("free", "median", "mean"):
            a = joint_seed(fam, p, est, role)[0]
            b = sweep_curve(fam, p, est, role)[0]
            assert np.allclose(a.values, b.values, atol=1e-9), (fam, p, est, role)

# Main joint table (VAL and seen)
with open(TAB / "joint_main.tex", "w") as f:
    for role in ("val_eval", "seen_train_eval"):
        f.write(f"\\multicolumn{{11}}{{l}}{{\\textit{{{'VAL (three held-out conditions)' if role=='val_eval' else 'seen (later frames of the 20 TRAIN conditions)'}}}}}\\\\\n")
        for fam, p in CONFIGS:
            fr = joint_seed(fam, p, "free", role)[0]
            md, mdlo, mdhi = joint_seed(fam, p, "median", role)
            mn = joint_seed(fam, p, "mean", role)[0]
            ao = sweep_curve(fam, p, "ao", role)[0]
            orc = sweep_curve(fam, p, "oracle", role)[0]
            asd, aslo, ashi = joint_seed(fam, p, "assisted", role)
            f.write(f"{FAM[fam]} & {'yes' if p else 'no'} & {fr[0]:.2f} & {mn[1]:.2f} & {md[4]:.2f} & {md[32]:.2f} & {ao[32]:.2f} & {orc[0]:.2f} & "
                    f"{asd[0]:.2f} & {asd[32]:.2f} & {asd[32]-asd[0]:+.2f} \\\\\n")
            lab = cfg_plain(fam, p)
            for K in KS:
                note(f"{role} common {lab} K={K} free/mean/median/AO/assisted",
                     f"{fmt(fr[K])}/{fmt(mn[K])}/{fmt(md[K])}/{fmt(ao[K])}/{fmt(asd[K])}", "joint/joint_by_seed_K.csv; k_sweep/by_seed.csv (AO)")
            note(f"{role} common {lab} oracle", fmt(orc[0]), "k_sweep/by_seed.csv")
            note(f"{role} common {lab} median K=4 seed range", f"{fmt(mdlo[4])}-{fmt(mdhi[4])}", "joint/joint_by_seed_K.csv")
            note(f"{role} common {lab} median K=32 seed range", f"{fmt(mdlo[32])}-{fmt(mdhi[32])}", "joint/joint_by_seed_K.csv")
            note(f"{role} common {lab} assisted K=0 / K=32 seed range", f"{fmt(aslo[0])}-{fmt(ashi[0])} / {fmt(aslo[32])}-{fmt(ashi[32])}", "joint/joint_by_seed_K.csv")
            frs = joint_seed(fam, p, "free", role)
            note(f"{role} common {lab} free seed range", f"{fmt(frs[1][0])}-{fmt(frs[2][0])}", "joint/joint_by_seed_K.csv")
        if role == "val_eval":
            f.write("\\midrule\n")

# Baseline sweep table rows
with open(TAB / "baselines_K.tex", "w") as f:
    for role in ("val_eval", "seen_train_eval"):
        f.write(f"\\multicolumn{{8}}{{l}}{{\\textit{{{'VAL' if role=='val_eval' else 'seen'}}}}}\\\\\n")
        for meth, p, treat, lab in (("GappyPOD_free", True, "median", "GappyPOD with $p$ + median"),
                                    ("GappyPOD_free", True, "tblock", "GappyPOD with $p$ + T-block"),
                                    ("GappyPOD_free", False, "median", "GappyPOD without $p$ + median"),
                                    ("GappyPOD_free", False, "tblock", "GappyPOD without $p$ + T-block"),
                                    ("ParamConditionedMean", True, "median", "Parametric mean + median"),
                                    ("GlobalTrainMean", False, "median", "Global mean + median")):
            c = sweep_curve(meth, p, treat, role)[0]
            orc = sweep_curve(meth, p, "oracle", role)[0][0]
            f.write(f"{lab} & " + " & ".join(f"{c[K]:.1f}" for K in KS) + f" & {orc:.1f} \\\\\n")
            note(f"baseline {role} {meth} p={p} {treat} K-curve", "/".join(fmt(c[K]) for K in KS) + f"; oracle {fmt(orc)}", "k_sweep/by_seed.csv")
        if role == "val_eval":
            f.write("\\midrule\n")

# ---------- Figure: RMSE vs K, four configurations, VAL and seen
fig, axs = plt.subplots(2, 4, figsize=(7.1, 4.4), sharex=True)
xk = np.array(KS)
xplot = np.array([0, 1, 2, 3, 4, 5])  # categorical spacing
for col, (fam, p) in enumerate(CONFIGS):
    for row, role in enumerate(("val_eval", "seen_train_eval")):
        ax = axs[row, col]
        fr = joint_seed(fam, p, "free", role)
        ax.plot(xplot, fr[0].values, color="k", lw=1.1, label="free (no T)")
        ax.fill_between(xplot, fr[1].values, fr[2].values, color="k", alpha=0.08, lw=0)
        orc = sweep_curve(fam, p, "oracle", role)[0][0]
        ax.axhline(orc, color="0.4", lw=0.8, ls=(0, (1, 1.5)), label="free + oracle shift")
        ao = sweep_curve(fam, p, "ao", role)[0]
        ax.plot(xplot, ao.values, color="#7b8a2e", lw=0.9, ls="--", marker="^", ms=2.5, label=r"free + AO ($\tau{=}1$)")
        mn = joint_seed(fam, p, "mean", role)[0]
        ax.plot(xplot, mn.values, color="#9b59b6", lw=0.9, ls="-.", marker="v", ms=2.5, label="free + mean")
        md = joint_seed(fam, p, "median", role)
        ax.plot(xplot, md[0].values, color="#2e8b57", lw=1.3, marker="o", ms=3, label="free + median")
        ax.fill_between(xplot, md[1].values, md[2].values, color="#2e8b57", alpha=0.15, lw=0)
        asd = joint_seed(fam, p, "assisted", role)
        ax.plot(xplot, asd[0].values, color="#c0392b", lw=1.3, marker="s", ms=3, label="assisted (T as input)")
        ax.fill_between(xplot, asd[1].values, asd[2].values, color="#c0392b", alpha=0.15, lw=0)
        lo = min(orc, md[1].min(), asd[1].min()) - 1.0
        hi = max(fr[2].max(), asd[2].max()) + 2.5
        ax.set_ylim(lo, hi)
        ax.set_xticks(xplot); ax.set_xticklabels([str(k) for k in KS])
        if row == 0:
            ax.set_title(cfg_label(fam, p))
        if row == 1:
            ax.set_xlabel("K (temperature sites)")
        if col == 0:
            ax.set_ylabel(("VAL" if role == "val_eval" else "seen") + " RMSE (K)")
        # list off-scale K=1 values in one box
        off = []
        if md[0][1] > hi:
            off.append(f"mean = median: {md[0][1]:.1f}")
        if ao[1] > hi:
            off.append(f"AO: {ao[1]:.1f}")
        if off:
            ax.text(0.97, 0.97, "K=1 off scale\n" + "\n".join(off), transform=ax.transAxes, ha="right", va="top", fontsize=5.6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4), zorder=10)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.035), fontsize=6.5)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=0.5)
fig.savefig(FIG / "rmse_vs_K.pdf")
plt.close(fig)

# ---------- Figure: K=1 per replica
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

fig, ax = plt.subplots(1, 1, figsize=(5.2, 2.4))
w = 0.26
RC = ["#9ecae1", "#6baed6", "#e6550d"]
for i, (fam, p) in enumerate(CONFIGS):
    for r in (0, 1, 2):
        sel = (jr.family == fam) & (jr.p_conditioning == p) & (jr.role == "val_eval") & (jr.checkpoint == "best") & (jr.mask_replica_id == r)
        x = jr[sel & (jr.K == 1)].sort_values("model_seed")
        x0 = jr[sel & (jr.K == 0)].sort_values("model_seed")
        assert len(x) == 3 and len(x0) == 3
        d_med = x.common_median_minus_free
        d_ass = x.common_assisted.values - x0.common_assisted.values
        xpos = i + (r - 1) * w
        ax.bar(xpos, d_med.mean(), width=w * 0.9, color=RC[r], edgecolor="k", lw=0.4)
        ax.scatter([xpos] * 3, d_med, s=6, color="k", zorder=3)
        ax.scatter([xpos] * 3, d_ass, s=14, marker="D", facecolor="#c0392b", edgecolor="white", lw=0.4, zorder=4)
        note(f"K=1 VAL replica {r} {cfg_plain(fam,p)} median-free (common)", f"{fmt(d_med.mean())} [{fmt(d_med.min())},{fmt(d_med.max())}]", "joint/joint_by_run_replica_K.csv")
        note(f"K=1 VAL replica {r} {cfg_plain(fam,p)} assisted K1-K0 (common)", f"{fmt(d_ass.mean())} [{fmt(d_ass.min())},{fmt(d_ass.max())}]", "joint/joint_by_run_replica_K.csv")
        note(f"K=1 VAL replica {r} {cfg_plain(fam,p)} free / median abs (common)", f"{fmt(x.common_free.mean())} / {fmt(x.common_median.mean())}", "joint/joint_by_run_replica_K.csv")
ax.axhline(0, color="k", lw=0.6)
ax.set_yscale("symlog", linthresh=5)
ax.set_yticks([-5, 0, 5, 10, 50, 100]); ax.set_yticklabels(["$-5$", "0", "5", "10", "50", "100"])
ax.set_xticks(range(4)); ax.set_xticklabels([f"{FAM[f]}\n{'with' if p else 'without'} $p$" for f, p in CONFIGS])
ax.set_ylabel("RMSE change at K=1 (K)")
handles = [Patch(facecolor=RC[r], edgecolor="k", lw=0.4, label=f"layout {r}: free + one-site offset") for r in range(3)]
handles.append(Line2D([], [], marker="o", ls="", color="k", ms=2.5, label="individual seeds (offset)"))
handles.append(Line2D([], [], marker="D", ls="", markerfacecolor="#c0392b", markeredgecolor="white", ms=4, label="assisted model, K=1 vs its K=0"))
ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig.savefig(FIG / "k1_by_layout.pdf")
plt.close(fig)

# ---------- Figure: assisted gain vs own K=0
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
gb = gain[gain.checkpoint_kind == "best"]
for ax, role, ttl in ((axs[0], "val_eval", "(a) VAL"), (axs[1], "seen_train_eval", "(b) seen")):
    for fam, p in CONFIGS:
        for sd in (0, 1, 2):
            x = gb[(gb.family == fam) & (gb.p_conditioning == p) & (gb.model_seed == sd) & (gb.role == role)].sort_values("K")
            ax.plot(xplot, x.common_rmse_K_minus_K0.values, color=COL[fam], ls="-" if p else "--", lw=0.9,
                    marker="o", ms=2.5, mfc=COL[fam] if p else "white", label=cfg_label(fam, p) if sd == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xplot); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K (temperature sites)")
    ax.set_ylabel("RMSE change vs own K=0 (K)")
    ax.set_title(ttl)
axs[0].legend(frameon=False, fontsize=6, loc="lower left")
ax = axs[2]
for i, (fam, p) in enumerate(CONFIGS):
    for j, role in enumerate(("val_eval", "seen_train_eval")):
        x = gb[(gb.family == fam) & (gb.p_conditioning == p) & (gb.role == role) & (gb.K == 32)]
        assert len(x) == 3
        xpos = i + (j - 0.5) * 0.34
        ax.scatter([xpos] * 3, x.common_centered_rmse_K_minus_K0, s=14, marker="o" if role == "val_eval" else "s",
                   facecolor=COL[fam] if p else "white", edgecolor=COL[fam], lw=0.9, zorder=3)
        ax.hlines(x.common_centered_rmse_K_minus_K0.mean(), xpos - 0.12, xpos + 0.12, color="k", lw=1)
        lab = cfg_plain(fam, p)
        for col in ("common_rmse_K_minus_K0", "common_centered_rmse_K_minus_K0"):
            note(f"assisted gain K=32 {role} {lab} {col} mean[min,max]", f"{fmt(x[col].mean())} [{fmt(x[col].min())},{fmt(x[col].max())}]", "assisted_v2/gain_vs_K0_by_run.csv")
        for K in (1, 4):
            y = gb[(gb.family == fam) & (gb.p_conditioning == p) & (gb.role == role) & (gb.K == K)]
            note(f"assisted gain K={K} {role} {lab} common mean[min,max]", f"{fmt(y.common_rmse_K_minus_K0.mean())} [{fmt(y.common_rmse_K_minus_K0.min())},{fmt(y.common_rmse_K_minus_K0.max())}]", "assisted_v2/gain_vs_K0_by_run.csv")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{FAM[f]}\n{'with' if p else 'w/o'} $p$" for f, p in CONFIGS])
ax.set_ylabel("centered-RMSE change (K)")
ax.set_title("(c) K=32 vs K=0, centered part")
fig.tight_layout(w_pad=0.9)
fig.savefig(FIG / "assisted_gain.pdf")
plt.close(fig)

# Assisted level/structure details (VAL, best) from family dispersion: bias and centered at K=0/32
for fam, p in CONFIGS:
    for role in ("val_eval", "seen_train_eval"):
        x = ag[(ag.family == fam) & (ag.p_conditioning == p) & (ag.role == role) & (ag.checkpoint_kind == "best")]
        for K in (0, 32):
            y = x[x.K == K]
            assert len(y) == 3
            note(f"assisted {role} {cfg_plain(fam,p)} K={K} common bias / centered / full hot / jet",
                 f"{fmt(y.common_bias_K.mean())} / {fmt(y.common_centered_rmse_K.mean())} / {fmt(y.high_rmse_K.mean())} / {fmt(y.jet_rmse_K.mean())}",
                 "assisted_v2/global_by_setting.csv")

# Regional table at K=32, VAL: free / median / assisted (full-field regions)
with open(TAB / "regional_K32.tex", "w") as f:
    for fam, p in CONFIGS:
        def swreg(treat, K, col):
            x = swp[(swp.method == fam) & (swp.p_conditioning == p) & (swp.treatment == treat) & (swp.role == "val_eval") & (swp.checkpoint == "best") & (swp.K == K)]
            return x.groupby("model_seed")[col].mean().mean()
        y = ag[(ag.family == fam) & (ag.p_conditioning == p) & (ag.role == "val_eval") & (ag.checkpoint_kind == "best")]
        a0, a32 = y[y.K == 0], y[y.K == 32]
        vals = [swreg("free", 0, "full_hot_rmse_K"), swreg("median", 32, "full_hot_rmse_K"), a0.high_rmse_K.mean(), a32.high_rmse_K.mean(),
                swreg("free", 0, "full_jet_rmse_K"), swreg("median", 32, "full_jet_rmse_K"), a0.jet_rmse_K.mean(), a32.jet_rmse_K.mean(),
                swreg("free", 0, "full_centered_rmse_K"), swreg("median", 32, "full_centered_rmse_K"), a0.full_centered_rmse_K.mean(), a32.full_centered_rmse_K.mean()]
        f.write(f"{FAM[fam]} & {'yes' if p else 'no'} & " + " & ".join(f"{v:.1f}" for v in vals) + " \\\\\n")
        note(f"regional VAL {cfg_plain(fam,p)} hot free/med32/ass0/ass32 | jet ... | centered ...", " / ".join(fmt(v) for v in vals), "k_sweep/by_seed.csv; assisted_v2/global_by_setting.csv")

# Per-condition VAL table: free vs median K=4 / K=32 vs assisted K=32 (common domain)
with open(TAB / "per_condition_K.tex", "w") as f:
    for fam, p in CONFIGS:
        cells = []
        for c in VALC:
            def pcv(treat, K):
                x = swpc[(swpc.method == fam) & (swpc.p_conditioning == p) & (swpc.treatment == treat) & (swpc.K == K) & (swpc.sim_id == c)]
                assert len(x) == 9, len(x)
                return x.groupby("model_seed").common_rmse_K.mean().mean()
            y = apc[(apc.family == fam) & (apc.p_conditioning == p) & (apc.sim_id == c)]
            a0 = y[y.K == 0].common_rmse_K.mean(); a32 = y[y.K == 32].common_rmse_K.mean()
            v = [pcv("free", 0), pcv("median", 4), pcv("median", 32), a0, a32]
            cells.append(" & ".join(f"{t:.1f}" for t in v))
            note(f"per-condition VAL common {c} {cfg_plain(fam,p)} free/med4/med32/ass0/ass32", " / ".join(fmt(t) for t in v), "k_sweep/per_condition_val_best_primary.csv; assisted_v2/per_condition_val_best.csv")
        f.write(f"{FAM[fam]} & {'yes' if p else 'no'} & " + " & ".join(cells) + " \\\\\n")

# Seen worsening counts for median (best, neural rows) at each K
for role in ("val_eval", "seen_train_eval"):
    for treat in ("median", "mean", "ao"):
        out = []
        for K in KS[1:]:
            fr = swp[(swp.treatment == "free") & (swp.role == role) & (swp.checkpoint == "best") & (swp.K == K)].set_index(["method", "p_conditioning", "model_seed", "mask_replica_id"]).full_rmse_K
            tr = swp[(swp.treatment == treat) & (swp.role == role) & (swp.checkpoint == "best") & (swp.K == K)].set_index(["method", "p_conditioning", "model_seed", "mask_replica_id"]).full_rmse_K
            d = (tr - fr).dropna()
            out.append(f"K={K}: {(d > 0).sum()}/{len(d)}")
        note(f"neural best rows worse than free (full field) {role} {treat}", ", ".join(out), "k_sweep/by_seed.csv")

# Assisted vs free+median per seed at K=32, VAL and seen
for role in ("val_eval", "seen_train_eval"):
    for fam, p in CONFIGS:
        x = js[(js.family == fam) & (js.p_conditioning == p) & (js.role == role) & (js.checkpoint == "best") & (js.K == 32)]
        d = x.common_assisted - x.common_median
        note(f"assisted - free+median K=32 {role} {cfg_plain(fam,p)} per seed", ", ".join(fmt(v) for v in d.values) + f" (n seeds assisted lower: {(d<0).sum()})", "joint/joint_by_seed_K.csv")
        x0 = js[(js.family == fam) & (js.p_conditioning == p) & (js.role == role) & (js.checkpoint == "best") & (js.K == 0)]
        d0 = x0.common_assisted.values - x0.common_free.values
        note(f"assisted(K=0) - free {role} {cfg_plain(fam,p)} per seed", ", ".join(fmt(v) for v in d0), "joint/joint_by_seed_K.csv")

# =============================================================== Costs
with open(TAB / "costs.tex", "w") as f:
    def rng(v):
        return f"{min(v):.1f}--{max(v):.1f}"
    for camp, df, pref in (("Temperature-free (12 runs)", s7cost, ""), ("Temperature as input (12 runs)", acost, "TA-")):
        for fam_key, fam_name in (("unet", "U-Net"), ("glu", "GLU")):
            if camp.startswith("Temperature-free"):
                sel = df[df.run_id.str.contains("glu" if fam_key == "glu" else "unet|O-C149")]
            else:
                sel = df[df.run_id.str.contains(fam_key)]
            f.write(f"{camp if fam_key=='unet' else ''} & {fam_name} & {len(sel)} & {rng(sel.compute_h)} & {rng(sel.queue_wait_h)} & {int(sel.n_attempts.min())}--{int(sel.n_attempts.max())} \\\\\n")
            note(f"cost {camp} {fam_name} compute h range / queue h range / attempts", f"{rng(sel.compute_h)} / {rng(sel.queue_wait_h)} / {sel.n_attempts.min()}-{sel.n_attempts.max()}", "free_s7/training_costs_by_run.csv; assisted_v2/training_costs_by_run.csv")
        if camp.startswith("Temperature-free"):
            f.write("\\addlinespace[2pt]\n")
note("S7 final evaluation compute per run (min)", f"{s7eval.compute_s.min()/60:.0f}-{s7eval.compute_s.max()/60:.0f}", "free_s7/eval_costs_by_task.csv")
note("assisted final evaluation compute per run (min)", f"{aeval.compute_s.min()/60:.1f}-{aeval.compute_s.max()/60:.1f}", "assisted_v2/eval_costs_by_task.csv")

# Run matrix (appendix)
with open(TAB / "run_matrix.tex", "w") as f:
    for fam, p in CONFIGS:
        for sd in (0, 1, 2):
            st = s7runs[(s7runs.family == fam) & (s7runs.p_conditioning == p) & (s7runs.model_seed == sd)].iloc[0]
            bep = s7[(s7.family == fam) & (s7.p_conditioning == p) & (s7.model_seed == sd) & (s7.checkpoint_kind == "best")].epoch.iloc[0]
            lep = s7[(s7.family == fam) & (s7.p_conditioning == p) & (s7.model_seed == sd) & (s7.checkpoint_kind == "last")].epoch.iloc[0]
            c = s7cost[s7cost.run_id == st.run_id].iloc[0]
            tag = "TA-" + ("unet" if fam == "sparse_unet" else "glu") + ("-on" if p else "-off") + f"-s{sd}"
            ac = acomp[acomp.run_id == tag].iloc[0]
            ab = int(jep[(jep.run_id == tag) & (jep.checkpoint == "best")].epoch.iloc[0])
            al = int(jep[(jep.run_id == tag) & (jep.checkpoint == "last")].epoch.iloc[0])
            acc = acost[acost.run_id == tag].iloc[0]
            reuse = "reused$^\\dagger$" if st.reused_not_relaunched else "new"
            ext = "$^\\ddagger$" if st.extended_past_initial_tranche else ""
            ext2 = "$^\\ddagger$" if ac.n_training_attempts > 1 and tag in ("TA-glu-on-s1", "TA-glu-off-s1", "TA-glu-off-s2") else ""
            f.write(f"{FAM[fam]} & {'yes' if p else 'no'} & {sd} & {reuse} & {int(bep)} & {int(lep)}{ext} & {c.compute_h:.1f} & "
                    f"{ab} & {al}{ext2} & {int(ac.n_training_attempts)} & {acc.compute_h:.1f} \\\\\n")
        if (fam, p) != CONFIGS[-1]:
            f.write("\\addlinespace[2pt]\n")

# =============================================================== Observation layouts figure
img = np.asarray(Image.open(DATA / "fields/fields_val_frame1000_best.png").convert("RGB"))
dark = img.sum(2) < 120


def find_boxes():
    """Panel frames of the published field figure (black 1-px boxes)."""
    rows = []
    for y in range(img.shape[0]):
        v = dark[y]
        runs, s = [], None
        for i, b in enumerate(v):
            if b and s is None:
                s = i
            if not b and s is not None:
                if i - s > 100:
                    runs.append((s, i - 1))
                s = None
        if runs:
            rows.append((y, runs))
    tops = []
    for y, runs in rows:
        if not tops or y - tops[-1][0] > 5:
            tops.append((y, runs))
    return tops  # alternating top/bottom lines


lines = find_boxes()
ys = [y for y, _ in lines]
assert len(ys) % 2 == 0
row_boxes = [(ys[2 * i], ys[2 * i + 1], lines[2 * i + 1][1]) for i in range(len(ys) // 2)]
# row 0 = truth (3 panels); rows 1.. = runs (6 panels: T, error per condition)
order = ["truth", "U-Net with p s0", "U-Net with p s1", "U-Net with p s2", "U-Net without p s0", "U-Net without p s1",
         "U-Net without p s2", "GLU with p s0", "GLU with p s1", "GLU with p s2", "GLU without p s0", "GLU without p s1", "GLU without p s2"]
assert len(row_boxes) == 13, len(row_boxes)


def panel(row, idx):
    y0, y1, runs = row_boxes[row]
    x0, x1 = runs[idx]
    return img[y0 + 1:y1, x0 + 1:x1]


truth_cols = {c: i for i, c in enumerate(VALC)}
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35))
lab_c = ["#1f77b4", "#2ca02c", "#d62728"]
for ax, L in zip(axs, anchors["layouts"]):
    p0 = panel(0, truth_cols["0NH3_1.05.h5"])
    h, w = p0.shape[:2]
    ax.imshow(p0, extent=(-0.5, 127.5, 127.5, -0.5), interpolation="nearest")
    iy, ix = np.array(L["iy"]), np.array(L["ix"])
    ax.scatter(ix[4:], iy[4:], s=9, facecolor="none", edgecolor="white", lw=0.7, label="sites 5--32")
    ax.scatter(ix[1:4], iy[1:4], s=16, facecolor="#00e5ff", edgecolor="k", lw=0.5, label="sites 2--4")
    ax.scatter(ix[0], iy[0], s=60, marker="*", facecolor="#ffff33", edgecolor="k", lw=0.6, label="site 1 (K=1)")
    ax.set_title(f"layout {L['replica_id']} (with velocity mask {L['replica_id']})")
    ax.set_xticks([0, 64, 127]); ax.set_yticks([0, 64, 127])
    ax.grid(False)
    ax.set_xlabel("column")
axs[0].set_ylabel("row")
axs[0].legend(loc="lower left", fontsize=5.8, framealpha=0.85, borderpad=0.3, handletextpad=0.2)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG / "anchor_layouts.pdf", dpi=300)
plt.close(fig)

# =============================================================== Field crops figure
def panel_lab(row, idx, top=18):
    """Run panel including the strip above it, which carries the frame RMSE printed in the source figure."""
    y0, y1, runs = row_boxes[row]
    x0, x1 = runs[idx]
    return img[y0 - top:y1 + 1, x0:x1 + 1]


sel_rows = [("truth", 0), ("U-Net with $p$\nseed 0", 1), ("U-Net with $p$\nseed 1", 2), ("GLU with $p$\nseed 0", 7)]
fig = plt.figure(figsize=(6.6, 6.1))
gs = fig.add_gridspec(len(sel_rows), 6, width_ratios=[1, 1, 0.08, 1, 1, 0.95], wspace=0.04, hspace=0.06)
for r, (lab, ri) in enumerate(sel_rows):
    for ci, cond in enumerate(["0NH3_1.05.h5", "40NH3_0.9.h5"]):
        k = VALC.index(cond)
        for e in (0, 1):
            ax = fig.add_subplot(gs[r, ci * 3 + e])
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                if e == 1:
                    ax.axis("off")
                    if ci == 0:
                        ax.text(0.05, 0.5, "frame RMSE of each\nprediction is printed\nabove its error map", fontsize=6.3, va="center", color="0.3", transform=ax.transAxes)
                    continue
                ax.imshow(panel_lab(0, k), interpolation="nearest")
                ax.set_title(VALLAB[cond], fontsize=8)
            else:
                ax.imshow(panel_lab(ri, 2 * k + e), interpolation="nearest")
            if r == 1:
                ax.text(0.5, 1.0, "temperature" if e == 0 else "error (pred $-$ true)", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color="0.25")
            if ci == 0 and e == 0:
                ax.set_ylabel(lab, fontsize=7)
axc = fig.add_subplot(gs[:, 5])
axc.imshow(img[870:1430, 1225:1452], interpolation="lanczos")
axc.axis("off")
fig.savefig(FIG / "fields_level_structure.png", dpi=220)
plt.close(fig)

TRACE_HEADER = """# Numeric trace for the Month 2 report

Every number quoted in `main.tex` comes from the rows below (generated by
`month2_sources/make_figures.py`) or directly from the cited report tables.
Units: kelvin unless stated. "common" = grid minus the 32 sites of the layout
(16352 pixels, identical for every K).

| Quantity | Value | Source file (`month2_sources/data/...`) |
|---|---|---|
"""
(ROOT / "month2_sources" / "NUMBERS.md").write_text(TRACE_HEADER + "\n".join(TRACE) + "\n")
import re  # noqa: E402
for t in TAB.glob("*.tex"):  # avoid printing negative zero
    t.write_text(re.sub(r"(?<![0-9.])[-+]0\.(0+)(?![0-9])", r"0.\1", t.read_text()))
print("ok", len(TRACE), "trace rows")
