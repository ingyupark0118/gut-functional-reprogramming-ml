#!/usr/bin/env python
"""Network metrics of the within-group Spearman correlation networks, re-computed in size-matched sub-samples (Fig. 4E).

For NC, DM and Cirrhosis the metrics (number of solid edges = |rho| > 0.6 and q < 0.05, number of pairs with |rho| > 0.6, mean |rho| of the testable pairs, ...)
are re-computed in 1,000 sub-samples of n = 14 (the size of CDM) drawn without replacement, and compared with the observed CDM value.
Sensitivity analysis: 1,000 sub-samples of n = 13 against the 14 leave-one-out values of CDM.
Random numbers: numpy.random.default_rng(20261005) for the n = 14 sub-samples (groups in the order NC, DM, Cirrhosis; 1,000 draws each, the same sub-sample for both
KO units); default_rng(20261006) for the n = 13 sub-samples; no other random numbers.
Output tables (csv): the percentile summary of the sub-sampling distributions (all metrics; and the KO-sum / n = 14 / three main metrics subset with the observed
CDM rows = figure_values/Fig4E_network_metrics_subsampled.csv), the replicate table, the verdict table and the table of untestable pairs.
The script ends with checks of the tables re-read from disk against the B-3 result tables and against each other; mismatches must be 0."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from fig4_common import load_all, main_family, spearman_pairs, group_matrix, bh_nan, GROUPS, UNITS, NEXP, NODES, LC, PI, PJ, B3, F_COR
STEM = "Fig4F_network_metrics_subsample"; OUT = f"./panels/{STEM}"; os.makedirs(OUT, exist_ok=True)
def note(s): print(s)
SEED14, SEED13, NREP = 20261005, 20261006, 1000; SUBG = ["NC", "DM", "Cirrhosis"]
MET = ["solid_edges", "pairs_absrho_gt_0.6", "mean_absrho", "solid_edges_family173", "pairs_P_lt_0.05", "testable_pairs"]
MAIN = MET[:3]

lab, K, T3, clin, lip = load_all(); D, dev = main_family(lab, K, T3, clin)
note(f"re-derived 1,384 correlations vs T_B3_correlations.csv: max abs deviation {dev}")
LCI = NODES.index(LC); LC_PAIR = (PI == LCI) | (PJ == LCI)
def metrics(X):
    rho, P = spearman_pairs(X); ok = ~np.isnan(rho); a = np.abs(rho); q = bh_nan(P); q173 = bh_nan(np.where(ok, P, 1.0))
    with np.errstate(invalid="ignore"):
        return {"solid_edges": int(((a > 0.6) & (q < 0.05)).sum()), "pairs_absrho_gt_0.6": int((a > 0.6).sum()), "mean_absrho": float(a[ok].mean()),
                "solid_edges_family173": int(((a > 0.6) & (q173 < 0.05)).sum()), "pairs_P_lt_0.05": int((P < 0.05).sum()), "testable_pairs": int(ok.sum()),
                "untestable_pairs": int((~ok).sum()), "untestable_Lcrispatus_pairs": int((~ok & LC_PAIR).sum()), "untestable_other_pairs": int((~ok & ~LC_PAIR).sum())}
XG = {(g, u): group_matrix(g, u, lab, K, T3, clin) for g in GROUPS for u in UNITS}
rows = []
for u in UNITS:                                                   # observed, full sample
    for g in GROUPS: rows.append(dict(design="observed_full_n", group=g, unit=u, n=NEXP[g], rep=0, **metrics(XG[(g, u)])))
for design, n_sub, seed in (("subsample_n14", 14, SEED14), ("subsample_n13_sensitivity", 13, SEED13)):
    rng = np.random.default_rng(seed)
    for g in SUBG:
        for r in range(1, NREP + 1):
            idx = np.sort(rng.choice(NEXP[g], n_sub, replace=False))                  # the same sub-sample for both units
            for u in UNITS: rows.append(dict(design=design, group=g, unit=u, n=n_sub, rep=r, **metrics(XG[(g, u)][idx])))
for u in UNITS:                                                   # leave-one-out on CDM (sensitivity)
    for i in range(14): rows.append(dict(design="CDM_leave_one_out_n13_sensitivity", group="CDM", unit=u, n=13, rep=i + 1, **metrics(np.delete(XG[("CDM", u)], i, axis=0))))
R = pd.DataFrame(rows); OBS = R[R.design == "observed_full_n"].set_index(["unit", "group"])
note("observed (full n):\n" + OBS[MET + ["untestable_pairs"]].to_string())
PCT = [0, 2.5, 25, 50, 75, 97.5, 100]
def dist(design, cdm_ref):
    out = []
    for u in UNITS:
        for m in MET:
            for g in SUBG:
                x = R[(R.design == design) & (R.unit == u) & (R.group == g)][m].values; assert len(x) == NREP
                c = cdm_ref(u, m); p = np.percentile(x, PCT)
                out.append(dict(design=design, unit=u, metric=m, group=g, n_group=NEXP[g], n_subsample=int(design[11:13]), n_replicates=NREP, **{f"p{q:g}": v for q, v in zip(PCT, p)}, mean=x.mean(),
                                prop_ge_CDM=float((x >= c).mean()), n_ge_CDM=int((x >= c).sum()), prop_nonzero=float((x > 0).mean()), CDM_reference=c,
                                observed_full_n=OBS.loc[(u, g), m], **{"CDM_outside_2.5_97.5": bool(c < p[1] or c > p[5])}))
    return pd.DataFrame(out)
D14 = dist("subsample_n14", lambda u, m: OBS.loc[(u, "CDM"), m])
def verdict_text(b): return "difference can be stated" if b else "not distinguishable from the sample-size effect"
VR = []
for (u, m), d in D14.groupby(["unit", "metric"], sort=False):
    ok = bool(d["CDM_outside_2.5_97.5"].all())
    VR.append(dict(analysis="PRIMARY n = 14 (family = testable pairs)" if m != "solid_edges_family173" else "VARIANT n = 14 (family fixed at 173, P = 1 for untestable pairs)", unit=u, metric=m, CDM_observed=OBS.loc[(u, "CDM"), m],
                   **{f"{g}_p2.5_p97.5": f"{r['p2.5']:.4g} - {r['p97.5']:.4g}" for g, r in d.set_index("group").iterrows()}, **{f"{g}_prop_ge_CDM": r.prop_ge_CDM for g, r in d.set_index("group").iterrows()},
                   CDM_outside_all_three=ok, verdict=verdict_text(ok), LOO_runs_stated_of_14=""))
# sensitivity: each CDM leave-one-out value against the n = 13 sub-sampling intervals
LOO = R[R.design == "CDM_leave_one_out_n13_sensitivity"]; D13 = dist("subsample_n13_sensitivity", lambda u, m: float(LOO[LOO.unit == u][m].min()))   # reference = the LOWEST of the 14 leave-one-out values (least favourable)
for (u, m), d in D13.groupby(["unit", "metric"], sort=False):
    v = LOO[LOO.unit == u][m].values; st = [all((c < r["p2.5"]) or (c > r["p97.5"]) for _, r in d.iterrows()) for c in v]
    VR.append(dict(analysis="SENSITIVITY n = 13 (CDM leave-one-out vs n = 13 sub-samples)", unit=u, metric=m, CDM_observed=f"LOO range {v.min():.4g} - {v.max():.4g}",
                   **{f"{g}_p2.5_p97.5": f"{r['p2.5']:.4g} - {r['p97.5']:.4g}" for g, r in d.set_index("group").iterrows()}, **{f"{g}_prop_ge_CDM": r.prop_ge_CDM for g, r in d.set_index("group").iterrows()},
                   CDM_outside_all_three=bool(all(st)), verdict=verdict_text(all(st)) if len(set(st)) == 1 else f"mixed: stated in {sum(st)} of 14 leave-one-out runs", LOO_runs_stated_of_14=int(sum(st))))
VT = pd.DataFrame(VR); VT.loc[VT.metric.isin(MET[4:]), "verdict"] = "stored only (not one of the pre-fixed verdict metrics)"; DIST = pd.concat([D14, D13], ignore_index=True)
note("verdicts:\n" + VT[["analysis", "unit", "metric", "CDM_observed", "CDM_outside_all_three", "verdict", "LOO_runs_stated_of_14"]].to_string(index=False))
note("distribution (n = 14):\n" + D14.drop(columns=["design", "n_replicates", "n_subsample"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
note("distribution (n = 13 sensitivity; CDM_reference = lowest leave-one-out value):\n" + D13.drop(columns=["design", "n_replicates", "n_subsample"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
U = R[R.design.str.startswith("subsample")].groupby(["design", "unit", "group"], sort=False).agg(replicates=("rep", "size"), reps_with_untestable=("untestable_pairs", lambda x: int((x > 0).sum())),
      reps_Lcrispatus_constant=("untestable_Lcrispatus_pairs", lambda x: int((x > 0).sum())), reps_other_variable_constant=("untestable_other_pairs", lambda x: int((x > 0).sum())),
      mean_untestable_pairs=("untestable_pairs", "mean"), max_untestable_pairs=("untestable_pairs", "max"), min_testable=("testable_pairs", "min"),
      reps_solid_differs_family173=("solid_edges", lambda x: int((x != R.loc[x.index, "solid_edges_family173"]).sum()))).reset_index()
note("untestable pairs per sub-sample (a variable constant in the sub-sample):\n" + U.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
note("CDM leave-one-out (n = 13):\n" + LOO[["unit", "rep"] + MET].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ------------------------------------------------------------------ output tables (values table: KO-sum unit, n = 14 sub-samples, metrics 1-3, and the observed CDM values)
u = "KO_SUM"
F_SUM = f"{B3}/T_B3_summary_counts.csv"
VAL = pd.concat([D14[(D14.unit == u) & D14.metric.isin(MAIN)].assign(row="sub-sampling distribution (drawn)"),
                 pd.DataFrame([dict(row="CDM observed (drawn)", unit=u, metric=m, group="CDM", n_group=14, CDM_reference=OBS.loc[(u, "CDM"), m], observed_full_n=OBS.loc[(u, "CDM"), m]) for m in MAIN])], ignore_index=True)
for name, df in {f"{STEM}_values.csv": VAL, "Fig4F_subsample_distribution.csv": DIST, "Fig4F_subsample_replicates.csv.gz": R, "Fig4F_verdict.csv": VT, "Fig4F_untestable_pairs.csv": U}.items():
    df.to_csv(f"{OUT}/{name}", index=False, compression="infer")

# ------------------------------------------------------------------ checks (tables re-read from disk)
mism = 0; S = pd.read_csv(F_SUM); S = S[S.pair_type == "ALL"].set_index(["unit", "group"]); dd = pd.read_csv(f"{OUT}/Fig4F_subsample_distribution.csv"); rr = pd.read_csv(f"{OUT}/Fig4F_subsample_replicates.csv.gz"); vv = pd.read_csv(f"{OUT}/{STEM}_values.csv")
o = rr[rr.design == "observed_full_n"].set_index(["unit", "group"]); C = pd.read_csv(F_COR)
for (u_, g), r in o.iterrows():
    mism += int(r.solid_edges != S.loc[(u_, g), "both"]) + int(r["pairs_absrho_gt_0.6"] != S.loc[(u_, g), "absrho_gt_06"]) + int(r["pairs_P_lt_0.05"] != S.loc[(u_, g), "P_lt_05"]) + int(r.testable_pairs != 173)
    mism += int(abs(r.mean_absrho - C[(C.unit == u_) & (C.group == g)].rho.abs().mean()) > 1e-12)
for _, r in dd.iterrows():                                            # distribution table = recomputed from the replicate table
    x = rr[(rr.design == r.design) & (rr.unit == r.unit) & (rr.group == r.group)][r.metric].values
    mism += int(len(x) != NREP) + int(abs(np.percentile(x, 2.5) - r["p2.5"]) > 1e-12) + int(abs(np.percentile(x, 97.5) - r["p97.5"]) > 1e-12) + int(abs(x.mean() - r["mean"]) > 1e-12)
m_ = vv[vv.row.str.startswith("sub")].merge(dd, on=["design", "unit", "metric", "group"], suffixes=("", "_d")); mism += int(len(m_) != 9) + int((m_["p2.5"] != m_["p2.5_d"]).sum() + (m_["p97.5"] != m_["p97.5_d"]).sum() + (m_.p50 != m_.p50_d).sum())
assert mism == 0, mism
print("CHECK observed full-n metrics vs T_B3_summary_counts.csv (solid = both, |rho| > 0.6, P < 0.05; 8 group x unit) and mean |rho| vs T_B3_correlations.csv; "
      f"distribution table vs replicate table ({len(dd)} rows); values csv vs distribution table: mismatches {mism}")
