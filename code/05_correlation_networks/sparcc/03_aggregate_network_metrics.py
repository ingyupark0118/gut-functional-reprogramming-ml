#!/usr/bin/env python
"""P9_SparCC aggregation (design: ../DESIGN_prefixed.md sections 5 and 7). No SparCC is run here.
filtered (MAIN)          : filtered_run/net/W/*.json
unfiltered (SUPPLEMENT)  : stored C1 outputs - T_C1_subsample_all.csv / T_C1_main.csv (nodes, edges, modularity, cross-phylum) and npz cor_upper (mean |r|, pairs |r| > 0.6)
unfiltered_posthoc_removed (REFERENCE column only): the earlier x5-type sensitivity = rare species removed from the STORED unfiltered matrices (Louvain best of seeds 0-9)
Old outputs are moved to ../_discard/<stamp>/, never overwritten. Ends with assertions (C1 report numbers reproduced; tables re-read from disk)."""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
import sys, glob, json, time, shutil, hashlib
import numpy as np, pandas as pd, networkx as nx
REV = "."; C1 = f"{REV}/work_a/C1_sparcc"; P9 = f"{REV}/sparcc_runs"
GROUPS = ["NC", "DM", "Cirrhosis", "CDM"]; SUBG = GROUPS[:3]; NG = {"NC": 59, "DM": 46, "Cirrhosis": 30, "CDM": 14}
PCT = [0, 2.5, 25, 50, 75, 97.5, 100]
METRICS = ["modularity", "cross_phylum_frac", "edges", "mean_abs_r_all", "nodes", "n_species", "mean_abs_r_edges", "pairs_abs_r_gt_0.6"]
PANEL = METRICS[:4]                                   # pre-fixed primary (panel) metrics
STAMP = time.strftime("%Y%m%d_%H%M%S"); LOG = []
def log(s): LOG.append(str(s)); print(s, flush=True)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def keep_old(p):
    if os.path.exists(p):
        d = f"{P9}/_discard/{STAMP}"; os.makedirs(d, exist_ok=True); shutil.move(p, f"{d}/{os.path.basename(p)}")
    return p
def louvain_best(i, j):
    g = nx.Graph(); g.add_edges_from(zip(i.tolist(), j.tolist()))
    return g.number_of_nodes(), max(nx.community.modularity(g, nx.community.louvain_communities(g, seed=s)) for s in range(10))

# ---------------------------------------------------------------- filtered (main)
F = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(f"{P9}/filtered_run/net/W/*.json"))])
assert len(F) == 304 and F.id.is_unique, len(F)
log(f"filtered networks: {len(F)}; status {F.status.value_counts().to_dict()}")
F = F.rename(columns={"pairs_abs_r_gt_06": "pairs_abs_r_gt_0.6", "n_species_retained": "n_species"})
F["analysis"] = "filtered"
# ---------------------------------------------------------------- unfiltered (stored C1 outputs) + post hoc removal reference
cnt = pd.read_csv(f"{C1}/inputs/species_s_counts_raw.csv", index_col=0); info = pd.read_csv(f"{C1}/inputs/taxon_info.csv").set_index("taxon")
S = pd.read_csv(f"{C1}/T_C1_subsample_all.csv"); M = pd.read_csv(f"{C1}/T_C1_main.csv")
T = pd.concat([M[M.taxon_set == "all1338"], S[S.taxon_set == "all1338"]], ignore_index=True); assert len(T) == 304
I, J = np.triu_indices(1338, 1); rowsU, rowsP = [], []; nb = 0
dr = pd.read_csv(f"{C1}/inputs/subsample_draws.csv")
for _, t in T.iterrows():
    z = np.load(f"{C1}/net/all1338/{t.id}.npz", allow_pickle=True); taxa = z["taxa"]; ids = list(z["samples"]); assert list(taxa) == list(cnt.columns)
    if t.draw >= 0: assert ids == list(dr[(dr.group == t.group) & (dr.draw == t.draw)].iloc[0, 2:])       # same participants as the filtered replicate
    a = np.abs(z["cor_upper"].astype(np.float64)); p = z["nsig_upper"] / 100.0; prev = (cnt.loc[ids].values > 0).sum(0)
    e = (a > 0.5) & (a < 0.9) & (p < 0.05); pres = (prev[I] > 0) & (prev[J] > 0); ph = info.loc[taxa, "phylum_metaphlan"].values
    nb += int(e.sum() != t.edges)
    rowsU.append(dict(analysis="unfiltered", id=t.id, group=t.group, draw=int(t.draw), n_samples=int(t.n_samples), status="ok", n_species=int((prev > 0).sum()), nodes=int(t.nodes), edges=int(t.edges),
                      modularity=t.modularity, cross_phylum_frac=t.cross_phylum_frac, mean_abs_r_all=float(a[pres].mean()), mean_abs_r_all1338=float(a.mean()),
                      mean_abs_r_edges=float(a[e].mean()), **{"pairs_abs_r_gt_0.6": int((a[pres] > 0.6).sum())}, pairs_abs_r_gt_06_all1338=int((a > 0.6).sum()), edges_recounted_from_npz=int(e.sum())))
    assert int((prev > 0).sum()) == int(t.n_taxa_present)
    k3 = (prev[I] >= 3) & (prev[J] >= 3); e3 = e & k3; n3, m3 = louvain_best(I[e3], J[e3]) if e3.any() else (0, np.nan)
    rowsP.append(dict(analysis="unfiltered_posthoc_removed", id=t.id, group=t.group, draw=int(t.draw), n_samples=int(t.n_samples), status="ok", n_species=int((prev >= 3).sum()), nodes=n3, edges=int(e3.sum()),
                      modularity=m3, cross_phylum_frac=float((ph[I[e3]] != ph[J[e3]]).mean()) if e3.any() else np.nan, mean_abs_r_all=float(a[k3].mean()),
                      mean_abs_r_edges=float(a[e3].mean()) if e3.any() else np.nan, **{"pairs_abs_r_gt_0.6": int((a[k3] > 0.6).sum())}))
log(f"unfiltered: edges recounted from the float32 npz differ from the stored table in {nb} of 304 networks (float32 rounding at the 0.5 / 0.9 borders); table values are used")
U = pd.DataFrame(rowsU); PH = pd.DataFrame(rowsP)
# the filtered replicate must be the same participants and its species count must equal 'present in >= 3' recomputed here
chk = F.merge(PH[["group", "draw", "n_species"]], on=["group", "draw"], suffixes=("", "_recount")); assert len(chk) == 304 and (chk.n_species == chk.n_species_recount).all()
cols = ["analysis", "id", "group", "draw", "n_samples", "status"] + METRICS
R = pd.concat([F[cols + ["edges_pos", "edges_neg", "n_components", "clr_fallback_main", "sec_total"]], U[cols + ["mean_abs_r_all1338", "pairs_abs_r_gt_06_all1338", "edges_recounted_from_npz"]], PH[cols]], ignore_index=True)
R["group"] = pd.Categorical(R.group, GROUPS); R = R.sort_values(["analysis", "group", "draw"]).reset_index(drop=True); R["group"] = R.group.astype(str)

# ---------------------------------------------------------------- distribution tables, verdicts
def midrank(x, v): return float(100 * (np.mean(x < v) + 0.5 * np.mean(x == v)))
def dist(an):
    d = R[R.analysis == an]; out = []
    for m in METRICS:
        cdm = float(d[(d.group == "CDM")][m].iloc[0])
        for g in SUBG:
            xa = d[(d.group == g) & (d.draw >= 0)][m].astype(float).values; assert len(xa) == 100
            x = xa[~np.isnan(xa)]; own = float(d[(d.group == g) & (d.draw < 0)][m].iloc[0]); p = np.percentile(x, PCT)
            pos = "above" if cdm > p[5] else "below" if cdm < p[1] else "inside"
            out.append(dict(analysis=an, metric=m, role="primary (panel)" if m in PANEL else "auxiliary", group=g, n_group=NG[g], n_subsample=14, n_replicates=len(x), n_undefined=int(len(xa) - len(x)),
                            **{f"p{q:g}": v for q, v in zip(PCT, p)}, mean=float(x.mean()), n_ge_CDM=int((x >= cdm).sum()), prop_ge_CDM=float((x >= cdm).mean()),
                            n_le_CDM=int((x <= cdm).sum()), prop_le_CDM=float((x <= cdm).mean()), CDM_percentile_midrank=midrank(x, cdm), observed_full_n=own,
                            own_full_percentile_midrank=midrank(x, own), CDM_observed=cdm, **{"CDM_outside_2.5_97.5": pos != "inside"}, CDM_position=pos))
    return pd.DataFrame(out)
D = {an: dist(an) for an in ["filtered", "unfiltered", "unfiltered_posthoc_removed"]}
def verdicts(an):
    out = []
    for m, d in D[an].groupby("metric", sort=False):
        d = d.set_index("group"); ok = bool(d["CDM_outside_2.5_97.5"].all())
        out.append(dict(analysis=an, metric=m, role=d.role.iloc[0], CDM_observed=d.CDM_observed.iloc[0],
                        **{f"{g}_p2.5_p97.5": f"{d.loc[g, 'p2.5']:.4g} - {d.loc[g, 'p97.5']:.4g}" for g in SUBG}, **{f"{g}_min_max": f"{d.loc[g, 'p0']:.4g} - {d.loc[g, 'p100']:.4g}" for g in SUBG},
                        **{f"{g}_CDM_percentile": d.loc[g, "CDM_percentile_midrank"] for g in SUBG}, **{f"{g}_n_ge_CDM_of_100": int(d.loc[g, "n_ge_CDM"]) for g in SUBG},
                        **{f"{g}_CDM_position": d.loc[g, "CDM_position"] for g in SUBG}, CDM_outside_all_three=ok,
                        verdict="difference can be stated" if ok else "not distinguishable from the sample-size effect"))
    return pd.DataFrame(out)
V = pd.concat([verdicts(an) for an in D], ignore_index=True)
# paired filtered versus unfiltered (same participants in each replicate)
w = R[R.draw >= 0].pivot(index=["group", "draw"], columns="analysis", values=METRICS); pr = []
for m in METRICS:
    for g in SUBG:
        a_, b_ = w.loc[g][(m, "unfiltered")].astype(float), w.loc[g][(m, "filtered")].astype(float); df = b_ - a_
        pr.append(dict(metric=m, group=g, n_pairs=int((~(a_.isna() | b_.isna())).sum()), unfiltered_median=a_.median(), filtered_median=b_.median(), median_difference_filtered_minus_unfiltered=df.median(),
                       difference_p2_5=np.nanpercentile(df, 2.5), difference_p97_5=np.nanpercentile(df, 97.5), n_filtered_lower=int((df < 0).sum()), n_filtered_higher=int((df > 0).sum()),
                       spearman_rho_between_runs=a_.corr(b_, method="spearman"),
                       CDM_unfiltered=float(R[(R.analysis == "unfiltered") & (R.group == "CDM")][m].iloc[0]), CDM_filtered=float(R[(R.analysis == "filtered") & (R.group == "CDM")][m].iloc[0]),
                       CDM_percentile_unfiltered=float(D["unfiltered"].set_index(["metric", "group"]).loc[(m, g), "CDM_percentile_midrank"]),
                       CDM_percentile_filtered=float(D["filtered"].set_index(["metric", "group"]).loc[(m, g), "CDM_percentile_midrank"]),
                       CDM_percentile_posthoc_removed=float(D["unfiltered_posthoc_removed"].set_index(["metric", "group"]).loc[(m, g), "CDM_percentile_midrank"])))
PAIR = pd.DataFrame(pr)
# retention
ret = []
for g in GROUPS:
    f = F[F.group == g]; fu = f[f.draw < 0].iloc[0]; u = U[(U.group == g) & (U.draw < 0)].iloc[0]; r = dict(group=g, n_group=NG[g], species_present_full=int(u.n_species), species_retained_full=int(fu.n_species), nodes_full=int(fu.nodes), edges_full=int(fu.edges))
    if g != "CDM":
        s = f[f.draw >= 0]; us = U[(U.group == g) & (U.draw >= 0)]
        for nm, x in (("species_present_sub", us.n_species), ("species_retained_sub", s.n_species), ("nodes_sub", s.nodes), ("edges_sub", s.edges), ("edges_neg_frac_sub", s.edges_neg / s.edges)):
            r.update({f"{nm}_median": float(np.median(x)), f"{nm}_min": float(np.min(x)), f"{nm}_max": float(np.max(x))})
        r.update(sub_reps_edges_lt_10=int((s.edges < 10).sum()), sub_reps_modularity_undefined=int(s.modularity.isna().sum()), sub_reps_not_run=int((s.status != "ok").sum()))
    ret.append(r)
RET = pd.DataFrame(ret)

out = {f"{P9}/aggregate/sparcc_subsample_replicates.csv": R, f"{P9}/tables/T_sparcc_subsample_distribution_filtered.csv": D["filtered"], f"{P9}/tables/T_sparcc_subsample_distribution_unfiltered.csv": D["unfiltered"],
       f"{P9}/tables/T_sparcc_subsample_distribution_unfiltered_posthoc_removed.csv": D["unfiltered_posthoc_removed"], f"{P9}/tables/T_sparcc_verdicts.csv": V,
       f"{P9}/tables/T_sparcc_paired_filtered_vs_unfiltered.csv": PAIR, f"{P9}/tables/T_sparcc_filter_retention.csv": RET}
for p, df in out.items(): df.to_csv(keep_old(p), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
log("observed (full group):\n" + R[R.draw < 0][["analysis", "group", "n_samples"] + METRICS].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
for an in D: log(f"distribution [{an}]:\n" + D[an].drop(columns=["analysis", "role", "n_group", "n_subsample", "prop_ge_CDM", "prop_le_CDM"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
log("verdicts:\n" + V[["analysis", "metric", "role", "CDM_observed"] + [f"{g}_CDM_percentile" for g in SUBG] + [f"{g}_CDM_position" for g in SUBG] + ["verdict"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
log("paired:\n" + PAIR.to_string(index=False, float_format=lambda v: f"{v:.4g}")); log("retention:\n" + RET.T.to_string(float_format=lambda v: f"{v:.4g}"))

# ---------------------------------------------------------------- assertions
du = pd.read_csv(f"{P9}/tables/T_sparcc_subsample_distribution_unfiltered.csv").set_index(["metric", "group"])
assert [round(du.loc[("modularity", g), "CDM_percentile_midrank"]) for g in SUBG] == [29, 56, 24]                 # C1 report 4.2 / crosscheck
assert [round(du.loc[("cross_phylum_frac", g), "CDM_percentile_midrank"]) for g in SUBG] == [100, 100, 74]
assert [int(R[(R.analysis == "unfiltered") & (R.group == g) & (R.draw < 0)].edges.iloc[0]) for g in GROUPS] == [2665, 5861, 3995, 6336]     # counted once (the manuscript's were doubled)
assert abs(du.loc[("modularity", "NC"), "CDM_observed"] - 0.599) < 5e-4
cs = pd.read_csv(f"{C1}/T_C1_subsample_summary.csv"); cs = cs[cs.taxon_set == "all1338"].set_index(["property", "group"]); n = 0
for m in ["modularity", "cross_phylum_frac", "edges", "nodes"]:
    for g in SUBG:
        for a_, b_ in (("p2.5", "q2_5"), ("p97.5", "q97_5"), ("p50", "median"), ("p0", "min"), ("p100", "max"), ("mean", "mean"), ("CDM_percentile_midrank", "CDM_percentile")):
            assert abs(du.loc[(m, g), a_] - cs.loc[(m, g), b_]) < 1e-9, (m, g, a_); n += 1
x5 = pd.read_csv(f"{REV}/work_a/_crosscheck_C1/T_x5_prev3_sensitivity.csv"); x5 = x5[x5.taxon_set == "all1338"].set_index("id"); ph = PH.set_index("id")
assert (ph.loc[x5.index, "edges"] == x5.prev3_edges).all() and (ph.loc[x5.index, "nodes"] == x5.prev3_nodes).all() and np.allclose(ph.loc[x5.index, "cross_phylum_frac"], x5.prev3_cross, equal_nan=True)
dp = D["unfiltered_posthoc_removed"].set_index(["metric", "group"])
log(f"CHECK unfiltered aggregation reproduces C1: modularity percentiles 29/56/24, cross-phylum 100/100/74, full-group edges 2665/5861/3995/6336 (once), {n} cells of T_C1_subsample_summary.csv within 1e-9: OK")
log(f"CHECK post hoc removal reproduces crosscheck x5 (edges, nodes, cross-phylum of 304 networks): OK; cross-phylum CDM percentile NC/DM/Cirr = "
    f"{[round(dp.loc[('cross_phylum_frac', g), 'CDM_percentile_midrank']) for g in SUBG]} (x5 report: 100/98/3); modularity (Louvain 0-9 here, 0-2 in x5) = {[round(dp.loc[('modularity', g), 'CDM_percentile_midrank']) for g in SUBG]} (x5: 67/74/30)")
rr = pd.read_csv(f"{P9}/aggregate/sparcc_subsample_replicates.csv"); mism = 0
for an in ["filtered", "unfiltered"]:
    dd = pd.read_csv(f"{P9}/tables/T_sparcc_subsample_distribution_{an}.csv")
    for _, r in dd.iterrows():
        x = rr[(rr.analysis == an) & (rr.group == r.group) & (rr.draw >= 0)][r.metric].dropna().values
        tol = 1e-12 * max(1.0, abs(r["p97.5"]))          # relative: edge counts are ~1e4, where the float spacing itself is ~2e-12
        mism += int(len(x) != r.n_replicates) + int(abs(np.percentile(x, 2.5) - r["p2.5"]) > tol) + int(abs(np.percentile(x, 97.5) - r["p97.5"]) > tol) + int(abs(x.mean() - r["mean"]) > 1e3 * tol)
assert mism == 0, mism
log(f"CHECK distribution tables re-read from disk vs replicate table: mismatches {mism}")
log("outputs (sha256):\n" + "\n".join(f"  {sha(p)}  {p}" for p in out))
open(keep_old(f"{P9}/logs/LOG_aggregate.txt"), "w").write("\n".join(LOG) + "\n")
