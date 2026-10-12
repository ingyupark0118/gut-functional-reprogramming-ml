#!/usr/bin/env python
"""A1-v2 aggregation. Produces the section 7 deliverables.

Two estimators are kept apart, because v1 mixed them (point estimate from a pooled
out-of-fold AUC, interval from a different statistic):

  A  fold AUC -> mean within seed -> mean across seeds.
     Spread reported as SD across seeds and SD across folds.
     This matches how the published nested result was aggregated.
  B  average each sample's predicted probability across seeds, then compute ONE AUC
     on those averaged probabilities. Interval from a sample-level bootstrap.

Neither is a full generalisation interval for the whole procedure; the log says so.

Usage: python a1v2_analyze.py [--comps g1]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

A1 = "./nested_cv"
TAG = os.environ.get("A1_TAG", "A1v2")
RES = f"{A1}/" + os.environ.get("A1_RES", "results_v2")
LBL = {"g1": "NC vs DM", "g2": "Cirrhosis vs CDM",
       "g3": "NC vs Cirrhosis", "g4": "DM vs CDM"}
GRP = {"g1": ("NC", "DM"), "g2": ("Cirrhosis", "CDM"),
       "g3": ("NC", "Cirrhosis"), "g4": ("DM", "CDM")}
INPUTS = ["species", "ko", "combined"]

ap = argparse.ArgumentParser()
ap.add_argument("--comps", default="g1,g2,g3,g4")
COMPS = ap.parse_args().comps.split(",")

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


cells = {}
for f in glob.glob(f"{RES}/*.json"):
    d = json.load(open(f))
    if d["comp"] in COMPS:
        cells[(d["input"], d["comp"], d["seed"])] = d["folds"]
P(f"loaded cells: {len(cells)}   comparisons: {sorted(set(k[1] for k in cells))}")

lab = pd.read_csv(f"{A1}/labels.csv", index_col=0)["Group"]
perf, stab, bstat, hps, gn = [], [], [], [], []

for comp in COMPS:
    a, b = GRP[comp]
    y_all = (lab[lab.isin([a, b])] == b).astype(int).values
    gn.append(dict(comparison=LBL[comp], g=comp, n=len(y_all), control=a, case=b,
                   n_control=int((y_all == 0).sum()), n_case=int(y_all.sum()),
                   mean_case_per_test_fold=round(y_all.sum() / 5, 2)))
    for inp in INPUTS:
        seeds = sorted(s for (i, c, s) in cells if i == inp and c == comp)
        if not seeds:
            continue
        for model in ("xgb", "lr"):
            seed_means, fold_aucs, nsel, ndrop, prob = [], [], [], 0, {}
            for s in seeds:
                fl = cells[(inp, comp, s)]
                ok = [x for x in fl if x["status"] == "ok"]
                ndrop += len(fl) - len(ok)
                aucs = [x[model]["auc"] for x in ok]
                if aucs:
                    seed_means.append(np.mean(aucs)); fold_aucs += aucs
                nsel += [x["n_selected"] for x in ok]
                for x in ok:
                    for idx, pv in zip(x["test_idx"], x[f"p_{model}"]):
                        prob.setdefault(idx, []).append(pv)
            if not seed_means:
                continue
            idx = sorted(prob)
            pb = np.array([np.mean(prob[i]) for i in idx]); yb = y_all[idx]
            lo, hi = boot_ci(yb, pb)
            row = dict(comparison=LBL[comp], g=comp, input=inp, model=model,
                       n_seeds=len(seeds), n_folds_used=len(fold_aucs),
                       n_folds_dropped=ndrop,
                       estA_auc=float(np.mean(seed_means)),
                       estA_sd_across_seeds=float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else np.nan,
                       estA_sd_across_folds=float(np.std(fold_aucs, ddof=1)),
                       estA_fold_min=float(np.min(fold_aucs)),
                       estA_fold_max=float(np.max(fold_aucs)),
                       estB_auc=float(roc_auc_score(yb, pb)),
                       estB_ci_lo=lo, estB_ci_hi=hi,
                       n_selected_median=float(np.median(nsel)) if nsel else np.nan)
            for m in ("acc", "f1", "sens", "spec"):
                vals = [x[model][m] for s in seeds
                        for x in cells[(inp, comp, s)] if x["status"] == "ok"]
                row[f"estA_{m}"] = float(np.nanmean(vals)) if vals else np.nan
            perf.append(row)

        freq, nok = {}, 0
        for s in seeds:
            for x in cells[(inp, comp, s)]:
                bstat.append(dict(comparison=LBL[comp], g=comp, input=inp, seed=s,
                                  fold=x["fold"], status=x["status"],
                                  n_confirmed=x["n_confirmed"], n_tentative=x["n_tentative"],
                                  n_selected=x["n_selected"],
                                  inner_zero_folds=sum(1 for q in x.get("inner", [])
                                                       if q["n_selected"] == 0),
                                  seconds=x["seconds"]))
                if x["status"] == "ok":
                    nok += 1
                    for ft in x["confirmed_features"]:
                        freq[ft] = freq.get(ft, 0) + 1
                    hps.append(dict(comparison=LBL[comp], g=comp, input=inp, seed=s,
                                    fold=x["fold"],
                                    **{f"xgb_{k}": v for k, v in x["best_params_xgb"].items()},
                                    lr_C=x["best_C_lr"]))
        for ft, n in sorted(freq.items(), key=lambda t: -t[1]):
            stab.append(dict(comparison=LBL[comp], g=comp, input=inp, feature=ft,
                             n_folds_confirmed=n, n_folds_ok=nok,
                             pct_confirmed=round(100 * n / nok, 1) if nok else np.nan))

pf = pd.DataFrame(perf)
pf.to_csv(f"{A1}/T_{TAG}_performance.csv", index=False)
pd.DataFrame(stab).to_csv(f"{A1}/T_{TAG}_feature_stability.csv", index=False)
bs = pd.DataFrame(bstat); bs.to_csv(f"{A1}/T_{TAG}_boruta_status.csv", index=False)
pd.DataFrame(hps).to_csv(f"{A1}/T_{TAG}_hyperparams.csv", index=False)
pd.DataFrame(gn).to_csv(f"{A1}/T_{TAG}_group_n.csv", index=False)

v1 = pd.read_csv(f"{A1}/T_A1_summary.csv")
rows = []
for _, r in pf[pf.model == "xgb"].iterrows():
    m = v1[(v1.input == r.input) & (v1.comp == r.g)]
    v1a = float(m.auc_mean.iloc[0]) if len(m) else np.nan
    rows.append(dict(comparison=r.comparison, input=r.input, v1_pooled_oof_auc=v1a,
                     v1_n_sel_mean=float(m.n_sel_mean.iloc[0]) if len(m) else np.nan,
                     v2_estA_auc=r.estA_auc, v2_estB_auc=r.estB_auc,
                     v2_n_sel_median=r.n_selected_median,
                     delta_estA_minus_v1=r.estA_auc - v1a))
pd.DataFrame(rows).to_csv(f"{A1}/T_{TAG}_vs_v1.csv", index=False)

P("\n" + "=" * 104)
P("[performance]  A = fold AUC -> mean within seed -> mean across seeds   |   B = one AUC on the seed-averaged probabilities")
P("=" * 104)
for comp in COMPS:
    sub = pf[pf.g == comp]
    if sub.empty:
        continue
    g = [x for x in gn if x["g"] == comp][0]
    P(f"\n{LBL[comp]}   n={g['n']} ({g['control']} {g['n_control']} / {g['case']} {g['n_case']})")
    P(f"  {'input':10s} {'model':5s} {'estA':>7s} {'sd_seed':>8s} {'sd_fold':>8s} "
      f"{'fold range':>15s} {'estB':>7s} {'95% CI':>17s} {'n_sel':>6s} {'drop':>5s}")
    for _, r in sub.sort_values(["input", "model"]).iterrows():
        P(f"  {r.input:10s} {r.model:5s} {r.estA_auc:7.4f} {r.estA_sd_across_seeds:8.4f} "
          f"{r.estA_sd_across_folds:8.4f} "
          f"{r.estA_fold_min:.3f}-{r.estA_fold_max:.3f}".rjust(15)
          + f" {r.estB_auc:7.4f} "
          + f"[{r.estB_ci_lo:.3f},{r.estB_ci_hi:.3f}]".rjust(17)
          + f" {r.n_selected_median:6.1f} {int(r.n_folds_dropped):5d}")
    for model in ("xgb", "lr"):
        s = sub[sub.model == model].set_index("input").estA_auc
        if len(s) == 3:
            P(f"  combined > single ({model}): "
              f"{'yes' if s['combined'] > max(s['species'], s['ko']) else 'no'}"
              f"   combined={s['combined']:.4f} species={s['species']:.4f} ko={s['ko']:.4f}")

P("\n" + "=" * 104)
P("[Boruta]")
P("=" * 104)
for comp in COMPS:
    for inp in INPUTS:
        d = bs[(bs.g == comp) & (bs.input == inp)]
        if d.empty:
            continue
        ok = d[d.status == "ok"]
        P(f"  {LBL[comp]:18s} {inp:10s} folds={len(d):3d} ok={len(ok):3d} "
          f"dropped={len(d)-len(ok):2d} | confirmed med={ok.n_confirmed.median():5.1f} "
          f"(min {int(ok.n_confirmed.min())}, max {int(ok.n_confirmed.max())}) "
          f"tentative med={ok.n_tentative.median():4.1f} | "
          f"inner folds with 0 selected={int(d.inner_zero_folds.sum())}/{len(d)*3}")

P("\n" + "=" * 104)
P("[vs v1]  v1 = pooled OOF AUC (v1 did not store fold-level AUCs, so it cannot be re-aggregated as A)")
P("=" * 104)
for r in rows:
    P(f"  {r['comparison']:18s} {r['input']:10s} v1={r['v1_pooled_oof_auc']:.4f} -> "
      f"v2_A={r['v2_estA_auc']:.4f} ({r['delta_estA_minus_v1']:+.4f})  "
      f"v2_B={r['v2_estB_auc']:.4f} | n_sel {r['v1_n_sel_mean']:7.1f} -> {r['v2_n_sel_median']:.1f}")

P("\nNote: neither estimator gives a final CI that includes the generalisation uncertainty of the whole training procedure.")
P("      The bootstrap of estimator B reflects sample-level variation only.")

open(f"{A1}/LOG_{TAG}_analyze.txt", "w").write("\n".join(log) + "\n")
print(f"\n[written] T_{TAG}_*.csv")
