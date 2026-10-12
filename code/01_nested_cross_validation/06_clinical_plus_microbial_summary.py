#!/usr/bin/env python
"""Aggregation for the triage nested CV. Adapted from project/nested_cv/a1v2_analyze.py (Fig 2):
estimator A / estimator B and boot_ci() are kept; added: stratified bootstrap, paired dAUROC
(bootstrap + DeLong), AUPRC, Brier, in-fold Youden counts, reclassification, feature selection,
leak diagnostic. Everything follows DESIGN_prefixed.md.
Usage: python triage_analyze.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import beta, norm
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triage_nested_cv as T

OUT, TAB, RES = T.OUT, f"{T.OUT}/tables", f"{T.OUT}/results"
SEEDS, NB = T.SEEDS, 2000
LEARN = {"xgb": "XGBoost", "lr": "L2-LR"}
NAME = {"clin": "Clinical", "micro": "Microbiome (in-fold Boruta)", "comb": "Combined (clinical + in-fold Boruta)",
        "micro11": "Microbiome-11 (pre-selected)", "comb11": "Combined-11 (clinical + pre-selected 11)"}
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
def md(df, floatfmt="%.3f"):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt % v)
    P("| " + " | ".join(map(str, d.columns)) + " |"); P("|" + "---|" * len(d.columns))
    for r in d.itertuples(index=False):
        P("| " + " | ".join(map(str, r)) + " |")


def boot_ci(y, p, n=2000, seed=0):               # verbatim Fig 2 (unstratified)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def strat_idx(y, n=NB, seed=0):
    """Participant resamples stratified by outcome; ONE index set reused for every model (paired)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]


def pct(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def delong(y, p1, p2):
    """Paired DeLong for AUC(p2) - AUC(p1): delta, se, lo, hi, two-sided P."""
    V10, V01, auc = [], [], []
    for p in (p1, p2):
        a, b = p[y == 1], p[y == 0]
        psi = (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])
        V10.append(psi.mean(1)); V01.append(psi.mean(0)); auc.append(psi.mean())
        assert abs(auc[-1] - roc_auc_score(y, p)) < 1e-12
    S = np.cov(np.vstack(V10)) / (y == 1).sum() + np.cov(np.vstack(V01)) / (y == 0).sum()
    d = auc[1] - auc[0]; se = float(np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]))
    return d, se, d - 1.96 * se, d + 1.96 * se, float(2 * norm.sf(abs(d / se))) if se > 0 else np.nan


def cp(k, n):
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1)), (1.0 if k == n else beta.ppf(.975, k + 1, n - k))


# ---------------------------------------------------------------- load
T._init()
cells = {}
for f in glob.glob(f"{RES}/folds/*.json"):
    d = json.load(open(f)); cells.setdefault(d["variant"], {})[(d["seed"], d["fold"])] = d
P("variants loaded:", {v: len(c) for v, c in sorted(cells.items())})
GROUP = T._G["lab"]


def collect(variant, model, learner):
    """prob[n, seed], above[n, seed] (in-fold Youden), per-fold rows."""
    D = T.get_data(variant); y = D["y"]; c = cells[variant]
    seeds = sorted({s for s, _ in c})
    prob = np.full((len(y), len(seeds)), np.nan); above = np.zeros_like(prob, bool); rows = []
    for (s, f), d in sorted(c.items()):
        m = d["models"].get(model)
        if m is None:
            return None
        te = np.array(d["test_idx"]); yte = y[te]; j = seeds.index(s)
        ok = m["status"] == "ok"
        p = np.array(m[f"p_{learner}"]) if ok else np.full(len(te), d["train_prevalence"])  # intercept-only
        thr = m.get(f"thr_{learner}") if ok else None
        ab = p >= thr if thr is not None else np.zeros(len(te), bool)
        prob[te, j] = p; above[te, j] = ab
        rows.append(dict(variant=variant, model=model, learner=learner, seed=s, fold=f, status=m["status"],
                         n_test=len(te), n_pos_test=int(yte.sum()),
                         fold_auc=roc_auc_score(yte, p) if ok else np.nan, threshold_inner_youden=thr,
                         TP=int((ab & (yte == 1)).sum()), FN=int((~ab & (yte == 1)).sum()),
                         FP=int((ab & (yte == 0)).sum()), TN=int((~ab & (yte == 0)).sum())))
    return dict(y=y, ids=np.array(D["ids"]), prob=prob, above=above, folds=pd.DataFrame(rows), seeds=seeds,
                complete=len(c) == len(seeds) * T.OUTER)


perf, delta, thr_rows, sens_rows, oof = [], [], [], [], []
store = {}
MAINV = [v for v in ("main",) if v in cells]
for variant in MAINV + sorted(v for v in cells if v.startswith("nullfull")):
    y = T.get_data(variant)["y"]; BI = strat_idx(y)
    for model in NAME:
        for learner in LEARN:
            c = collect(variant, model, learner)
            if c is None:
                continue
            store[(variant, model, learner)] = c
            fd = c["folds"]; okf = fd[fd.status == "ok"]
            sm = okf.groupby("seed").fold_auc.mean()
            pb = np.nanmean(c["prob"], 1)
            ba = [roc_auc_score(y[i], pb[i]) for i in BI]; bp = [average_precision_score(y[i], pb[i]) for i in BI]
            f2 = boot_ci(y, pb)
            perf.append(dict(variant=variant, model=model, model_label=NAME[model], learner=LEARN[learner],
                             n=len(y), n_pos=int(y.sum()), n_seeds=len(c["seeds"]), n_folds_used=len(okf),
                             n_folds_dropped=len(fd) - len(okf),
                             estA_auc=sm.mean(), estA_sd_across_seeds=sm.std(ddof=1) if len(sm) > 1 else np.nan,
                             estA_sd_across_folds=okf.fold_auc.std(ddof=1), estA_fold_min=okf.fold_auc.min(),
                             estA_fold_max=okf.fold_auc.max(),
                             estA_auc_dropped_as_half=fd.assign(a=fd.fold_auc.fillna(.5)).groupby("seed").a.mean().mean(),
                             estB_auc=roc_auc_score(y, pb), estB_ci_lo=pct(ba)[0], estB_ci_hi=pct(ba)[1],
                             estB_ci_lo_fig2_unstratified=f2[0], estB_ci_hi_fig2_unstratified=f2[1],
                             auprc=average_precision_score(y, pb), auprc_ci_lo=pct(bp)[0], auprc_ci_hi=pct(bp)[1],
                             auprc_baseline=y.mean(), brier=float(np.mean((pb - y) ** 2))))
            # thresholds / counts
            if variant in MAINV:
                thr_rows.append(fd)
                maj = c["above"].sum(1) >= 3
                at30 = pb >= 0.30
                for rule, cl in (("in-fold Youden, majority of 5 seeds", maj), ("fixed 0.30 on seed-mean probability (NOT pre-specified)", at30)):
                    tp, fn = int((cl & (y == 1)).sum()), int((~cl & (y == 1)).sum())
                    fp, tn = int((cl & (y == 0)).sum()), int((~cl & (y == 0)).sum())
                    sl, sh = cp(tp, tp + fn); pl, ph = cp(tn, tn + fp)
                    ps = [(int((c["above"][:, j] & (y == 1)).sum()), int((~c["above"][:, j] & (y == 0)).sum()))
                          for j in range(len(c["seeds"]))]
                    sens_rows.append(dict(variant=variant, model=model, learner=LEARN[learner], rule=rule,
                                          TP=tp, FN=fn, FP=fp, TN=tn, sens=tp / (tp + fn), sens_lo=sl, sens_hi=sh,
                                          spec=tn / (tn + fp), spec_lo=pl, spec_hi=ph,
                                          per_seed_TP="/".join(str(a) for a, _ in ps) if "Youden" in rule else "",
                                          per_seed_TN="/".join(str(b) for _, b in ps) if "Youden" in rule else "",
                                          pooled_TP_of=f"{sum(a for a, _ in ps)}/{int(y.sum())*len(ps)}" if "Youden" in rule else "",
                                          pooled_TN_of=f"{sum(b for _, b in ps)}/{int((y==0).sum())*len(ps)}" if "Youden" in rule else "",
                                          threshold_median=fd.threshold_inner_youden.median() if "Youden" in rule else 0.30,
                                          threshold_min=fd.threshold_inner_youden.min() if "Youden" in rule else 0.30,
                                          threshold_max=fd.threshold_inner_youden.max() if "Youden" in rule else 0.30))
                for j, s in enumerate(c["seeds"]):
                    oof.append(pd.DataFrame(dict(variant=variant, SampleID=c["ids"], y=y, seed=s, model=model,
                                                 learner=LEARN[learner], prob=c["prob"][:, j],
                                                 above_infold_youden=c["above"][:, j])))
    # paired differences
    for space, (a, mi, co) in {"primary (in-fold selection)": ("clin", "micro", "comb"),
                               "comparator (11 pre-selected)": ("clin", "micro11", "comb11")}.items():
        for learner in LEARN:
            for lab, hi in (("Combined - Clinical", co), ("Microbiome - Clinical", mi)):
                if (variant, hi, learner) not in store:
                    continue
                c1, c2 = store[(variant, a, learner)], store[(variant, hi, learner)]
                p1, p2 = np.nanmean(c1["prob"], 1), np.nanmean(c2["prob"], 1)
                bd = [roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]) for i in BI]
                lo, hi_ = pct(bd); d, se, dl, dh, pv = delong(y, p1, p2)
                f = c1["folds"].merge(c2["folds"], on=["seed", "fold"], suffixes=("_1", "_2"))
                f["d"] = f.fold_auc_2 - f.fold_auc_1; fs = f.dropna(subset=["d"]).groupby("seed").d.mean()
                delta.append(dict(variant=variant, feature_space=space, contrast=lab, learner=LEARN[learner],
                                  estB_auc_clinical=roc_auc_score(y, p1), estB_auc_other=roc_auc_score(y, p2),
                                  delta_auroc_estB=d, boot_ci_lo=lo, boot_ci_hi=hi_,
                                  boot_ci_excludes_0=bool(lo > 0 or hi_ < 0),
                                  delong_se=se, delong_ci_lo=dl, delong_ci_hi=dh, delong_p=pv,
                                  delta_estA_mean=fs.mean(), delta_estA_sd_across_seeds=fs.std(ddof=1) if len(fs) > 1 else np.nan,
                                  n_folds_delta_gt0=int((f.d > 0).sum()), n_folds_delta_eq0=int((f.d == 0).sum()),
                                  n_folds_delta_lt0=int((f.d < 0).sum()), n_folds_compared=int(f.d.notna().sum())))

pf = pd.DataFrame(perf); pf.to_csv(f"{TAB}/T_triage_performance.csv", index=False)
dl = pd.DataFrame(delta)
prim = dl[(dl.variant == "main") & dl.feature_space.str.startswith("primary") &
          (dl.contrast == "Combined - Clinical") & (dl.learner == "XGBoost")]
dl["prefixed_verdict_row"] = False
if len(prim):
    dl.loc[prim.index, "prefixed_verdict_row"] = True
dl.to_csv(f"{TAB}/T_triage_delta_auroc.csv", index=False)
pd.concat(thr_rows).to_csv(f"{TAB}/T_triage_thresholds.csv", index=False)
ss = pd.DataFrame(sens_rows); ss.to_csv(f"{TAB}/T_triage_sens_spec_summary.csv", index=False)
pd.concat(oof).merge(GROUP.rename("Group"), left_on="SampleID", right_index=True).to_csv(
    f"{RES}/oof_probabilities_by_seed.csv", index=False)

P("\n## VERDICT (pre-fixed rule: XGBoost, primary feature space, estimator B, stratified bootstrap)")
if len(prim) and store[("main", "comb", "xgb")]["complete"]:
    r = prim.iloc[0]
    v = "Improvement" if (r.boot_ci_lo > 0) else "No improvement"
    P(f"{v}: dAUROC = {r.delta_auroc_estB:+.3f}, 95% CI {r.boot_ci_lo:+.3f} to {r.boot_ci_hi:+.3f} "
      f"(DeLong {r.delong_ci_lo:+.3f} to {r.delong_ci_hi:+.3f}, P = {r.delong_p:.3f})")
else:
    P("main variant incomplete - no verdict")

P("\n## performance"); md(pf.drop(columns=["model_label", "n_seeds", "auprc_baseline"]))
P("\n## delta AUROC"); md(dl)
P("\n## sensitivity / specificity"); md(ss)

# ---------------------------------------------------------------- reclassification (main)
for learner in LEARN:
    if ("main", "comb", learner) not in store:
        continue
    y = T.get_data("main")["y"]; ids = np.array(T.get_data("main")["ids"])
    rc = pd.DataFrame(dict(SampleID=ids, Group=GROUP.loc[ids].values, y=y))
    for m in NAME:
        c = store[("main", m, learner)]
        rc[f"prob_{m}"] = np.nanmean(c["prob"], 1); rc[f"votes_{m}"] = c["above"].sum(1)
        rc[f"above_{m}"] = c["above"].sum(1) >= 3
    for sp, (a, b) in {"primary": ("clin", "comb"), "comparator11": ("clin", "comb11")}.items():
        rc[f"category_{sp}"] = np.select(
            [~rc[f"above_{a}"] & rc[f"above_{b}"], rc[f"above_{a}"] & rc[f"above_{b}"],
             rc[f"above_{a}"] & ~rc[f"above_{b}"]], ["moved above (Clin below -> Comb above)", "above in both",
                                                     "moved below (Clin above -> Comb below)"], "below in both")
    rc.insert(0, "learner", LEARN[learner])
    store[("rc", learner)] = rc
    P(f"\n## reclassification, {LEARN[learner]} (in-fold Youden, majority of 5 seeds)")
    for sp in ("primary", "comparator11"):
        for yy, nm in ((1, "CDM"), (0, "non-CDM")):
            s = rc[rc.y == yy]
            P(f"  [{sp}] {nm}: " + "; ".join(f"{k} = {len(g)}" + (f" ({', '.join(g.SampleID)})" if (yy == 1 or 'moved above' in k) else "")
                                           for k, g in s.groupby(f"category_{sp}")))
    P(f"  CDM above threshold: Clinical {int(rc[rc.y==1].above_clin.sum())}/14, Combined {int(rc[rc.y==1].above_comb.sum())}/14, "
      f"Combined-11 {int(rc[rc.y==1].above_comb11.sum())}/14")
if ("rc", "xgb") in store:
    pd.concat([store[("rc", l)] for l in LEARN if ("rc", l) in store]).to_csv(f"{TAB}/T_triage_reclassification.csv", index=False)
    P("\n## CDM participants (XGBoost): seed-mean probability and votes (k/5 seeds above the in-fold threshold)")
    md(store[("rc", "xgb")].query("y == 1")[["SampleID", "prob_clin", "votes_clin", "prob_comb", "votes_comb", "category_primary",
                                             "prob_comb11", "votes_comb11", "category_comparator11"]])

# ---------------------------------------------------------------- feature selection (main)
if "main" in cells:
    c = cells["main"]; nf = len(c); per, hp = [], []
    cnt_sel, cnt_conf = {}, {}
    for (s, f), d in sorted(c.items()):
        per.append(dict(seed=s, fold=f, n_species_after_prevalence=d["n_prev_kept"]["species"],
                        n_ko_after_prevalence=d["n_prev_kept"]["ko"], n_confirmed=d["n_confirmed"],
                        n_tentative=d["n_tentative"], n_selected=d["n_selected"],
                        n_selected_species=sum(x.startswith("s__") for x in d["confirmed_features"] + d["tentative_features"]),
                        inner_n_selected="/".join(str(q["n_selected"]) for q in d["inner"]),
                        micro_status=d["models"]["micro"]["status"], boruta_seconds=d["boruta_seconds"], seconds=d["seconds"],
                        selected="|".join(d["confirmed_features"] + d["tentative_features"])))
        for ft in d["confirmed_features"]:
            cnt_conf[ft] = cnt_conf.get(ft, 0) + 1
        for ft in d["confirmed_features"] + d["tentative_features"]:
            cnt_sel[ft] = cnt_sel.get(ft, 0) + 1
        for m, v in d["models"].items():
            if "best_params_xgb" in v:
                hp.append(dict(seed=s, fold=f, model=m, status=v["status"], **{f"xgb_{k}": q for k, q in v["best_params_xgb"].items()},
                               lr_C=v["best_C_lr"], inner_auc_xgb=v["inner_auc_xgb"], inner_auc_lr=v["inner_auc_lr"],
                               thr_xgb=v["thr_xgb"], thr_lr=v["thr_lr"]))
    per = pd.DataFrame(per); per.to_csv(f"{TAB}/T_triage_feature_selection_per_fold.csv", index=False)
    pd.DataFrame(hp).to_csv(f"{RES}/hyperparameters_per_fold.csv", index=False)
    o11 = T.SP11 + T.KO11
    fs = [dict(feature=v, block="clinical (forced in)", n_folds_selected=nf, n_folds_confirmed=np.nan, original_11=False) for v in T.CLIN_VARS]
    fs += [dict(feature=ft, block="species" if ft.startswith("s__") else "KO", n_folds_selected=cnt_sel.get(ft, 0),
                n_folds_confirmed=cnt_conf.get(ft, 0), original_11=ft in o11)
           for ft in sorted(set(cnt_sel) | set(o11), key=lambda t: (-cnt_sel.get(t, 0), t))]
    fs = pd.DataFrame(fs); fs["n_folds"] = nf; fs["pct_selected"] = 100 * fs.n_folds_selected / nf
    fs.to_csv(f"{TAB}/T_triage_feature_selection.csv", index=False)
    P(f"\n## feature selection (main, {nf} outer folds)")
    P(f"  after prevalence filter: species median {per.n_species_after_prevalence.median():.0f} "
      f"({per.n_species_after_prevalence.min()}-{per.n_species_after_prevalence.max()}), KO median {per.n_ko_after_prevalence.median():.0f} "
      f"({per.n_ko_after_prevalence.min()}-{per.n_ko_after_prevalence.max()})")
    P(f"  selected (confirmed + tentative): median {per.n_selected.median():.0f} (min {per.n_selected.min()}, max {per.n_selected.max()}); "
      f"confirmed median {per.n_confirmed.median():.0f}; tentative median {per.n_tentative.median():.0f}; species among selected median {per.n_selected_species.median():.0f}")
    P(f"  outer folds with NO microbiome feature selected: {(per.n_selected == 0).sum()}/{nf}; "
      f"inner folds with none: {sum(x == '0' for r in per.inner_n_selected for x in r.split('/'))}/{nf*3}; "
      f"micro status: {per.micro_status.value_counts().to_dict()}; distinct features ever selected: {len(cnt_sel)}")
    P("  top 20 microbiome features:"); md(fs[fs.block != "clinical (forced in)"].head(20)[["feature", "block", "n_folds_selected", "n_folds_confirmed", "pct_selected", "original_11"]], "%.0f")
    P("  original 11:"); md(fs[fs.original_11][["feature", "n_folds_selected", "n_folds_confirmed", "pct_selected"]], "%.0f")
    h = pd.DataFrame(hp)
    P("  XGBoost tuned (folds): " + "; ".join(f"{m}: depth {g.xgb_max_depth.value_counts().sort_index().to_dict()}, spw>1 {(g.xgb_scale_pos_weight > 1).sum()}/{len(g)}"
                                              for m, g in h.groupby("model")))
    P(f"  runtime per fold (s): median {per.seconds.median():.0f}, max {per.seconds.max():.0f}; Boruta share {per.boruta_seconds.sum()/per.seconds.sum():.0%}; CPU-hours {per.seconds.sum()/3600:.2f}")

# ---------------------------------------------------------------- leak diagnostic
lk = []
for variant in sorted(v for v in cells if v.startswith("leak11") or v.startswith("nullfull")):
    for model in (["micro11"] if variant.startswith("leak11") else ["micro", "micro11", "clin", "comb"]):
        for learner in LEARN:
            c = collect(variant, model, learner)
            if c is None or not len(c["folds"]):
                continue
            fd = c["folds"]; okf = fd[fd.status == "ok"]; y = c["y"]; pb = np.nanmean(c["prob"], 1)
            done = ~np.isnan(pb)
            arm = ("(i) group-wise z (original flaw)" if "groupz" in variant else "(ii) group-independent, in-fold")
            lk.append(dict(variant=variant, data="real" if variant.endswith("real") else "permuted",
                           perm_seed=variant.split("_p")[-1] if "_p" in variant else "", arm=arm,
                           pipeline="11 fixed features" if variant.startswith("leak11") else "full pipeline (prevalence + Boruta + tuning)",
                           model=model, learner=LEARN[learner], n_folds=len(fd), n_folds_dropped=len(fd) - len(okf),
                           estA_auc=okf.groupby("seed").fold_auc.mean().mean() if len(okf) else np.nan,
                           estA_auc_dropped_as_half=fd.assign(a=fd.fold_auc.fillna(.5)).groupby("seed").a.mean().mean(),
                           estB_auc=roc_auc_score(y[done], pb[done]) if len(np.unique(y[done])) > 1 else np.nan))
lk = pd.DataFrame(lk)
if len(lk):
    sm = (lk[(lk.data == "permuted") & (lk.pipeline == "11 fixed features")].groupby(["arm", "learner"])
          .agg(n_perm=("estB_auc", "size"), estA_median=("estA_auc", "median"), estA_min=("estA_auc", "min"),
               estA_max=("estA_auc", "max"), estB_median=("estB_auc", "median"), estB_min=("estB_auc", "min"),
               estB_max=("estB_auc", "max")).reset_index())
    sm.insert(0, "variant", "SUMMARY over permutations 9101-9110")
    pd.concat([lk, sm]).to_csv(f"{TAB}/T_triage_leak_diagnostic.csv", index=False)
    P("\n## leak diagnostic: summary over permutations (11 fixed features, micro11 model)"); md(sm.drop(columns="variant"))
    P("\n## leak diagnostic: per variant"); md(lk.drop(columns=["variant", "pipeline"]) if False else lk[["variant", "model", "learner", "n_folds", "n_folds_dropped", "estA_auc", "estA_auc_dropped_as_half", "estB_auc"]])
    for v in sorted(x for x in cells if x.startswith("nullfull")):
        ns = [d["n_selected"] for d in cells[v].values()]
        P(f"  {v}: outer folds {len(ns)}, n_selected per fold {sorted(ns)}")

open(f"{OUT}/logs/LOG_triage_analyze.txt", "w").write("\n".join(log) + "\n")
