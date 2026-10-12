#!/usr/bin/env python
"""Triage (CDM vs non-CDM) nested CV. ADAPTED COPY of project/nested_cv/a1v2_nested_cv.py (Fig 2 runner;
sha256 a1109862...45ffe, unmodified copy kept next to this file).

Kept literally from Fig 2: PSEUDO / PREV / OUTER / INNER / BORUTA_ITER / SEEDS / C_GRID, xgb_grid(),
boruta_select(), the prevalence + log10 + StandardScaler block (Fig 2 "combined" input), the
StratifiedKFold calls (outer and inner, random_state=seed), the two learners and their arguments,
inner-AUROC tuning by nanargmax(nanmean), final features = Boruta confirmed | tentative.

Changed (see DESIGN_prefixed.md): outcome = CDM vs all others (n=149); one Boruta result per fold is
shared by Microbiome-only and Combined; clinical block (7 variables, forced, no selection); fixed
11-feature comparator; in-fold Youden threshold; permutation variants; per-fold JSON.

Usage: python triage_nested_cv.py --variants main --seeds 42 --workers 5
"""
import argparse, functools, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
R = "."
A1 = f"{R}/nested_cv"
OUT = f"{R}/work_c/triage_nested"
FOLDS = f"{OUT}/results/folds"
CLIN_FILE = f"{R}/genome_figures/inputs_corrected/clinical_corrected.csv"
PSEUDO = {"species": 1e-5, "ko": 1e-9}          # as Fig 2
PREV, OUTER, INNER, BORUTA_ITER = 0.10, 5, 3, 1000
SEEDS = [42, 43, 44, 45, 46]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CLIN_VARS = ["Age", "Sex", "BMI", "AST", "ALT", "PLATELET", "Cholesterol"]
SP11 = ["s__Lactobacillus_crispatus", "s__Veillonella_parvula"]
KO11 = ["K00075", "K01512", "K03060", "K03101", "K03116", "K03687", "K03799", "K04042", "K10716"]
_G = {}


def xgb_grid(y):                                  # verbatim Fig 2
    pos = float(max((y == 0).sum() / max((y == 1).sum(), 1), 1.0))
    return [dict(n_estimators=ne, max_depth=md, learning_rate=lr,
                 subsample=ss, scale_pos_weight=sp)
            for ne in (100, 300) for md in (3, 5, 7) for lr in (0.01, 0.1)
            for ss in (0.8, 1.0) for sp in (1, pos)]


def boruta_select(X, y, seed, threads):           # verbatim Fig 2
    """(confirmed_mask, tentative_mask). No fallback here by design."""
    rf = RandomForestClassifier(n_jobs=threads, max_depth=5,
                                class_weight="balanced", random_state=seed)
    b = BorutaPy(rf, n_estimators="auto", max_iter=BORUTA_ITER,
                 perc=100, alpha=0.05, random_state=seed)
    b.fit(X, y)
    return b.support_.copy(), b.support_weak_.copy()


def prep_fold(D, tr, te):
    """Fig 2 prep_fold with name == 'combined': prevalence + log + scale, learned on outer train."""
    btr, bte, names, nkeep = [], [], [], {}
    for key in ["species", "ko"]:
        M = D[key]
        keep = (M[tr] > 0).mean(0) >= PREV
        btr.append(np.log10(M[tr][:, keep] + PSEUDO[key]))
        bte.append(np.log10(M[te][:, keep] + PSEUDO[key]))
        names += list(np.asarray(_G[key + "_cols"])[keep])
        nkeep[key] = int(keep.sum())
    Xtr, Xte = np.hstack(btr), np.hstack(bte)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), names, nkeep


def scale(X, tr, te):
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), sc.transform(X[te])


def mk_xgb(seed, gp):                             # arguments as Fig 2
    return XGBClassifier(eval_metric="logloss", n_jobs=1, tree_method="hist",
                         random_state=seed, **gp)


def mk_lr(C):                                     # arguments as Fig 2
    return LogisticRegression(penalty="l2", C=C, max_iter=5000, solver="lbfgs")


def youden(y, p):
    """Threshold maximising sens + spec - 1 (first maximum = highest threshold on ties); classify p >= thr."""
    m = ~np.isnan(p)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return None
    fpr, tpr, thr = roc_curve(y[m], p[m], drop_intermediate=False)
    return float(thr[int(np.argmax(tpr - fpr))])


def tune_fit(inner_sets, final, ytr, inner, grid, seed):
    """Fig 2 inner loop for one feature block. inner_sets[k] = (A, B) or None; final = (A, B) or None."""
    n = len(ytr)
    xs = np.full((len(grid), INNER), np.nan); ls = np.full((len(C_GRID), INNER), np.nan)
    px = np.full((len(grid), n), np.nan); pl = np.full((len(C_GRID), n), np.nan)
    for k, (itr, iva) in enumerate(inner):
        if inner_sets[k] is None or len(np.unique(ytr[iva])) < 2:
            continue
        A, B = inner_sets[k]
        for gi, gp in enumerate(grid):
            px[gi, iva] = mk_xgb(seed, gp).fit(A, ytr[itr]).predict_proba(B)[:, 1]
            xs[gi, k] = roc_auc_score(ytr[iva], px[gi, iva])
        for ci, C in enumerate(C_GRID):
            pl[ci, iva] = mk_lr(C).fit(A, ytr[itr]).predict_proba(B)[:, 1]
            ls[ci, k] = roc_auc_score(ytr[iva], pl[ci, iva])
    if np.all(np.isnan(xs)):
        return dict(status="inner_all_empty")
    gi = int(np.nanargmax(np.nanmean(xs, axis=1))); ci = int(np.nanargmax(np.nanmean(ls, axis=1)))
    out = dict(status="ok", best_params_xgb=grid[gi], best_C_lr=C_GRID[ci],
               inner_auc_xgb=float(np.nanmean(xs[gi])), inner_auc_lr=float(np.nanmean(ls[ci])),
               thr_xgb=youden(ytr, px[gi]), thr_lr=youden(ytr, pl[ci]))
    if final is None:                             # Fig 2: drop, never fall back
        return dict(out, status="dropped_zero_features")
    A, B = final
    out["p_xgb"] = mk_xgb(seed, grid[gi]).fit(A, ytr).predict_proba(B)[:, 1].tolist()
    out["p_lr"] = mk_lr(C_GRID[ci]).fit(A, ytr).predict_proba(B)[:, 1].tolist()
    return out


@functools.lru_cache(maxsize=2)
def get_data(variant):
    """variant: main | nullfull_p<seed> | leak11_{groupz,infold}_{p<seed>,real}"""
    lab = _G["lab"]
    keep = np.ones(len(lab), bool)
    grp = lab.values[keep]
    D = dict(ids=list(lab.index[keep]), y=(grp == "CDM").astype(int), clin=_G["clin"][keep],
             species=_G["species"][keep], ko=_G["ko"][keep])
    tag = variant.split("_")[-1]
    if tag.startswith("p"):                       # move whole microbiome profiles between participants
        perm = np.random.default_rng(int(tag[1:])).permutation(len(grp))
        D["species"], D["ko"] = D["species"][perm], D["ko"][perm]
    m11 = np.hstack([np.log10(D["species"][:, _G["sp11"]] + PSEUDO["species"]),
                     np.log10(D["ko"][:, _G["ko11"]] + PSEUDO["ko"])])
    if "groupz" in variant:                       # the original's flaw: z within each of the 4 groups
        z = np.zeros_like(m11)
        for g in np.unique(grp):
            r = grp == g
            sd = m11[r].std(0, ddof=1)
            z[r] = np.where(sd > 0, (m11[r] - m11[r].mean(0)) / np.where(sd > 0, sd, 1), 0.0)
        m11 = z
    D["m11"] = m11
    return D


def run_fold(variant, seed, fi, threads=1):
    t0 = time.time()
    D = get_data(variant)
    y = D["y"]
    tr, te = list(StratifiedKFold(OUTER, shuffle=True,
                                  random_state=seed).split(np.zeros(len(y)), y))[fi]
    ytr, yte = y[tr], y[te]
    inner = list(StratifiedKFold(INNER, shuffle=True,
                                 random_state=seed).split(np.zeros(len(tr)), ytr))
    grid = xgb_grid(ytr)
    Ctr, Cte = scale(D["clin"], tr, te)
    Mtr, Mte = scale(D["m11"], tr, te)
    res = dict(variant=variant, seed=seed, fold=fi, test_idx=te.tolist(), y_test=yte.tolist(),
               n_pos_test=int(yte.sum()), inner_val_idx=[tr[iva].tolist() for _, iva in inner],
               inner_val_pos=[int(ytr[iva].sum()) for _, iva in inner], models={})
    T = lambda sets, fin: tune_fit(sets, fin, ytr, inner, grid, seed)
    I = lambda Xtr: [(Xtr[itr], Xtr[iva]) for itr, iva in inner]
    res["models"]["micro11"] = T(I(Mtr), (Mtr, Mte))
    if variant.startswith("leak11"):
        return dict(res, seconds=round(time.time() - t0, 1))
    res["models"]["clin"] = T(I(Ctr), (Ctr, Cte))
    C11tr, C11te = np.hstack([Ctr, Mtr]), np.hstack([Cte, Mte])
    res["models"]["comb11"] = T(I(C11tr), (C11tr, C11te))

    Ztr, Zte, feat, nkeep = prep_fold(D, tr, te)
    tb = time.time()
    inner_status, msets, csets = [], [], []
    for k, (itr, iva) in enumerate(inner):        # Boruta on the inner TRAIN fold (Fig 2 step 3)
        conf, tent = boruta_select(Ztr[itr], ytr[itr], seed, threads)
        sel = conf | tent
        inner_status.append(dict(inner_fold=k, n_confirmed=int(conf.sum()),
                                 n_tentative=int(tent.sum()), n_selected=int(sel.sum()),
                                 selected=[feat[i] for i in np.flatnonzero(sel)]))
        msets.append((Ztr[itr][:, sel], Ztr[iva][:, sel]) if sel.sum() else None)
        csets.append((np.hstack([Ctr[itr], Ztr[itr][:, sel]]), np.hstack([Ctr[iva], Ztr[iva][:, sel]])))
    conf, tent = boruta_select(Ztr, ytr, seed, threads)   # Boruta on the full outer train (step 5)
    sel = conf | tent
    res.update(n_prev_kept=nkeep, inner=inner_status, n_confirmed=int(conf.sum()),
               n_tentative=int(tent.sum()), n_selected=int(sel.sum()),
               confirmed_features=[feat[i] for i in np.flatnonzero(conf)],
               tentative_features=[feat[i] for i in np.flatnonzero(tent)],
               boruta_seconds=round(time.time() - tb, 1), train_prevalence=float(ytr.mean()))
    res["models"]["micro"] = T(msets, (Ztr[:, sel], Zte[:, sel]) if sel.sum() else None)
    res["models"]["comb"] = T(csets, (np.hstack([Ctr, Ztr[:, sel]]), np.hstack([Cte, Zte[:, sel]])))
    return dict(res, seconds=round(time.time() - t0, 1))


def _init():
    lab = pd.read_csv(f"{A1}/labels.csv", index_col=0)["Group"]
    sp = pd.read_csv(f"{A1}/species_s_matrix.csv", index_col=0).loc[lab.index]
    ko = pd.read_csv(f"{A1}/ko_matrix.csv", index_col=0).loc[lab.index]
    cl = pd.read_csv(CLIN_FILE, index_col=0).loc[lab.index]
    assert (cl["Group"] == lab).all() and cl[CLIN_VARS].notna().all().all()
    _G.update(lab=lab, species=sp.values, ko=ko.values, clin=cl[CLIN_VARS].values.astype(float),
              species_cols=list(sp.columns), ko_cols=list(ko.columns),
              sp11=[sp.columns.get_loc(c) for c in SP11], ko11=[ko.columns.get_loc(c) for c in KO11])


def _task(a):
    r = run_fold(*a)
    json.dump(r, open(f"{FOLDS}/{a[0]}_s{a[1]}_f{a[2]}.json", "w"))
    return a, r


def precheck():
    """Fold composition only (no model): positives per fold. Written before the freeze."""
    _init(); rows = []
    for v in ("main",):
        D = get_data(v); y = D["y"]; ids = np.array(D["ids"])
        for s in SEEDS:
            for fi, (tr, te) in enumerate(StratifiedKFold(OUTER, shuffle=True, random_state=s)
                                          .split(np.zeros(len(y)), y)):
                ip = [int(y[tr][iva].sum()) for _, iva in StratifiedKFold(
                    INNER, shuffle=True, random_state=s).split(np.zeros(len(tr)), y[tr])]
                for i in te:
                    rows.append(dict(variant=v, seed=s, outer_fold=fi, SampleID=ids[i], y=int(y[i]),
                                     n_test=len(te), n_pos_test=int(y[te].sum()),
                                     inner_val_pos="/".join(map(str, ip))))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/results/fold_membership.csv", index=False)
    for v, d in df.groupby("variant"):
        f = d.drop_duplicates(["seed", "outer_fold"])
        print(v, "n =", d.SampleID.nunique(), "| outer test positives per fold:",
              f.n_pos_test.value_counts().sort_index().to_dict(), "| inner validation positives:",
              pd.Series("/".join(f.inner_val_pos).split("/")).value_counts().sort_index().to_dict())
        assert f.n_pos_test.min() >= 1 and min(map(int, "/".join(f.inner_val_pos).split("/"))) >= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="main")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--precheck", action="store_true")
    args = ap.parse_args()
    if args.precheck:
        return precheck()
    assert args.workers <= 8
    logp = f"{OUT}/logs/LOG_triage_nested_cv.txt"

    def log(m):
        print(m, flush=True)
        with open(logp, "a") as fh:
            fh.write(m + "\n")

    jobs = [(v, int(s), f) for v in args.variants.split(",") for s in args.seeds.split(",")
            for f in range(OUTER) if not os.path.exists(f"{FOLDS}/{v}_s{s}_f{f}.json")]
    log(f"\n{'='*78}\n{time.strftime('%F %T')} variants={args.variants} seeds={args.seeds} "
        f"workers={args.workers} threads=1 | pending fold-jobs: {len(jobs)}")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        futs = [ex.submit(_task, j) for j in jobs]
        for done, fut in enumerate(as_completed(futs), 1):
            (v, s, f), r = fut.result()            # AUROC deliberately NOT logged (no peeking)
            log(f"  [{done}/{len(jobs)}] {v}/s{s}/f{f} sel={r.get('n_selected', '-')} "
                f"boruta={r.get('boruta_seconds', '-')}s total={r['seconds']}s")
    log(f"COMPLETE {args.variants} seeds={args.seeds} in {(time.time()-t0)/60:.1f} min "
        f"({time.strftime('%F %T')})")


if __name__ == "__main__":
    main()
