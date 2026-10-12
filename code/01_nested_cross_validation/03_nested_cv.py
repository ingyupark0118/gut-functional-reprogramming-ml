#!/usr/bin/env python
"""A1-v2 nested CV. Corrects the six design defects of v1 (a1_nested_cv.py, kept).

  v1                                        v2
  prevalence filter over all 149            learned inside each outer training fold
  Boruta once, before the inner CV          re-run inside every inner training fold
  <3 selected -> use ALL features           1-2 kept as is; 0 -> fold dropped
  fallback counted as "selected"            confirmed / tentative / dropped kept apart
  s__ and SGB rows mixed                    s__ only (SGB is a separate sensitivity input)
  point estimate and CI from different      estimator A and estimator B reported apart
    statistics

Per outer fold (spec section 3):
  1. prevalence >= 10% learned on outer train only, applied to test
  2. log10(x + pseudo) + standardise fit on outer train only
  3. inner 3-fold: Boruta on the inner TRAIN fold, models scored on inner validation
  4. hyperparameters chosen from the inner CV
  5. Boruta re-run on the full outer train -> features for the final model
  6. predict the outer test

Boruta output is shared by both models within a fold. This is the published design;
R1-M5d asked for justification, not a change, so it is kept and logged.

Usage: python a1v2_nested_cv.py --comps g1 --workers 12 --threads 3
"""
import argparse, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
A1 = "./nested_cv"
TAG = os.environ.get("A1_TAG", "A1v2")               # A1v3: ConQuR-corrected species
RES = os.path.join(A1, os.environ.get("A1_RES", "results_v2"))
SPECIES = os.environ.get("A1_SPECIES", f"{A1}/species_s_matrix.csv")
PSEUDO = {"species": 1e-5, "ko": 1e-9}          # spec section 3
PREV, OUTER, INNER, BORUTA_ITER = 0.10, 5, 3, 1000
SEEDS = [42, 43, 44, 45, 46]                     # fixed before the run
COMPARISONS = {"g1": ("NC", "DM"), "g2": ("Cirrhosis", "CDM"),
               "g3": ("NC", "Cirrhosis"), "g4": ("DM", "CDM")}
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
_G = {}


def xgb_grid(y):
    pos = float(max((y == 0).sum() / max((y == 1).sum(), 1), 1.0))
    return [dict(n_estimators=ne, max_depth=md, learning_rate=lr,
                 subsample=ss, scale_pos_weight=sp)
            for ne in (100, 300) for md in (3, 5, 7) for lr in (0.01, 0.1)
            for ss in (0.8, 1.0) for sp in (1, pos)]


def boruta_select(X, y, seed, threads):
    """(confirmed_mask, tentative_mask). No fallback here by design."""
    rf = RandomForestClassifier(n_jobs=threads, max_depth=5,
                                class_weight="balanced", random_state=seed)
    b = BorutaPy(rf, n_estimators="auto", max_iter=BORUTA_ITER,
                 perc=100, alpha=0.05, random_state=seed)
    b.fit(X, y)
    return b.support_.copy(), b.support_weak_.copy()


def prep_fold(comp_mask, tr, te, name):
    """Prevalence filter + log + scale, all learned on the outer training fold."""
    btr, bte, names = [], [], []
    for key in (["species", "ko"] if name == "combined" else [name]):
        M = _G[key][comp_mask]
        keep = (M[tr] > 0).mean(0) >= PREV
        btr.append(np.log10(M[tr][:, keep] + PSEUDO[key]))
        bte.append(np.log10(M[te][:, keep] + PSEUDO[key]))
        names += list(np.asarray(_G[key + "_cols"])[keep])
    Xtr, Xte = np.hstack(btr), np.hstack(bte)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), names


def run_fold(name, comp, seed, fi, threads):
    t0 = time.time()
    lab = _G["lab"]
    a, b = COMPARISONS[comp]
    cm = lab.isin([a, b]).values
    y = (lab[cm] == b).astype(int).values
    tr, te = list(StratifiedKFold(OUTER, shuffle=True,
                                  random_state=seed).split(np.zeros(len(y)), y))[fi]
    Ztr, Zte, feat = prep_fold(cm, tr, te, name)
    ytr, yte = y[tr], y[te]

    inner = list(StratifiedKFold(INNER, shuffle=True,
                                 random_state=seed).split(Ztr, ytr))
    grid = xgb_grid(ytr)
    xs = np.full((len(grid), INNER), np.nan)
    ls = np.full((len(C_GRID), INNER), np.nan)
    inner_status = []
    for k, (itr, iva) in enumerate(inner):
        conf, tent = boruta_select(Ztr[itr], ytr[itr], seed, threads)
        sel = conf | tent
        inner_status.append(dict(inner_fold=k, n_confirmed=int(conf.sum()),
                                 n_tentative=int(tent.sum()), n_selected=int(sel.sum())))
        if sel.sum() == 0 or len(np.unique(ytr[iva])) < 2:
            continue
        A, B = Ztr[itr][:, sel], Ztr[iva][:, sel]
        for gi, gp in enumerate(grid):
            mdl = XGBClassifier(eval_metric="logloss", n_jobs=1, tree_method="hist",
                                random_state=seed, **gp).fit(A, ytr[itr])
            xs[gi, k] = roc_auc_score(ytr[iva], mdl.predict_proba(B)[:, 1])
        for ci, C in enumerate(C_GRID):
            mdl = LogisticRegression(penalty="l2", C=C, max_iter=5000,
                                     solver="lbfgs").fit(A, ytr[itr])
            ls[ci, k] = roc_auc_score(ytr[iva], mdl.predict_proba(B)[:, 1])

    base = dict(fold=fi, inner=inner_status, test_idx=te.tolist(),
                seconds=round(time.time() - t0, 1))
    if np.all(np.isnan(xs)):
        return dict(base, status="inner_all_empty",
                    n_confirmed=0, n_tentative=0, n_selected=0)
    best_xgb = grid[int(np.nanargmax(np.nanmean(xs, axis=1)))]
    best_C = C_GRID[int(np.nanargmax(np.nanmean(ls, axis=1)))]

    conf, tent = boruta_select(Ztr, ytr, seed, threads)
    sel = conf | tent
    if sel.sum() == 0:                       # spec section 4: drop, never fall back
        return dict(base, status="dropped_zero_features",
                    n_confirmed=0, n_tentative=0, n_selected=0)

    A, B = Ztr[:, sel], Zte[:, sel]
    px = XGBClassifier(eval_metric="logloss", n_jobs=1, tree_method="hist",
                       random_state=seed, **best_xgb).fit(A, ytr).predict_proba(B)[:, 1]
    pl = LogisticRegression(penalty="l2", C=best_C, max_iter=5000,
                            solver="lbfgs").fit(A, ytr).predict_proba(B)[:, 1]

    def metrics(p):
        h = (p > .5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, h, labels=[0, 1]).ravel()
        return dict(auc=float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else np.nan,
                    acc=float(accuracy_score(yte, h)),
                    f1=float(f1_score(yte, h, zero_division=0)),
                    sens=float(tp / (tp + fn)) if tp + fn else np.nan,
                    spec=float(tn / (tn + fp)) if tn + fp else np.nan)

    return dict(base, status="ok", n_confirmed=int(conf.sum()),
                n_tentative=int(tent.sum()), n_selected=int(sel.sum()),
                confirmed_features=[feat[i] for i in np.flatnonzero(conf)],
                best_params_xgb=best_xgb, best_C_lr=best_C,
                xgb=metrics(px), lr=metrics(pl),
                p_xgb=px.tolist(), p_lr=pl.tolist(), y_test=yte.tolist())


def _init(threads):
    os.environ.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    lab = pd.read_csv(f"{A1}/labels.csv", index_col=0)["Group"]
    sp = pd.read_csv(SPECIES, index_col=0).loc[lab.index]
    ko = pd.read_csv(f"{A1}/ko_matrix.csv", index_col=0).loc[lab.index]
    _G.update(lab=lab, species=sp.values, ko=ko.values,
              species_cols=list(sp.columns), ko_cols=list(ko.columns))


def _task(a):
    return a, run_fold(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="g1,g2,g3,g4")
    ap.add_argument("--inputs", default="species,ko,combined")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    logp = os.path.join(A1, f"LOG_{TAG}.txt")

    def log(m):
        print(m, flush=True)
        with open(logp, "a") as fh:
            fh.write(m + "\n")

    import hashlib                                  # input provenance (A1v3+)
    for path, env in ((SPECIES, "A1_SPECIES_SHA256"), (f"{A1}/ko_matrix.csv", "A1_KO_SHA256")):
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        log(f"input sha256 {os.path.basename(path)} = {h}")
        if os.environ.get(env) and os.environ[env] != h:
            raise SystemExit(f"S1: {path} sha256 {h} != expected {os.environ[env]}")
    _init(args.threads)
    counts = _G["lab"].value_counts().to_dict()
    if counts != {"NC": 59, "DM": 46, "Cirrhosis": 30, "CDM": 14}:
        raise SystemExit(f"S1: group sizes {counts}")
    log(f"\n{'='*78}\n{TAG} | seeds={SEEDS} outer={OUTER} inner={INNER} "
        f"boruta_iter={BORUTA_ITER} prevalence={PREV:.0%}")
    log(f"pseudocount species={PSEUDO['species']:g} ko={PSEUDO['ko']:g} | "
        f"workers={args.workers} threads={args.threads}")
    log(f"species(s__)={_G['species'].shape} ko={_G['ko'].shape} | S1 check passed")

    jobs = [(n, c, s, f, args.threads) for c in args.comps.split(",")
            for n in args.inputs.split(",") for s in SEEDS for f in range(OUTER)
            if not os.path.exists(os.path.join(RES, f"{n}_{c}_s{s}.json"))]
    log(f"pending fold-jobs: {len(jobs)}")

    store, t0 = {}, time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.threads,)) as ex:
        futs = [ex.submit(_task, j) for j in jobs]
        for done, fut in enumerate(as_completed(futs), 1):
            (n, c, s, f, _), r = fut.result()
            store.setdefault((n, c, s), []).append(r)
            auc = r.get("xgb", {}).get("auc", float("nan"))
            log(f"  [{done}/{len(jobs)}] {n}/{c}/s{s}/f{f} {r['status']} "
                f"sel={r['n_selected']} auc={auc:.3f} ({r['seconds']}s)")
            if len(store[(n, c, s)]) == OUTER:
                fl = sorted(store.pop((n, c, s)), key=lambda x: x["fold"])
                json.dump(dict(input=n, comp=c, seed=s, folds=fl),
                          open(os.path.join(RES, f"{n}_{c}_s{s}.json"), "w"))
                ok = [x for x in fl if x["status"] == "ok"]
                log(f"  [cell] {n}/{c}/s{s}: {len(ok)}/{OUTER} folds usable"
                    + (f"  meanAUC={np.mean([x['xgb']['auc'] for x in ok]):.4f}" if ok else ""))
    log(f"COMPLETE {args.comps} in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
