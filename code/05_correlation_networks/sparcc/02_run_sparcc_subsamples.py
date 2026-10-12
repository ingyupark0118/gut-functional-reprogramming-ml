#!/usr/bin/env python
"""P9_SparCC: rare-species-filtered SparCC networks (design: ../DESIGN_prefixed.md). c1_lib (C1 fast re-implementation) is imported unchanged.
Filter sets:  W = within each analysed sample set keep species with count > 0 in >= 3 samples of that set (PRIMARY)
              B = keep species with count > 0 in >= 3 samples of the FULL group, then sub-sample (sensitivity)
Resumable: a network is finished when net/<set>/<id>.json exists (written last, atomically). Nothing is overwritten.
usage: p9_run_filtered.py --set W|B [--nproc 4] [--only ID,ID]"""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]: os.environ[v] = "1"
import sys, json, time, argparse, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool
REV = "."; C1 = f"{REV}/work_a/C1_sparcc"; OUT = f"{REV}/sparcc_runs/filtered_run"
sys.path.insert(0, f"{C1}/scripts")
import c1_lib as L
SEED, GROUPS, MINPREV = 20261002, ["NC", "DM", "Cirrhosis", "CDM"], 3
SETIDX = {"W": 0, "B": 2}                      # W uses the same network seed as the C1 all1338 run; B a new stream
N_ITER, N_PERM, PERM_ITER = 100, 100, 20       # identical to C1

cnt = pd.read_csv(f"{C1}/inputs/species_s_counts_raw.csv", index_col=0)
lab = pd.read_csv(f"{REV}/nested_cv/labels.csv", index_col=0)["Group"]
info = pd.read_csv(f"{C1}/inputs/taxon_info.csv"); assert list(info.taxon) == list(cnt.columns)
dr = pd.read_csv(f"{C1}/inputs/subsample_draws.csv")

def net_id(s, g, k): return f"prev3{s}__{g}__" + ("full" if k < 0 else f"sub{k:03d}")
def sample_ids(g, k):
    if k < 0: return list(lab.index[lab == g])
    row = dr[(dr.group == g) & (dr.draw == k)].iloc[0]; return [row[c] for c in dr.columns[2:]]
def keep_cols(s, g, ids):
    ref = ids if s == "W" else list(lab.index[lab == g])
    return ((cnt.loc[ref].values > 0).sum(0) >= MINPREV)

def extra(cor, nsig, present):
    """mean |r| over all pairs among PRESENT retained species, over edges, and number of pairs with |r| > 0.6 (no P, no upper bound)."""
    D = len(present); i, j = np.triu_indices(D, 1); a = np.abs(cor); ok = present[i] & present[j]
    e = (a > L.R_LO) & (a < L.R_HI) & (nsig / float(N_PERM) < L.P_CUT)
    return dict(n_pairs_present=int(ok.sum()), mean_abs_r_all=float(a[ok].mean()), mean_abs_r_allinput=float(a.mean()),
                mean_abs_r_edges=float(a[e].mean()) if e.any() else float("nan"),
                pairs_abs_r_gt_06=int((a[ok] > 0.6).sum()), pairs_abs_r_gt_06_allinput=int((a > 0.6).sum()))

def run_one(job):
    s, g, k = job; nid = net_id(s, g, k); odir = f"{OUT}/net/{s}"; os.makedirs(odir, exist_ok=True); fj = f"{odir}/{nid}.json"
    if os.path.exists(fj): return nid, None
    t0 = time.time(); ids = sample_ids(g, k); cols = keep_cols(s, g, ids)
    X = cnt.loc[ids].values[:, cols]; taxa = info.taxon.values[cols]; phy = info.phylum_metaphlan.values[cols]
    base = dict(id=nid, filter_set=s, group=g, draw=k, n_samples=len(ids), n_species_retained=int(cols.sum()),
                n_species_present_unfiltered=int((cnt.loc[ids].values.sum(0) > 0).sum()))
    if X.shape[1] < 14:                           # c1_lib refuses D <= x_iter + 3; recorded, not worked around
        m = dict(base, status="not_run_fewer_than_14_species")
    else:
        seed = [SEED, SETIDX[s], GROUPS.index(g), k + 1]
        cor, nsig, pinfo = L.pipeline(X, seed, N_ITER, N_PERM, PERM_ITER); t1 = time.time()
        present = X.sum(0) > 0
        m, (ei, ej, er, ep) = L.metrics(cor, nsig, N_PERM, phy, present, n_seeds=10)     # modularity = best of Louvain seeds 0-9 (C1 definition)
        m.update(extra(cor, nsig, present)); m.update(pinfo); m.update(base)
        np.savez_compressed(f"{odir}/{nid}.npz", cor_upper=cor.astype(np.float32), nsig_upper=nsig, taxa=taxa, samples=np.array(ids), n_perm=N_PERM)
        pd.DataFrame(dict(taxon_a=taxa[ei], taxon_b=taxa[ej], phylum_a=phy[ei], phylum_b=phy[ej], sparcc_r=np.round(er, 6), p_perm=ep)).to_csv(f"{odir}/{nid}_edges.csv", index=False)
        m.update(status="ok", seed=str(seed), sec_sparcc=round(t1 - t0, 1), edge_rule=f"{L.R_LO}<|r|<{L.R_HI} & p<{L.P_CUT}", min_prevalence=MINPREV)
    m.update(sec_total=round(time.time() - t0, 1), finished=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(fj + ".tmp", "w") as fh: json.dump(m, fh, indent=1)
    os.replace(fj + ".tmp", fj); return nid, m

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["W", "B"]); ap.add_argument("--nproc", type=int, default=4); ap.add_argument("--only", default="")
    a = ap.parse_args(); assert a.nproc <= 4
    L._selfcheck()
    # check of the filter itself on a toy table: species present in 2 samples is dropped, in 3 kept
    T = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]]); assert ((T > 0).sum(0) >= MINPREV).tolist() == [True, False, False]
    js = [(a.set, g, -1) for g in GROUPS] + [(a.set, g, k) for g in GROUPS[:3] for k in range(100)]
    if a.set == "B": js = [j for j in js if j[2] >= 0]          # full-group networks of B are identical in definition to W
    if a.only: js = [j for j in js if net_id(*j) in a.only.split(",")]
    todo = [j for j in js if not os.path.exists(f"{OUT}/net/{j[0]}/{net_id(*j)}.json")]
    print(f"[{datetime.datetime.now():%F %T}] set {a.set}: jobs {len(js)}, to do {len(todo)}, nproc {a.nproc}, numpy {np.__version__}, pandas {pd.__version__}, load {open('/proc/loadavg').read().split()[0]}", flush=True)
    t0 = time.time()
    with Pool(a.nproc) as pool:
        for nid, m in pool.imap_unordered(run_one, todo, chunksize=1):
            if m is None: continue
            if m["status"] != "ok": print(f"[{datetime.datetime.now():%T}] {nid} {m['status']} D={m['n_species_retained']}", flush=True); continue
            print(f"[{datetime.datetime.now():%T}] {nid} n={m['n_samples']} D={m['n_species_retained']} nodes={m['nodes']} edges={m['edges']} (+{m['edges_pos']}/-{m['edges_neg']}) "
                  f"mod={m['modularity']:.3f} cross={m['cross_phylum_frac']:.3f} mean|r|={m['mean_abs_r_all']:.3f} clr={m['clr_fallback_main']}/{m['clr_fallback_perm']} {m['sec_total']}s", flush=True)
    print(f"[{datetime.datetime.now():%F %T}] done in {(time.time()-t0)/60:.1f} min", flush=True)
