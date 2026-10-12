"""work_c/sparcc_rebuild shared code: paths, independent re-computation of network statistics from the stored correlation / pseudo-P
matrices (own code; c1_lib is NOT imported), representative-replicate rule (DESIGN_prefixed.md section 2). Read-only on every input."""
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]: os.environ[v] = "1"
import json, hashlib, shutil, time
import numpy as np, pandas as pd, networkx as nx
REV = "."; OUT = f"{REV}/work_c/sparcc_rebuild"
P9 = f"{REV}/sparcc_runs"; C1 = f"{REV}/work_a/C1_sparcc"; ORIG = f"{REV}/input_sparcc"
GROUPS = ["NC", "DM", "Cirrhosis", "CDM"]; SUBG = GROUPS[:3]; NEXP = {"NC": 59, "DM": 46, "Cirrhosis": 30, "CDM": 14}
F_REP = f"{P9}/aggregate/sparcc_subsample_replicates.csv"; F_TAX = f"{C1}/inputs/taxon_info.csv"; F_DRAWS = f"{C1}/inputs/subsample_draws.csv"
NETDIR = {"filtered": f"{P9}/filtered_run/net/W", "unfiltered": f"{C1}/net/all1338"}; PREFIX = {"filtered": "prev3W", "unfiltered": "all1338"}
STATS = ["edges", "modularity", "cross_phylum_frac"]            # the three statistics of the representative-replicate rule
R_LO, R_HI, P_CUT = 0.5, 0.9, 0.05                               # edge rule as in the code that produced the networks

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def fresh(p):
    """never overwrite: an existing output is moved to _discard/<timestamp>/ first"""
    if os.path.exists(p):
        d = f"{OUT}/_discard/{time.strftime('%Y%m%d_%H%M%S')}"; os.makedirs(d, exist_ok=True); shutil.move(p, f"{d}/{os.path.basename(p)}")
    return p
def net_id(an, g, k): return f"{PREFIX[an]}__{g}__" + ("full" if k < 0 else f"sub{k:03d}")
PHY = pd.read_csv(F_TAX).set_index("taxon").phylum_metaphlan

def network(an, g, k):
    """Edge list and statistics re-computed from the stored npz (upper-triangle correlations + permutation counts). Each undirected edge once."""
    z = np.load(f"{NETDIR[an]}/{net_id(an, g, k)}.npz", allow_pickle=True)
    taxa = np.array([str(t) for t in z["taxa"]]); D = len(taxa); cor = z["cor_upper"].astype(np.float64); p = z["nsig_upper"] / float(z["n_perm"])
    assert cor.size == D * (D - 1) // 2 and int(z["n_perm"]) == 100
    i, j = np.triu_indices(D, 1); a = np.abs(cor); keep = (a > R_LO) & (a < R_HI) & (p < P_CUT)
    i, j, r = i[keep], j[keep], cor[keep]; phy = PHY.reindex(taxa).values; assert not pd.isna(phy).any()
    g_ = nx.Graph(); g_.add_edges_from(zip(i.tolist(), j.tolist()))          # integer ids in row-major upper-triangle order (Louvain depends on insertion order)
    assert g_.number_of_edges() == len(r)                                    # no pair entered twice
    N = g_.number_of_nodes(); E = len(r); cross = phy[i] != phy[j]
    mod = max(nx.community.modularity(g_, nx.community.louvain_communities(g_, seed=s)) for s in range(10))
    st = dict(analysis=an, group=g, draw=k, id=net_id(an, g, k), n_samples=len(z["samples"]), n_taxa_in_matrix=D, nodes=N, edges=E, edges_pos=int((r > 0).sum()), edges_neg=int((r < 0).sum()),
              density=E / (N * (N - 1) / 2), mean_degree=2 * E / N, modularity=float(mod), cross_phylum_n=int(cross.sum()), cross_phylum_frac=float(cross.mean()),
              n_components=nx.number_connected_components(g_), max_degree=max(d for _, d in g_.degree()))
    ed = pd.DataFrame(dict(a=taxa[i], b=taxa[j], phylum_a=phy[i], phylum_b=phy[j], r=r, cross=cross))
    return st, ed, [str(s) for s in z["samples"]]

def representative(R, an, g):
    """DESIGN section 2: z-standardise edges / modularity / cross-phylum over the 100 replicates (mean, SD ddof=1); replicate nearest (Euclidean) to the
    standardised per-statistic medians; ties -> lowest draw index. Returns the table of all 100 replicates sorted by (distance, draw)."""
    d = R[(R.analysis == an) & (R.group == g) & (R.draw >= 0)].sort_values("draw").reset_index(drop=True); assert len(d) == 100 and list(d.draw) == list(range(100))
    X = d[STATS].values.astype(float); mu = X.mean(0); sd = X.std(0, ddof=1); med = np.median(X, 0)
    d["distance"] = np.sqrt((((X - mu) / sd - (med - mu) / sd) ** 2).sum(1))
    for c, m in zip(STATS, med): d[c + "_median"] = m
    return d.sort_values(["distance", "draw"], kind="mergesort").reset_index(drop=True)

if __name__ == "__main__":                                       # self-check on a toy matrix: 4 taxa, pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    X = np.array([[1, 2, 3], [5, 2.1, 3], [9, 0, 1.0]]); mu = X.mean(0); sd = X.std(0, ddof=1); med = np.median(X, 0)
    assert np.argmin(np.sqrt((((X - mu) / sd - (med - mu) / sd) ** 2).sum(1))) == 1
    print("selfcheck ok")
