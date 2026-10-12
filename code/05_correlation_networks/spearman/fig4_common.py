#!/usr/bin/env python
"""Within-group Spearman correlation analysis between two species, nine KOs and 14 clinical variables (Fig. 4A, 4D, 4E).

This module is self-contained (numpy, pandas, scipy only) and computes:
- the 25 nodes (L. crispatus, V. parvula, the nine candidate KOs in the unified KO order, 14 clinical variables) and the 173 node pairs
  (1 species-species, 18 species-KO, 28 species-clinical, 126 KO-clinical);
- load_all(): group labels, the KO tables in two units (KO sum / original), the species table and the clinical table, all in the sample order of the labels;
- spearman_pairs(): vectorised Spearman rho with the two-sided P of the t approximation (df = n - 2);
- main_family(): rho, P and Benjamini-Hochberg q over the 173 pairs within each group, and the edge class
  (|rho| > 0.6 and q < 0.05 = "solid"; otherwise P < 0.05 = "dashed"; otherwise "none").
The clinical variables are used as stored in the clinical table (the MELD column is the stored value, not a recalculated one).
Paths refer to the original project layout ("." = project root). Holds no data values."""
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, t as tdist

REV = "."
A1 = f"{REV}/nested_cv"
F4 = f"{REV}/replot/fig 4_corrheatmap&network/"
K1P = f"{REV}/work_f/fig 1G&S3C KEGG_zscore/KO_merge.csv"   # ORIGINAL unit (K1)
K2P = f"{REV}/nested_cv/ko_matrix.csv"                                 # KO SUM unit (K2)
T3P = f"{REV}/nested_cv/species_s_matrix.csv"                          # raw MetaPhlAn (T3)
LABP = f"{REV}/nested_cv/labels.csv"
LC, VP = "s__Lactobacillus_crispatus", "s__Veillonella_parvula"
KO9 = {"K00075": "murB", "K01512": "acyP", "K03060": "rpoZ", "K03101": "lspA", "K03116": "tatA",
       "K03687": "grpE", "K03799": "htpX", "K04042": "glmU", "K10716": "kch"}       # identifiers, not data
CORE4 = ["K00075", "K04042", "K01512", "K03060"]
# ---- ONE order of the nine KOs, read from the data: Boruta selection frequency over the 25 outer folds, DM vs CDM, KO input (descending);
# ties: Combined-input frequency (descending), then MaAsLin2 q of the CDM coefficient relative to DM (ascending), then KO number
KO_ORDER_FILES = [f"{A1}/T_A1v2_feature_stability.csv", f"{REV}/refs/T_P3_4_maaslin2_kosum.csv"]
def ko_order():
    """Unified order of the nine KOs. The first four must be the core four (asserted)."""
    stab = pd.read_csv(KO_ORDER_FILES[0]); m = pd.read_csv(KO_ORDER_FILES[1]); q = m[m.comparison == "CDM vs DM"].set_index("feature").qval.astype(float)
    f = {i: stab[(stab.g == "g4") & (stab.input == i)].set_index("feature").n_folds_confirmed.reindex(list(KO9)).fillna(0).astype(int) for i in ("ko", "combined")}
    assert q.index.is_unique and set(q.index) == set(KO9)
    o = sorted(KO9, key=lambda k: (-f["ko"][k], -f["combined"][k], q[k], k)); assert set(o[:4]) == set(CORE4), o
    return o

def bh_nan(p):
    """Benjamini-Hochberg q values; NaN P values (untestable pairs) are left out of the family and stay NaN."""
    p = np.asarray(p, float); ok = ~np.isnan(p); r = np.full(len(p), np.nan); x = p[ok]; n = len(x); o = np.argsort(x)
    q = np.minimum.accumulate((x[o] * n / np.arange(1, n + 1))[::-1])[::-1]; t = np.empty(n); t[o] = np.minimum(q, 1); r[ok] = t; return r
assert abs(bh_nan([0.01, 0.04, 0.03])[0] - 0.03) < 1e-12   # self-check

def load_base():
    lab = pd.read_csv(LABP, index_col=0)['Group']
    K1 = pd.read_csv(K1P, index_col=0).T.loc[lab.index]; K2 = pd.read_csv(K2P, index_col=0).loc[lab.index]; T3 = pd.read_csv(T3P, index_col=0).loc[lab.index]
    return lab, K1, K2, T3

B3 = f"{REV}/correlations/B3_correlations"                # previously computed tables of the same analysis (used as regression checks and for the lipid / HbA1c columns)
GROUPS = ["NC", "DM", "Cirrhosis", "CDM"]; NEXP = {"NC": 59, "DM": 46, "Cirrhosis": 30, "CDM": 14}
SP = [LC, VP]; SPB3 = {LC: "L_crispatus", VP: "V_parvula"}
CLIN_FILE = ["Alcohol", "AST", "ALT", "PLATELET", "INR", "BIL", "Creatine", "Cholesterol", "GGT", "BMI", "APRI", "PT", "MDF", "MELD"]   # column order of clinic_dataset.csv ('Creatine' = file spelling)
# order of the clinical variables (fixed in advance): coagulation / hepatic severity, liver chemistry, fibrosis index and platelets, then renal, lipid, anthropometric, alcohol
CLIN = ["INR", "PT", "MDF", "MELD", "BIL", "AST", "ALT", "GGT", "APRI", "PLATELET", "Creatine", "Cholesterol", "BMI", "Alcohol"]
LIP = ["TG", "HDL", "LDL", "HbA1c"]
KOS = ko_order()                                     # computed from the data (ko_order)
assert [KO9[k] for k in KOS] == ["murB", "acyP", "glmU", "rpoZ", "htpX", "tatA", "kch", "grpE", "lspA"], KOS
B3NAME = {**SPB3, **{k: f"{k}_{KO9[k]}" for k in KOS}, **{v: v for v in CLIN + LIP}}
TYPE = {**{s: "species" for s in SP}, **{k: "KO" for k in KOS}, **{v: "clinical" for v in CLIN}}
NODES = SP + KOS + CLIN                              # 25 nodes
PAIRS = [(LC, VP)] + [(s, k) for s in SP for k in KOS] + [(s, v) for s in SP for v in CLIN] + [(k, v) for k in KOS for v in CLIN]; assert len(PAIRS) == 173
PI = np.array([NODES.index(a) for a, b in PAIRS]); PJ = np.array([NODES.index(b) for a, b in PAIRS])
PTYPE = [f"{TYPE[a]}-{TYPE[b]}" for a, b in PAIRS]
UNITS = ["KO_SUM", "ORIGINAL"]
F_CLIN = F4 + "clinic_dataset.csv"; F_NODE = f"{B3}/T_B3_node_matrix.csv"; F_COR = f"{B3}/T_B3_correlations.csv"

def load_all():
    """labels, {unit: KO table}, species table, clinical table - all in label order BY SAMPLE ID; checked against the B-3 node matrix."""
    lab, K1, K2, T3 = load_base(); assert lab.value_counts().to_dict() == NEXP
    c = pd.read_csv(F_CLIN, index_col=0, header=None).T.set_index("Sample"); assert list(c.columns[1:]) == CLIN_FILE
    clin = c[CLIN_FILE].apply(pd.to_numeric, errors="raise").loc[lab.index]; assert clin.isna().sum().sum() == 0
    nm = pd.read_csv(F_NODE, index_col=0); assert list(nm.index) == list(lab.index) and (nm.Group == lab).all()
    assert np.array_equal(nm[CLIN_FILE].values.astype(float), clin.values) and np.array_equal(nm[SP].values, T3[SP].values)
    assert np.allclose(nm[[k + "_KOSUM" for k in KOS]].values, K2[KOS].values, rtol=1e-12, atol=0) and np.allclose(nm[[k + "_ORIG" for k in KOS]].values, K1[KOS].values, rtol=1e-12, atol=0)
    return lab, {"KO_SUM": K2[KOS], "ORIGINAL": K1[KOS]}, T3[SP], clin, nm[LIP]

def spearman_pairs(X, I=PI, J=PJ):
    """Vectorised Spearman for the column pairs (I, J) of the complete matrix X (n x p): average ranks once, Pearson of ranks,
    two-sided P by the t approximation (df = n - 2) = scipy.stats.spearmanr default = R cor.test(exact = FALSE). A constant column gives NaN (untestable)."""
    n = X.shape[0]; R = rankdata(X, axis=0); R = R - R.mean(0); s = np.sqrt((R ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.clip((R[:, I] * R[:, J]).sum(0) / (s[I] * s[J]), -1, 1)
        P = 2 * tdist.sf(np.abs(rho * np.sqrt((n - 2) / (1 - rho ** 2))), n - 2)
    P[np.isnan(rho)] = np.nan
    return rho, P

def group_matrix(g, unit, lab, K, T3, clin):
    idx = lab.index[lab == g]; return np.column_stack([T3.loc[idx, SP].values, K[unit].loc[idx, KOS].values, clin.loc[idx, CLIN].values]).astype(float)

def main_family(lab, K, T3, clin):
    """Re-derive the 173-pair family for the four groups and both units; must equal T_B3_correlations.csv (asserted)."""
    rows = []
    for unit in UNITS:
        for g in GROUPS:
            X = group_matrix(g, unit, lab, K, T3, clin); rho, P = spearman_pairs(X); q = bh_nan(P)
            rows += [dict(group=g, unit=unit, node1=B3NAME[a], node2=B3NAME[b], id1=a, id2=b, pair_type=PTYPE[i], n=len(X), rho=rho[i], P=P[i], q=q[i]) for i, (a, b) in enumerate(PAIRS)]
    D = pd.DataFrame(rows); S = pd.read_csv(F_COR)
    m = D.merge(S, on=["group", "unit", "node1", "node2"], suffixes=("", "_b3"), validate="one_to_one"); assert len(m) == len(D) == len(S) == 1384
    dev = {c: float((m[c] - m[c + "_b3"]).abs().max()) for c in ("rho", "P", "q", "n")}; assert max(dev.values()) < 1e-10, dev
    # spot check of the vectorised function against scipy.stats.spearmanr itself
    X = group_matrix("CDM", "KO_SUM", lab, K, T3, clin); r0 = [spearmanr(X[:, i], X[:, j]) for i, j in zip(PI[:40], PJ[:40])]; r1 = spearman_pairs(X)
    assert np.allclose([r.statistic for r in r0], r1[0][:40], atol=1e-12) and np.allclose([r.pvalue for r in r0], r1[1][:40], atol=1e-12)
    D["edge"] = np.where((D.rho.abs() > 0.6) & (D.q < 0.05), "solid", np.where(D.P < 0.05, "dashed", "none"))
    return D, dev
