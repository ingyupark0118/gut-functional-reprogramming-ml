#!/usr/bin/env python
"""A2-KOsum (pre-fixed spec 2026-09-30; every result is reported regardless of outcome).

(1) Dir-sFSS in 4 cohorts with ONE normalisation: relab = KO / sum(all community KO)
      CNGB, MetaCardis : final_external_ko_cpm.tsv read exactly as G1 (g1_verify.load_relab)
      QinN_2014, QinJ_2012 : 9-KO numerators (<study>_9ko_relab.tsv) / KO_sum (<study>_ko_sum.tsv)
    z = zscore(log10(relab + 1e-5), ddof=1) within cohort; Dir_sFSS = sum W_i z_i (W as G1).
    Direction FIXED: AUC = P(score_case > score_ctrl), i.e. higher = disease. DeLong CI.
    Qin cohorts also: logistic OR per 1 SD adjusted for age, sex, BMI (as a2_finalize.py).
(2) murB (K00075) and acyP (K01512) alone, same relab: direction, Wilcoxon P, fixed-direction AUC.
(3) QinN_2014 within cirrhosis: Spearman of Dir_sFSS with the available severity variables
    (no Child-Pugh / MELD in cMD): INR, bilirubin, albumin, creatinine; ascites yes/no (Mann-Whitney).
Outputs (new files): T_A2_kosum_dirsfss.csv, T_A2_kosum_singleKO.csv, T_A2_kosum_QinN_severity.csv,
                     A2_kosum_scores.csv, LOG_A2_kosum.txt
"""
import os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

REV = "."
OUT = f"{REV}/dir_sfss"
DFS = f"{REV}/input_dirFss"
META = f"{REV}/external_cohorts/cMD_raw_tables/cMD_metadata"
PSEUDO = 1e-5
KOS = ["K00075", "K01512", "K03060", "K03101", "K03116", "K03687", "K03799", "K04042", "K10716"]
W = pd.Series({"K00075": +1, "K01512": +1, "K03060": +1, "K03101": -1, "K03116": -1,
               "K03687": -1, "K03799": +1, "K04042": +1, "K10716": -1}).reindex(KOS)
outs = [f"{OUT}/{f}" for f in ("T_A2_kosum_dirsfss.csv", "T_A2_kosum_singleKO.csv",
                               "T_A2_kosum_QinN_severity.csv", "A2_kosum_scores.csv", "LOG_A2_kosum.txt")]
if any(os.path.exists(f) for f in outs):
    raise SystemExit("output exists, back up and remove first: " + ", ".join(f for f in outs if os.path.exists(f)))
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)


def norm_id(i):                                   # = g1_verify.norm_id
    for _ in range(3):
        for suf in ("_sub20M", "_merged", "_Abundance-RPKs"):
            if i.endswith(suf):
                i = i[: -len(suf)]
    return i


def load_relab(path):                             # = g1_verify.load_relab
    keep = {f"{k}:" for k in KOS}
    rows = {}
    with open(path) as fh:
        samples = [norm_id(c) for c in fh.readline().rstrip("\n").split("\t")[1:]]
        total = np.zeros(len(samples))
        for line in fh:
            name, _, rest = line.partition("\t")
            if "|" in name:
                continue
            v = np.fromstring(rest, sep="\t")
            if not name.startswith(("UNMAPPED", "UNGROUPED")):
                total += v
            tag = name.split()[0]
            if tag in keep:
                rows[tag[:-1]] = v
    return pd.DataFrame(rows, index=samples)[KOS].div(pd.Series(total, index=samples), axis=0)


def delong(y, s):                                  # fixed direction: higher score = case
    pos, neg = s[y == 1], s[y == 0]; m, n = len(pos), len(neg)
    v10 = np.array([((p > neg).sum() + 0.5 * (p == neg).sum()) / n for p in pos])
    v01 = np.array([((pos > q).sum() + 0.5 * (pos == q).sum()) / m for q in neg])
    auc = v10.mean(); se = np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    return auc, auc - 1.96 * se, auc + 1.96 * se


def dirsfss(relab):
    lg = np.log10(relab.values + PSEUDO)
    return (((lg - lg.mean(0)) / lg.std(0, ddof=1)) * W.values).sum(1)


# ---------------------------------------------------------------- cohorts
coh = {}
g1c = pd.read_csv(f"{REV}/work_e/cirrhotic_scores.csv").set_index("ID")
r = load_relab(f"{DFS}/cirrhosis_axis/260317/final_external_ko_cpm.tsv").loc[g1c.index]
coh["CNGB"] = dict(relab=r, y=(g1c["Group"] == "LC").astype(int).values, case="LC", ctrl="HC",
                   check=g1c["Dir_sFSS"].values, cov=None)
g1m = pd.read_csv(f"{REV}/work_e/metabolic_scores.csv").set_index("ID")
r = load_relab(f"{DFS}/metabolic_axis/260316/final_external_ko_cpm.tsv").loc[g1m.index]
coh["MetaCardis"] = dict(relab=r, y=(g1m["Group"] == "High-risk").astype(int).values,
                         case="High-risk", ctrl="Low-risk", check=g1m["Dir_sFSS"].values, cov=None)
for study, case in (("QinN_2014", "cirrhosis"), ("QinJ_2012", "T2D")):
    ko = pd.read_csv(f"{OUT}/{study}_9ko_relab.tsv", sep="\t").set_index("sample_id")
    ks = pd.read_csv(f"{OUT}/{study}_ko_sum.tsv", sep="\t").set_index("sample_id")
    md = pd.read_csv(f"{META}/{study}_metadata.tsv", sep="\t").set_index("sample_id")
    ids = [i for i in ko.index if i in md.index and md.loc[i, "study_condition"] in (case, "control")]
    cov = md.loc[ids, ["age", "gender", "BMI"]].copy()
    cov["male"] = (cov["gender"].astype(str).str.lower() == "male").astype(float)
    coh[study] = dict(relab=ko.loc[ids, KOS].div(ks.loc[ids, "KO_sum"], axis=0),
                      y=(md.loc[ids, "study_condition"] == case).astype(int).values,
                      case=case, ctrl="control", check=None,
                      cov=cov[["age", "male", "BMI"]].apply(pd.to_numeric, errors="coerce"),
                      kosum=ks.loc[ids, "KO_sum"], mapped=ko.loc[ids, "mapped_total"])

P("=" * 90); P("A2-KOsum | relab = KO / sum(community KO) | pseudocount 1e-5 | AUC direction fixed: higher = case")
P("W = " + " ".join(f"{k}:{w:+d}" for k, w in W.items())); P("=" * 90)

rows, single, sc = [], [], []
for name, c in coh.items():
    y, rl = c["y"], c["relab"]
    d = dirsfss(rl)
    if c["check"] is not None:
        P(f"{name}: recomputed vs G1 stored Dir_sFSS max|diff| = {np.abs(d - c['check']).max():.2e}")
    else:
        P(f"{name}: KO_sum / old denominator (mapped UniRef90): median "
          f"{(c['kosum'] / c['mapped']).median():.4f} range {(c['kosum'] / c['mapped']).min():.4f}-"
          f"{(c['kosum'] / c['mapped']).max():.4f}")
    a, b = d[y == 1], d[y == 0]
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    auc, lo, hi = delong(y, d)
    row = dict(cohort=name, ctrl=c["ctrl"], case=c["case"], n_ctrl=int((y == 0).sum()), n_case=int(y.sum()),
               median_ctrl=np.median(b), median_case=np.median(a),
               direction="UP" if np.median(a) > np.median(b) else "DOWN",
               P_wilcoxon=p, AUC_fixed=auc, CI_low=lo, CI_high=hi)
    if c["cov"] is not None:
        X = c["cov"].copy(); X.insert(0, "Dir_sFSS_z", (d - d.mean()) / d.std(ddof=1))
        ok = X.notna().all(1).values
        res = sm.Logit(y[ok], sm.add_constant(X[ok])).fit(disp=0)
        ci = res.conf_int().loc["Dir_sFSS_z"]
        row.update(adj_OR=np.exp(res.params["Dir_sFSS_z"]), adj_OR_low=np.exp(ci.iloc[0]),
                   adj_OR_high=np.exp(ci.iloc[1]), adj_P=res.pvalues["Dir_sFSS_z"], adj_n=int(ok.sum()))
    rows.append(row)
    P(f"  {name:11s} {c['ctrl']} {row['n_ctrl']} / {c['case']} {row['n_case']}  {row['direction']}  "
      f"P={p:.3g}  AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]"
      + (f"  adjOR/1SD={row['adj_OR']:.3f} [{row['adj_OR_low']:.3f}, {row['adj_OR_high']:.3f}] "
         f"p={row['adj_P']:.3g} (n={row['adj_n']})" if "adj_OR" in row else ""))
    for k, lab in (("K00075", "murB"), ("K01512", "acyP")):
        v = rl[k].values; va, vb = v[y == 1], v[y == 0]
        pk = stats.mannwhitneyu(va, vb, alternative="two-sided").pvalue
        ak, lk, hk = delong(y, v)
        single.append(dict(cohort=name, KO=k, gene=lab, median_ctrl=np.median(vb), median_case=np.median(va),
                           direction="UP" if ak > 0.5 else ("DOWN" if ak < 0.5 else "none"),
                           P_wilcoxon=pk, AUC_fixed=ak, CI_low=lk, CI_high=hk,
                           detection_pct=float((v > 0).mean() * 100)))
        P(f"      {lab} ({k}): {single[-1]['direction']}  P={pk:.3g}  AUC={ak:.3f} [{lk:.3f}, {hk:.3f}]")
    sc.append(pd.DataFrame({"cohort": name, "sample": rl.index, "y": y, "Dir_sFSS": d}))

# ---------------------------------------------------------------- (3) QinN severity
sev = pd.read_csv(f"{OUT}/QinN_2014_severity_metadata.csv").set_index("sample_id")
q = sc[[s.cohort.iloc[0] for s in sc].index("QinN_2014")].set_index("sample")
cir = [i for i in q.index if q.loc[i, "y"] == 1]
P(f"\nQinN_2014 within cirrhosis (n={len(cir)}); Child-Pugh / MELD not in cMD metadata")
sv = []
for var, lab in (("inr", "INR"), ("bilubirin", "bilirubin"), ("albumine", "albumin"), ("creatine", "creatinine")):
    x = pd.to_numeric(sev.loc[cir, var], errors="coerce"); ok = x.notna()
    rho, pr = stats.spearmanr(q.loc[cir, "Dir_sFSS"][ok], x[ok])
    sv.append(dict(variable=lab, test="Spearman", n=int(ok.sum()), rho=rho, P=pr)); P(f"  {lab:11s} n={ok.sum():3d} rho={rho:+.3f} P={pr:.3g}")
asc = sev.loc[cir, "ascites"].astype(bool)
s1, s0 = q.loc[cir, "Dir_sFSS"][asc], q.loc[cir, "Dir_sFSS"][~asc]
pa = stats.mannwhitneyu(s1, s0, alternative="two-sided").pvalue
sv.append(dict(variable="ascites (yes vs no)", test="Mann-Whitney", n=len(cir), rho=np.nan, P=pa,
               median_yes=np.median(s1), median_no=np.median(s0), n_yes=int(asc.sum())))
P(f"  ascites     yes {asc.sum()} median={np.median(s1):+.3f} / no {(~asc).sum()} median={np.median(s0):+.3f}  P={pa:.3g}")

pd.DataFrame(rows).to_csv(outs[0], index=False); pd.DataFrame(single).to_csv(outs[1], index=False)
pd.DataFrame(sv).to_csv(outs[2], index=False); pd.concat(sc).to_csv(outs[3], index=False)
open(outs[4], "w").write("\n".join(log) + "\n")
P("[written] " + ", ".join(os.path.basename(f) for f in outs))
