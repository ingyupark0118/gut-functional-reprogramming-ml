#!/usr/bin/env python
"""C-1 input: raw (uncorrected) species counts = round(relative abundance x mapped reads),
exactly as project/nested_cv/a1v3_conqur.R:46-53 built the counts it handed to ConQuR (this is the table BEFORE correction).
Also builds the phylum map and the fixed subsample draws. Refuses to overwrite."""
import pandas as pd, numpy as np, glob, hashlib, os, sys
ROOT = ".."; A1 = f"{ROOT}/project/nested_cv"; OUT = f"{ROOT}/project/work_a/C1_sparcc/inputs"
SRC = f"{ROOT}/project/input_sparcc"
SEED = 20261002; NSUB = 14; NDRAW = 100
outs = [f"{OUT}/{f}" for f in ["species_s_counts_raw.csv", "mapped_reads.csv", "taxon_info.csv", "subsample_draws.csv"]]
if any(os.path.exists(o) for o in outs): sys.exit("output exists - not overwriting")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
sp = pd.read_csv(f"{A1}/species_s_matrix.csv", index_col=0); lab = pd.read_csv(f"{A1}/labels.csv", index_col=0)["Group"]
assert list(sp.index) == list(lab.index) and sp.shape == (149, 1338)
print("sha256 species_s_matrix.csv", sha(f"{A1}/species_s_matrix.csv")); print("sha256 labels.csv", sha(f"{A1}/labels.csv"))
print("groups", lab.value_counts().to_dict(), "| row sums", sp.sum(1).min(), sp.sum(1).max())
mapped = {}
for s in sp.index:
    f = glob.glob(f"{ROOT}/profiles/raw_metaphlan/{s}_taxonomic_profile*.tsv"); assert len(f) == 1, (s, f)
    h = [l for l in open(f[0]).readlines()[:10] if l.startswith("#estimated_reads_mapped_to_known_clades")]
    mapped[s] = float(h[0].split(":")[-1])
mapped = pd.Series(mapped, name="estimated_reads_mapped_to_known_clades")
cnt = np.round(sp.values * mapped.values[:, None])          # numpy and R both round half to even
cnt = pd.DataFrame(cnt.astype(np.int64), index=sp.index, columns=sp.columns)
print(f"mapped reads min={mapped.min():.0f} median={mapped.median():.0f} max={mapped.max():.0f}")
print("nonzero relab rounded to 0 count:", int(((sp.values > 0) & (cnt.values == 0)).sum()), "of", int((sp.values > 0).sum()))
print("count row sums: min", cnt.sum(1).min(), "max", cnt.sum(1).max())
# phylum of every species from the MetaPhlAn clade string
lv = pd.read_csv(f"{A1}/species_feature_levels.csv").set_index("feature")
phy = [next(x for x in lv.loc[c, "source_clade"].split("|") if x.startswith("p__"))[3:] for c in sp.columns]
T1 = pd.read_csv(f"{SRC}/dataset/total_table.tsv", sep="\t", index_col=0)
tax = pd.read_csv(f"{SRC}/taxomony.csv"); old = {t.split(";")[6]: t.split(";")[1] for t in tax.Taxonomy}
info = pd.DataFrame({"taxon": sp.columns, "phylum_metaphlan": phy,
                     "in_original_346": sp.columns.isin(T1.index),
                     "phylum_original_taxonomy_csv": [old.get(c.replace("s__", ""), "") for c in sp.columns]})
for g in ["NC", "DM", "Cirrhosis", "CDM"]:
    info[f"n_present_{g}"] = (cnt[lab == g] > 0).sum(0).values
print("original 346 features found at s__ level:", int(info.in_original_346.sum()), "(the other", 346 - int(info.in_original_346.sum()), "are t__/SGB-level rows)")
print("phylum (MetaPhlAn) counts:", info.phylum_metaphlan.value_counts().to_dict())
o = info[info.in_original_346]; print("phylum cross-tab original taxonomy.csv vs MetaPhlAn (274 shared):\n", pd.crosstab(o.phylum_original_taxonomy_csv, o.phylum_metaphlan))
# membership check against the original per-group input files
for g, f in [("NC", "ncnon"), ("DM", "dmnon"), ("Cirrhosis", "nc_cirr"), ("CDM", "dmcirr")]:
    oc = pd.read_csv(f"{SRC}/dataset/{f}.csv", index_col=0, nrows=1).columns
    print(g, "same sample set as original", f, ":", set(oc) == set(lab.index[lab == g]), "| taxa present (>=1 sample):", int((cnt[lab == g].sum(0) > 0).sum()))
rng = np.random.default_rng(SEED); rows = []
for g in ["NC", "DM", "Cirrhosis"]:
    ids = np.array(lab.index[lab == g])
    for k in range(NDRAW):
        rows.append(dict(group=g, draw=k, **{f"s{m+1:02d}": s for m, s in enumerate(sorted(rng.choice(ids, NSUB, replace=False)))}))
dr = pd.DataFrame(rows); assert not dr.drop(columns=["group", "draw"]).apply(tuple, axis=1).groupby(dr.group).apply(lambda x: x.duplicated().any()).any()
cnt.to_csv(outs[0]); mapped.rename_axis("SampleID").to_csv(outs[1]); info.to_csv(outs[2], index=False); dr.to_csv(outs[3], index=False)
print(f"subsample draws: seed {SEED}, default_rng, order NC->DM->Cirrhosis, {NDRAW} x n={NSUB} without replacement, no duplicate draws")
for o_ in outs: print("sha256", os.path.basename(o_), sha(o_))
