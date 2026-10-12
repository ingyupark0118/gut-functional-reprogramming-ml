#!/usr/bin/env python
"""A1-v2 prep: split the species matrix by taxonomic level.

build_matrices.py collapsed both levels into one namespace:
    ...|s__X            -> "s__X"
    ...|s__X|t__SGB123  -> "s__XSGB123"
so a parent species and its SGB child sit side by side as separate features.

A string filter on "SGB" cannot undo this: 2,064 of the 2,773 feature names contain
"SGB", and most are genuine MetaPhlAn 4 unnamed species at s__ level
(s__GGB1148_SGB1475, s__Faecalibacterium_SGB15346, ...). The level has to come from
the clade string, not the name. This rebuild records it explicitly.

Outputs
  species_s_matrix.csv        s__ rows only            <- A1-v2 primary input
  species_sgb_matrix.csv      t__ rows only            <- sensitivity input
  species_feature_levels.csv  feature, level, source_clade
"""
import os, re, glob
import pandas as pd

A1 = "./nested_cv"
MPA_ROOTS = ["data/metaphlan/set1",
             "data/metaphlan/set2",
             "data/metaphlan/set3",
             "data/metaphlan/set4"]

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)


labels = pd.read_csv(f"{A1}/labels.csv", index_col=0)["Group"]
SAMPLES = list(labels.index)
P(f"labels: n={len(labels)}  {labels.value_counts().to_dict()}")

prof = {}
for root in MPA_ROOTS:
    for p in glob.glob(root + "/**/*taxonomic_profile*.tsv", recursive=True):
        m = re.match(r"([A-Za-z]+\d+)", os.path.basename(p))
        if m and m.group(1) not in prof:
            prof[m.group(1)] = p
missing = [s for s in SAMPLES if s not in prof]
if missing:
    raise SystemExit(f"S1: no profile for {missing[:10]}")
P(f"profiles located for all {len(SAMPLES)} samples")

sp_rows, sgb_rows, meta = {}, {}, {}
for s in SAMPLES:
    sd, gd = {}, {}
    with open(prof[s]) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            clade = f[0]
            if "s__" not in clade:
                continue
            parts = clade.split("|")
            name = next(x for x in parts if x.startswith("s__"))
            sgb = next((x for x in parts if x.startswith("t__")), None)
            val = float(f[2]) / 100.0
            if sgb is None:
                sd[name] = val
                meta.setdefault(name, ("species", clade))
            else:
                key = name + sgb[3:]
                gd[key] = val
                meta.setdefault(key, ("sgb", clade))
    sp_rows[s], sgb_rows[s] = sd, gd

spm = pd.DataFrame.from_dict(sp_rows, orient="index").fillna(0.0).reindex(SAMPLES)
sgbm = pd.DataFrame.from_dict(sgb_rows, orient="index").fillna(0.0).reindex(SAMPLES)
spm.index.name = sgbm.index.name = "SampleID"

P(f"\ns__ level : {spm.shape[0]} x {spm.shape[1]}   row sums "
  f"{spm.sum(1).min():.4f}-{spm.sum(1).max():.4f}")
P(f"t__ level : {sgbm.shape[0]} x {sgbm.shape[1]}   row sums "
  f"{sgbm.sum(1).min():.4f}-{sgbm.sum(1).max():.4f}")
P(f"combined namespace (v1 species_matrix.csv): {spm.shape[1] + sgbm.shape[1]}")

nm_sgb = sum(1 for c in spm.columns if "SGB" in c)
P(f"\ns__ features whose NAME contains 'SGB': {nm_sgb} / {spm.shape[1]}")
P("  -> MetaPhlAn 4 unnamed species, not SGB-level rows.")
P("     A name-based filter would have wrongly dropped them.")

spm.to_csv(f"{A1}/species_s_matrix.csv")
sgbm.to_csv(f"{A1}/species_sgb_matrix.csv")
pd.DataFrame([dict(feature=k, level=v[0], source_clade=v[1]) for k, v in meta.items()]
             ).to_csv(f"{A1}/species_feature_levels.csv", index=False)
P("\n[written] species_s_matrix.csv, species_sgb_matrix.csv, species_feature_levels.csv")

old = pd.read_csv(f"{A1}/species_matrix.csv", index_col=0, nrows=1)
P(f"\nsanity: v1 matrix {old.shape[1]} cols vs rebuilt "
  f"{spm.shape[1]}+{sgbm.shape[1]}={spm.shape[1]+sgbm.shape[1]}  "
  f"match={old.shape[1] == spm.shape[1] + sgbm.shape[1]}")
open(f"{A1}/LOG_A1v2_build.txt", "w").write("\n".join(log) + "\n")
