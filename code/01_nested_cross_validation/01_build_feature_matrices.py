#!/usr/bin/env python
"""A1 prep: rebuild the UNSELECTED species / KO matrices for all 149 CDM samples.

tables/dataset/ holds supervised-selected feature subsets (species 7/3/41/10,
KO 77/16/204/134 per comparison), so a leakage-free nested CV cannot start there.
This rebuilds the full feature space from the primary profiles.

Species : MetaPhlAn 4.1 per-sample taxonomic_profile.tsv, s__ and t__(SGB) rows.
          Feature naming reproduces the tables/dataset convention:
              ...|s__Roseburia_hominis            -> s__Roseburia_hominis
              ...|s__Roseburia_hominis|t__SGB4936 -> s__Roseburia_hominisSGB4936
          Value = relative_abundance% / 100.
KO      : merged_genefamilies_ko.tsv, community-level rows only
          (drop stratified '|', UNMAPPED, UNGROUPED); value = KO / sum(mapped KO).

Outputs species_matrix.csv, ko_matrix.csv, labels.csv, LOG_A1_build.txt.
"""
import os, re, glob, csv
import numpy as np, pandas as pd

ROOT = ".."
OUT = f"{ROOT}/project/nested_cv"
KO_MERGED = "data/humann/merged_genefamilies_ko.tsv"
KO_GENEDIR = "data/humann/total_gene"
MPA_ROOTS = ["data/metaphlan/set1",
             "data/metaphlan/set2",
             "data/metaphlan/set3",
             "data/metaphlan/set4"]
EXPECT = {"NC": 59, "DM": 46, "Cirrhosis": 30, "CDM": 14}

log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)


# ---------------------------------------------------------------- labels
rows = list(csv.reader(open(f"{ROOT}/output/composite_scores.csv")))
labels = pd.Series({r[0]: r[1] for r in rows[1:]}, name="Group")
labels.index.name = "SampleID"
P(f"labels: n={len(labels)}  {labels.value_counts().to_dict()}")
if labels.value_counts().to_dict() != EXPECT:
    raise SystemExit(f"S5: group sizes {labels.value_counts().to_dict()} != {EXPECT}")
if labels.index.duplicated().any():
    raise SystemExit("S5: duplicate SampleID in metadata")
SAMPLES = list(labels.index)

# ---------------------------------------------------------------- species
prof = {}
for root in MPA_ROOTS:
    for p in glob.glob(root + "/**/*taxonomic_profile*.tsv", recursive=True):
        m = re.match(r"([A-Za-z]+\d+)", os.path.basename(p))
        if m and m.group(1) not in prof:          # first root wins
            prof[m.group(1)] = p
missing = [s for s in SAMPLES if s not in prof]
if missing:
    raise SystemExit(f"S5: no MetaPhlAn profile for {len(missing)}: {missing[:10]}")
P(f"metaphlan profiles: {len(prof)} unique sample ids; all 149 target samples present")


def parse_profile(path):
    out, ver = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("#mpa_"):
                    ver = line[1:].strip()
                continue
            f = line.rstrip("\n").split("\t")
            if "s__" not in f[0]:
                continue
            parts = f[0].split("|")
            sp = next((x for x in parts if x.startswith("s__")), None)
            sgb = next((x for x in parts if x.startswith("t__")), None)
            out[sp + sgb[3:] if sgb else sp] = float(f[2]) / 100.0
    return out, ver


sp_rows, versions = {}, set()
for s in SAMPLES:
    d, v = parse_profile(prof[s])
    sp_rows[s] = d
    if v:
        versions.add(v)
species = pd.DataFrame.from_dict(sp_rows, orient="index").fillna(0.0).reindex(SAMPLES)
P(f"MetaPhlAn versions: {sorted(versions)}")
if len(versions) > 1:
    P("  *** WARNING: mixed MetaPhlAn database versions across samples")
P(f"species matrix: {species.shape[0]} x {species.shape[1]} "
  f"(non-SGB s__ features: {sum(1 for c in species.columns if 'SGB' not in c)})")
P(f"  row sums: min={species.sum(1).min():.4f} max={species.sum(1).max():.4f} "
  "(~2.0 expected: s__ and its t__/SGB child are both retained, matching the "
  "tables/dataset feature-naming convention)")

# ---------------------------------------------------------------- KO
# Column -> sample id. The merged header carries each file's internal sample name.
# Where the name prefix and the sequencing tag of a column disagree, the sequencing
# tag (unique across the per-run files) takes precedence over the name prefix.
alias, by_stag = {}, {}
for p in glob.glob(f"{KO_GENEDIR}/*_genefamilies.tsv"):
    b = os.path.basename(p)[: -len("_genefamilies.tsv")]
    m = re.match(r"([A-Za-z]+\d+)", b)
    if not m:
        continue
    sid = m.group(1)
    alias[b] = sid                                   # full file stem
    alias[b[len(sid) + 1:]] = sid                    # file stem without the sample prefix
    alias.setdefault(sid, sid)
    st = re.search(r"_(S\d+)$", b)
    if st:
        by_stag.setdefault(st.group(1), []).append(sid)
by_stag = {k: v[0] for k, v in by_stag.items() if len(set(v)) == 1}

with open(KO_MERGED) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    cols = [c.replace("_Abundance-RPKs", "") for c in header[1:]]
    mapped, relabelled = [], []
    for c in cols:
        st = re.search(r"_(S\d+)$", c)
        sid = by_stag.get(st.group(1)) if st else None
        if sid is None:
            sid = alias.get(c) or (re.match(r"([A-Za-z]+\d+)", c).group(1)
                                   if re.match(r"[A-Za-z]+\d+", c) else None)
        elif not c.startswith(sid + "_"):
            relabelled.append((c, sid))
        mapped.append(sid)
    for c, sid in relabelled:
        P(f"  header relabelled via unique S-tag: column '{c}' -> {sid}")
    bad = [c for c, m in zip(cols, mapped) if m is None]
    if bad:
        P(f"  unresolved KO columns (dropped): {bad}")
    ko_rows, total, n_comm = {}, np.zeros(len(cols)), 0
    for line in fh:
        name, _, rest = line.partition("\t")
        if "|" in name:
            continue
        n_comm += 1
        if name.startswith(("UNMAPPED", "UNGROUPED")):
            continue
        v = np.fromstring(rest, sep="\t")
        total += v
        ko_rows[name.split(":")[0]] = v

ko = pd.DataFrame(ko_rows, index=cols)
ko["_sid"] = mapped
ko = ko[ko["_sid"].notna()].set_index("_sid")
tot = pd.Series(total, index=mapped)
dups = ko.index[ko.index.duplicated()].unique().tolist()
if dups:
    P(f"  duplicate sample ids in KO table (keeping first): {dups}")
ko = ko[~ko.index.duplicated(keep="first")]
tot = tot[~tot.index.duplicated(keep="first")]
missing = [s for s in SAMPLES if s not in ko.index]
if missing:
    raise SystemExit(f"S5: KO profile missing for {missing}")
ko = ko.loc[SAMPLES].div(tot.loc[SAMPLES], axis=0)
ko.index.name = "SampleID"
P(f"KO: {n_comm} community rows -> {ko.shape[1]} KO features x {ko.shape[0]} samples")
P(f"  row sums: min={ko.sum(1).min():.6f} max={ko.sum(1).max():.6f}")

# ---------------------------------------------------------------- write
species.index.name = "SampleID"
species.to_csv(f"{OUT}/species_matrix.csv")
ko.to_csv(f"{OUT}/ko_matrix.csv")
labels.to_frame().to_csv(f"{OUT}/labels.csv")
P(f"\nwritten: species_matrix.csv {species.shape}, ko_matrix.csv {ko.shape}, labels.csv")

# ------------------- coverage check vs the selected sets -------------------
P("\nfeature coverage vs tables/dataset selected sets:")
for kind, mat, pat in (("species", species, "Species"), ("ko", ko, "KO")):
    for g in "g1 g2 g3 g4".split():
        feats = next(csv.reader(open(f"{ROOT}/tables/dataset/{kind}/{pat}_dis_{g}.csv")))
        miss = [x for x in feats if x not in mat.columns]
        P(f"  {pat:8s} {g}: {len(feats)} selected, {len(miss)} absent from rebuilt matrix"
          + (f"  -> {miss[:5]}" if miss else ""))

open(f"{OUT}/LOG_A1_build.txt", "w").write("\n".join(log) + "\n")
