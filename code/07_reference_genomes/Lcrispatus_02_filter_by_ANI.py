#!/usr/bin/env python
"""ANI filter + habitat classification (step before CheckM).

Input
  lc_ani_raw.tsv                  skani 0.3.2 dist, query=lc_fna/*.fna, ref=lc_types/*.fna
  T_P3_lc_all_metadata.csv        NCBI datasets metadata (2,065 records)
  gtdb_r226_bac120_metadata.tsv.gz
Output
  T_P3_50_candidates.csv, T_P3_50_flow.csv, T_P3_50_gtdb_mismatch.csv,
  T_P3_50_setB_ambiguous.csv, acc_ani_pass.txt, logs/LOG_P3_50_filter.txt
Rules (fixed in advance)
  ANI  : ANI >= 95 to a type strain AND align fraction(query) >= 60
  Set B: host=Homo sapiens and a gut keyword in isolation_source/body_site;
         excluded if a vagina / oral / urinary / breast-milk keyword is also present; ambiguous records are excluded (no manual decision)
"""
import gzip
import os
import sys

import pandas as pd

P3 = "./refs"
TYPE_STRAIN = {"GCF_002088015.1": "ATCC33820", "GCF_018987235.1": "ATCC33820",
               "GCF_001311685.1": "JCM1185", "GCF_001434005.1": "DSM20584",
               "GCF_008694755.1": "NCK2488"}
GUT = ["feces", "faeces", "fecal", "faecal", "stool", "gut", "intestin", "colon",
       "rectal", "ileum", "caecum", "cecum", "duodenum", "jejunum"]
NONGUT = ["vagin", "cervic", "cervix", "oral", "saliva", "tongue", "dental", "plaque",
          "urine", "urinary", "urethr", "milk", "breast", "nasal", "skin", "eye",
          "sputum", "lung", "blood", "wound"]
outs = [f"{P3}/T_P3_50_candidates.csv", f"{P3}/T_P3_50_flow.csv",
        f"{P3}/T_P3_50_gtdb_mismatch.csv", f"{P3}/T_P3_50_setB_ambiguous.csv",
        f"{P3}/acc_ani_pass.txt", f"{P3}/logs/LOG_P3_50_filter.txt"]
if any(os.path.exists(f) for f in outs):
    sys.exit("output exists: " + ", ".join(f for f in outs if os.path.exists(f)))
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


P("=" * 80)
P("ANI filter + habitat classification | skani 0.3.2 | rules fixed in advance")
P("type strain assemblies: " + ", ".join(f"{k}({v})" for k, v in TYPE_STRAIN.items()))
P("=" * 80)

a = pd.read_csv(f"{P3}/lc_ani_raw.tsv", sep="\t")
a["q"] = a.Query_file.map(lambda f: os.path.basename(f)[:-4])
a["r"] = a.Ref_file.map(lambda f: os.path.basename(f)[:-4])
best = a.sort_values("ANI", ascending=False).drop_duplicates("q", keep="first")
best = best.set_index("q")[["r", "ANI", "Align_fraction_query"]]
best.columns = ["ani_best_ref", "ani_best", "af_query_best"]

m = pd.read_csv(f"{P3}/T_P3_lc_all_metadata.csv").set_index("accession")
d = best.join(m, how="left").reset_index().rename(columns={"index": "accession", "q": "accession"})
d["ani_best_strain"] = d.ani_best_ref.map(TYPE_STRAIN)

# ---- GTDB
gt = {}
with gzip.open(f"{P3}/gtdb_r226_bac120_metadata.tsv.gz", "rt") as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    ia, it = hdr.index("accession"), hdr.index("gtdb_taxonomy")
    ig = hdr.index("ncbi_genbank_assembly_accession")
    for line in fh:
        f = line.rstrip("\n").split("\t")
        sp = f[it].split(";s__")[-1]
        acc = f[ia][3:] if f[ia][:3] in ("RS_", "GB_") else f[ia]
        gt[acc] = sp
        if f[ig] and f[ig] != "none":
            gt.setdefault(f[ig], sp)


def gl(acc):
    for v in (acc, acc.replace("GCF_", "GCA_"), acc.replace("GCA_", "GCF_")):
        if v in gt:
            return gt[v]
    return None


d["gtdb_species"] = d.accession.map(gl)
d["gtdb_mismatch"] = d.gtdb_species.notna() & (d.gtdb_species != "Lactobacillus crispatus")

# ---- ANI filter
d["pass_ani"] = (d.ani_best >= 95) & (d.af_query_best >= 60)

# ---- habitat
txt = (d.isolation_source.fillna("") + " | " + d.body_site.fillna("")).str.lower()
host = d.host.fillna("").str.lower()
d["hit_gut"] = txt.apply(lambda s: any(k in s for k in GUT))
d["hit_nongut"] = txt.apply(lambda s: [k for k in NONGUT if k in s])
d["host_human"] = host.str.contains("homo sapiens")
d["host_animal"] = host.str.contains("gallus|turkey|meleagris|chicken|sus scrofa|equus|bos |broiler|mouse|mus musculus|pig|swine") | txt.str.contains("chicken|turkey|broiler|poultry")


def habitat(r):
    if r.host_animal:
        return "poultry/animal"
    if r.hit_gut and not r.hit_nongut:
        return "gut/feces"
    if r.hit_nongut:
        n = r.hit_nongut[0]
        return {"vagin": "vagina/cervix", "cervic": "vagina/cervix", "cervix": "vagina/cervix", "oral": "oral", "saliva": "oral",
                "tongue": "oral", "dental": "oral", "plaque": "oral", "urine": "urinary",
                "urinary": "urinary", "urethr": "urinary", "milk": "breast milk", "breast": "breast milk"}.get(n, "other")
    if str(r.isolation_source).strip().lower() in ("", "nan", "none", "not applicable",
                                                   "not collected", "missing", "unknown",
                                                   "not provided", "na"):
        return "not stated"
    return "food/environment/other"


d["habitat"] = d.apply(habitat, axis=1)


def setb(r):
    if not r.pass_ani:
        return False, "ANI not passed"
    if not r.host_human:
        return False, "host not Homo sapiens / not stated"
    if r.hit_nongut:
        return False, f"non-gut keyword present({','.join(r.hit_nongut)})"
    if not r.hit_gut:
        return False, "no gut keyword"
    return True, "OK"


sb = d.apply(setb, axis=1, result_type="expand")
d["setB_eligible"], d["setB_reason"] = sb[0], sb[1]

# ---- flow table
flow = [("S1 NCBI search (GCA+GCF records)", 2065),
        ("unique genomes after removing GCA/GCF pairs", 1405),
        ("atypical excluded", 1187),
        ("FASTA obtained", len(d)),
        ("passed ANI >=95 & AF >=60", int(d.pass_ani.sum())),
        ("  (of these) set B eligible = human gut", int(d.setB_eligible.sum())),
        ("CheckM / de-duplication", None)]
fl = pd.DataFrame(flow, columns=["step", "n_genomes"])
P("\n[flow]")
for s, n in flow:
    P(f"  {s:38s} {n if n is not None else '(not run)'}")

P(f"\ntype strain with the highest ANI: {d.ani_best_strain.value_counts().to_dict()}")
P(f"ANI summary: min {d.ani_best.min():.2f} / median {d.ani_best.median():.2f} / max {d.ani_best.max():.2f}")
P(f"ANI<95 : {int((d.ani_best < 95).sum())}   AF<60 : {int((d.af_query_best < 60).sum())}   failed either: {int((~d.pass_ani).sum())}")
P(f"\n[set A] passed ANI: {int(d.pass_ani.sum())} genomes; habitat distribution:")
for k, v in d[d.pass_ani].habitat.value_counts().items():
    P(f"   {k:16s} {v}")
P(f"\n[set B] human gut: {int(d.setB_eligible.sum())} genomes")
P("set B exclusion reasons:")
for k, v in d[d.pass_ani & ~d.setB_eligible].setB_reason.value_counts().items():
    P(f"   {k:40s} {v}")
P(f"\nwith GTDB assignment {int(d.gtdb_species.notna().sum())} / mismatch {int(d.gtdb_mismatch.sum())}")
if d.gtdb_mismatch.any():
    P(f"   mismatching species: {d[d.gtdb_mismatch].gtdb_species.value_counts().to_dict()}")

cur = ["GCF_000497065.1", "GCF_013456995.1", "GCF_015669875.1", "GCF_020224235.1",
       "GCF_038001325.1", "GCF_038001345.1", "GCF_038001405.1", "GCF_048541565.1"]
c = d[d.accession.isin(cur)]
P(f"\n[the 8 references used before] passed ANI {int(c.pass_ani.sum())}/8, set B eligible {int(c.setB_eligible.sum())}/8")
for _, x in c.iterrows():
    P(f"   {x.accession}  ANI={x.ani_best:.2f}({x.ani_best_strain}) AF={x.af_query_best:.1f} "
      f"habitat={x.habitat} setB={x.setB_eligible} [{x.setB_reason}]")

d.to_csv(outs[0], index=False)
fl.to_csv(outs[1], index=False)
d[d.gtdb_mismatch].to_csv(outs[2], index=False)
d[d.pass_ani & ~d.setB_eligible][["accession", "host", "isolation_source", "body_site",
                                  "habitat", "setB_reason"]].to_csv(outs[3], index=False)
d[d.pass_ani][["accession"]].to_csv(outs[4], index=False, header=False)
open(outs[5], "w").write("\n".join(log) + "\n")
P("\n[written] " + ", ".join(os.path.basename(f) for f in outs))
