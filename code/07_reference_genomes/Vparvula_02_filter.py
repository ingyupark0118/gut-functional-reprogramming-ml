#!/usr/bin/env python
"""V1 steps 2-3: ANI filter vs GTDB parvula_A representative + habitat (keyword rules copied from p3_50_filter.py).
Pre-fixed: ANI>=95 AND query AF>=60; human gut = host Homo sapiens AND gut keyword AND no non-gut keyword."""
import os, sys
import pandas as pd
O = "./refs/vp_refs"
outs = [f"{O}/T_V1_candidates.csv", f"{O}/T_V1_humangut_excluded.csv", f"{O}/acc_humangut.txt",
        f"{O}/T_V1_cdm_ani_to_rep.csv"]
if any(os.path.exists(f) for f in outs):
    sys.exit("output exists: " + ", ".join(f for f in outs if os.path.exists(f)))
GUT = ["feces", "faeces", "fecal", "faecal", "stool", "gut", "intestin", "colon",
       "rectal", "ileum", "caecum", "cecum", "duodenum", "jejunum"]
NONGUT = ["vagin", "cervic", "cervix", "oral", "saliva", "tongue", "dental", "plaque",
          "urine", "urinary", "urethr", "milk", "breast", "nasal", "skin", "eye",
          "sputum", "lung", "blood", "wound"]
d = pd.read_csv(f"{O}/T_V1_candidates_meta.csv")
rep = d[d.gtdb_representative == "t"].use_acc.iloc[0]          # from GTDB metadata, not hard-coded
a = pd.read_csv(f"{O}/ani_all_vs_rep_cdm.tsv", sep="\t")
a["q"] = a.Query_file.map(lambda f: os.path.basename(f)[:-4])
a["r"] = a.Ref_file.map(lambda f: os.path.basename(f)[:-4])
ar = a[a.r == rep].set_index("q")
d["fasta_ncbi"] = d.use_acc.map(lambda x: os.path.exists(f"{O}/fna/{x}.fna"))
d["ani_rep"] = d.use_acc.map(ar.ANI); d["af_query_rep"] = d.use_acc.map(ar.Align_fraction_query)
d["af_ref_rep"] = d.use_acc.map(ar.Align_fraction_ref)
d["pass_ani"] = d.fasta_ncbi & (d.ani_rep >= 95) & (d.af_query_rep >= 60)

txt = (d.isolation_source.fillna("") + " | " + d.body_site.fillna("")).str.lower()
host = d.host.fillna("").str.lower()
d["hit_gut"] = txt.apply(lambda s: any(k in s for k in GUT))
d["hit_nongut"] = txt.apply(lambda s: ",".join(k for k in NONGUT if k in s))
d["host_human"] = host.str.contains("homo sapiens")
d["host_missing"] = d.host.isna()
d["host_animal"] = host.str.contains("gallus|turkey|meleagris|chicken|sus scrofa|equus|bos |broiler|mouse|mus musculus|pig|swine") | txt.str.contains("chicken|turkey|broiler|poultry")
def habitat(r):
    if r.host_animal: return "poultry/animal"
    if r.hit_gut and not r.hit_nongut: return "gut/feces"
    if r.hit_nongut:
        n = r.hit_nongut.split(",")[0]
        return {"vagin": "vagina/cervix", "cervic": "vagina/cervix", "cervix": "vagina/cervix", "oral": "oral", "saliva": "oral",
                "tongue": "oral", "dental": "oral", "plaque": "oral", "urine": "urinary",
                "urinary": "urinary", "urethr": "urinary", "milk": "breast milk", "breast": "breast milk"}.get(n, "other")
    if str(r.isolation_source).strip().lower() in ("", "nan", "none", "not applicable", "not collected",
                                                   "missing", "unknown", "not provided", "na"):
        return "not stated"
    return "food/environment/other"
d["habitat"] = d.apply(habitat, axis=1)
def hg(r):
    if not r.fasta_ncbi: return False, "FASTA not available from NCBI (suppressed)"
    if not r.pass_ani: return False, "ANI not passed"
    if not r.host_human: return False, "host not Homo sapiens / not stated"
    if r.hit_nongut: return False, f"non-gut keyword present({r.hit_nongut})"
    if not r.hit_gut: return False, "no gut keyword"
    return True, "OK"
x = d.apply(hg, axis=1, result_type="expand")
d["humangut_eligible"], d["humangut_reason"] = x[0], x[1]
# information only, NOT the pre-fixed rule: host field missing but gut keyword present, no non-gut keyword
d["info_a_hostmissing_gut"] = d.pass_ani & d.host_missing & d.hit_gut & (d.hit_nongut == "")
d.to_csv(outs[0], index=False)
d[~d.humangut_eligible][["use_acc", "strain", "host", "isolation_source", "body_site", "habitat", "ani_rep",
                         "af_query_rep", "humangut_reason", "info_a_hostmissing_gut"]].to_csv(outs[1], index=False)
d[d.humangut_eligible][["use_acc"]].to_csv(outs[2], index=False, header=False)
cdm = ar[ar.index.str.startswith("CDM")][["ANI", "Align_fraction_query", "Align_fraction_ref"]]
cdm.to_csv(outs[3])
print(f"representative (GTDB field): {rep}")
print(f"GTDB parvula_A total {len(d)} | NCBI FASTA {int(d.fasta_ncbi.sum())} | ANI pass {int(d.pass_ani.sum())} | human gut {int(d.humangut_eligible.sum())}")
print(f"ANI to rep: min {d.ani_rep.min():.2f} median {d.ani_rep.median():.2f} | AFq min {d.af_query_rep.min():.1f}")
print("ANI failures:", d[d.fasta_ncbi & ~d.pass_ani][["use_acc", "ani_rep", "af_query_rep"]].to_dict("records"))
print("CDM MAG vs rep:\n", cdm.to_string())
print("habitat of ANI-pass:", d[d.pass_ani].habitat.value_counts().to_dict())
print("human-gut exclusion reasons:", d[~d.humangut_eligible].humangut_reason.value_counts().to_dict())
print("info (a) host-missing + gut keyword:", int(d.info_a_hostmissing_gut.sum()))
