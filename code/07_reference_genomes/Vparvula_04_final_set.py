#!/usr/bin/env python
"""V1 steps 4-7: CheckM QC (>=90 / <5), dedup at ANI>=99.9 (single linkage; rep = max comp-5*cont, tie fewest contigs),
prior-15 reference table, CDM MAG nearest genomes, flow table. Thresholds pre-fixed."""
import os, sys
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
O = "./refs/vp_refs"
names = ["T_V1_candidates_all.csv", "T_V1_humangut_final.csv", "T_V1_dedup_clusters.csv", "T_V1_prior15_refs.csv",
         "T_V1_cdm_vs_final_ani.csv", "T_V1_cdm_nearest4.csv", "T_V1_flow.csv", "T_V1_info_a_hostmissing.csv",
         "T_V1_habitat_anipass.csv"]
outs = [f"{O}/{n}" for n in names]
if any(os.path.exists(f) for f in outs):
    sys.exit("output exists: " + ", ".join(f for f in outs if os.path.exists(f)))
CDM = ["CDM-1_bin.1", "CDM-4_bin.2", "CDM-3_bin.12"]
bn = lambda f: os.path.basename(f)[:-4]
d = pd.read_csv(f"{O}/T_V1_candidates.csv")
rep = d[d.gtdb_representative == "t"].use_acc.iloc[0]
ck = pd.read_csv(f"{O}/checkm_humangut.tsv", sep="\t").set_index("Bin Id")
d["checkm124_comp"] = d.use_acc.map(ck["Completeness"]); d["checkm124_cont"] = d.use_acc.map(ck["Contamination"])
d["checkm124_lineage"] = d.use_acc.map(ck["Marker lineage"])
d["pass_qc"] = d.humangut_eligible & (d.checkm124_comp >= 90) & (d.checkm124_cont < 5)
assert d[d.humangut_eligible].checkm124_comp.notna().all()

# all-vs-all matrix
L = open(f"{O}/ani_triangle_all.tsv").read().rstrip("\n").split("\n")[1:]
nm = [bn(l.split("\t")[0]) for l in L]
M = pd.DataFrame([[float(x) for x in l.split("\t")[1:]] for l in L], index=nm, columns=nm)
def cluster(accs):
    A = (M.loc[accs, accs].values >= 99.9).astype(int)
    return dict(zip(accs, connected_components(A, directed=False)[1]))
q = d[d.pass_qc].copy()
q["qs"] = q.checkm124_comp - 5 * q.checkm124_cont
cl = cluster(list(q.use_acc)); q["cluster_id"] = q.use_acc.map(cl)
q = q.sort_values(["cluster_id", "qs", "n_contigs"], ascending=[True, False, True])
q["cluster_rep"] = q.groupby("cluster_id").use_acc.transform("first")
q["is_cluster_rep"] = q.use_acc == q.cluster_rep
q["cluster_size"] = q.groupby("cluster_id").use_acc.transform("size")
q["max_ani_to_other_member"] = [M.loc[a, [b for b in q.use_acc if b != a]].max() for a in q.use_acc]
d["final_set"] = d.use_acc.isin(q[q.is_cluster_rep].use_acc)
d = d.merge(q[["use_acc", "qs", "cluster_id", "cluster_rep", "cluster_size"]], on="use_acc", how="left")
final = list(d[d.final_set].use_acc)

# CDM vs public (MAG = query)
c = pd.read_csv(f"{O}/ani_cdm_vs_public.tsv", sep="\t")
c["mag"] = c.Query_file.map(bn); c["acc"] = c.Ref_file.map(bn)
c = c[["mag", "acc", "ANI", "Align_fraction_query", "Align_fraction_ref"]].rename(
    columns={"Align_fraction_query": "AF_mag", "Align_fraction_ref": "AF_genome"})
for m in CDM:
    s = c[c.mag == m].set_index("acc")
    d[f"ani_{m}"] = d.use_acc.map(s.ANI); d[f"afmag_{m}"] = d.use_acc.map(s.AF_mag)
cf = c[c.acc.isin(final)].sort_values(["mag", "ANI"], ascending=[True, False])
cf["rank"] = cf.groupby("mag").cumcount() + 1
info = d.set_index("use_acc")[["strain", "geo_loc_name", "assembly_level"]]
cf = cf.join(info, on="acc")
near = cf[cf["rank"] <= 4].copy(); near["scope"] = "final human-gut set (pre-fixed)"
ap = c[c.acc.isin(d[d.pass_ani].use_acc)].sort_values(["mag", "ANI"], ascending=[True, False])
ap["rank"] = ap.groupby("mag").cumcount() + 1
ap = ap[ap["rank"] <= 4].join(info, on="acc"); ap["scope"] = "INFO ONLY: all 40 ANI-pass parvula_A"
ap["in_final_set"] = ap.acc.isin(final); near["in_final_set"] = True

# prior 15
g = pd.read_csv(f"{O}/gtdb_r226_prior15_rows.tsv", sep="\t", header=None,
                names=["gacc", "gc", "gx", "grep_", "tax", "gca"])
g["num"] = g.gacc.str[7:]; g = g.set_index("num")
a = pd.read_csv(f"{O}/ani_all_vs_rep_cdm.tsv", sep="\t"); a["q"] = a.Query_file.map(bn); a["r"] = a.Ref_file.map(bn)
rows = []
for i, acc in enumerate(open(f"{O}/acc_prior15.txt").read().split()):
    num = acc[4:]
    r = dict(accession=acc, role="pangenome ref (11)" if i < 11 else "nearest-genome conservation ref (4)",
             gtdb_r226_species=g.tax[num].split(";s__")[-1] if num in g.index else "(not in GTDB r226)",
             gtdb_checkm_comp=g.gc.get(num), gtdb_checkm_cont=g.gx.get(num))
    r["is_parvula_A"] = r["gtdb_r226_species"] == "Veillonella parvula_A"
    for ref, lab in [(rep, "rep")] + [(m, m) for m in CDM]:
        s = a[(a.q == acc) & (a.r == ref)]
        r[f"ani_{lab}"] = s.ANI.iloc[0] if len(s) else np.nan
        r[f"af_query_{lab}"] = s.Align_fraction_query.iloc[0] if len(s) else np.nan
    r["ani95_af60_vs_rep"] = bool(r["ani_rep"] >= 95 and r["af_query_rep"] >= 60)
    r["in_final_set"] = acc in final
    r["in_candidate_list"] = acc in set(d.use_acc)
    rows.append(r)
p15 = pd.DataFrame(rows)

# info (a): host missing + gut keyword (NOT the pre-fixed rule); QC by GTDB CheckM (v1) because CheckM v1.2.4 not run on them
ia = d[d.info_a_hostmissing_gut].copy()
ia["gtdb_qc_pass"] = (ia.gtdb_checkm_comp >= 90) & (ia.gtdb_checkm_cont < 5)
ia["ncbi_qc_pass"] = (ia.ncbi_checkm_comp >= 90) & (ia.ncbi_checkm_cont.fillna(0) < 5)
u = list(q.use_acc) + list(ia[ia.gtdb_qc_pass].use_acc)
cu = cluster(u); ia["union_cluster"] = ia.use_acc.map(cu)
n_union_clusters = len(set(cu.values()))
un = list(q.use_acc) + list(ia[ia.ncbi_qc_pass].use_acc)
n_union_clusters_ncbi = len(set(cluster(un).values()))

flow = pd.DataFrame([
    ("GTDB r226 s__Veillonella parvula_A, all", len(d)),
    ("with NCBI accession (datasets summary record present)", int(d.accession.notna().sum())),
    ("FASTA obtained from NCBI FTP", int(d.fasta_ncbi.sum())),
    ("ANI>=95 & AF(query)>=60 to the representative genome", int(d.pass_ani.sum())),
    ("human gut (host=Homo sapiens & gut keyword, no non-gut keyword)", int(d.humangut_eligible.sum())),
    ("CheckM v1.2.4 comp>=90 & cont<5", int(d.pass_qc.sum())),
    ("after ANI>=99.9 de-duplication (final)", len(final))], columns=["step", "n_genomes"])
hab = d[d.pass_ani].groupby("habitat").agg(n=("use_acc", "size"), host_human=("host_human", "sum"),
      host_missing=("host_missing", "sum"), genome_isolate=("ncbi_genome_category", lambda s: int((s == "none").sum())),
      mag=("ncbi_genome_category", lambda s: int((s != "none").sum()))).sort_values("n", ascending=False).reset_index()

d.to_csv(outs[0], index=False)
cols = ["use_acc", "strain", "geo_loc_name", "assembly_level", "ncbi_genome_category", "host", "isolation_source",
        "body_site", "n_contigs", "size", "release_date", "checkm124_comp", "checkm124_cont", "ncbi_checkm_comp",
        "ncbi_checkm_cont", "gtdb_checkm_comp", "gtdb_checkm_cont", "ani_rep", "af_query_rep", "pass_qc", "qs",
        "cluster_id", "cluster_rep", "cluster_size", "final_set", "is_atypical", "genome_notes"] + \
       [f"ani_{m}" for m in CDM]
d[d.humangut_eligible][cols].to_csv(outs[1], index=False)
q[["cluster_id", "cluster_rep", "cluster_size", "use_acc", "strain", "is_cluster_rep", "checkm124_comp",
   "checkm124_cont", "qs", "n_contigs", "max_ani_to_other_member"]].to_csv(outs[2], index=False)
p15.to_csv(outs[3], index=False); cf.to_csv(outs[4], index=False)
pd.concat([near, ap]).to_csv(outs[5], index=False); flow.to_csv(outs[6], index=False)
ia[["use_acc", "strain", "geo_loc_name", "assembly_level", "ncbi_genome_category", "isolation_source", "body_site",
    "ani_rep", "af_query_rep", "gtdb_checkm_comp", "gtdb_checkm_cont", "ncbi_checkm_comp", "ncbi_checkm_cont",
    "gtdb_qc_pass", "ncbi_qc_pass", "union_cluster", "is_atypical", "genome_notes"]].to_csv(outs[7], index=False)
hab.to_csv(outs[8], index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40)
print(flow.to_string(index=False)); print("\n[habitat of ANI-pass]\n", hab.to_string(index=False))
print("\n[human-gut set]\n", d[d.humangut_eligible][["use_acc", "strain", "geo_loc_name", "assembly_level", "n_contigs", "checkm124_comp", "checkm124_cont", "ncbi_checkm_comp", "ncbi_checkm_cont", "gtdb_checkm_comp", "gtdb_checkm_cont", "ani_rep", "af_query_rep", "pass_qc", "final_set"]].to_string(index=False))
print("\n[clusters]\n", q[["cluster_id", "cluster_rep", "use_acc", "qs", "n_contigs", "max_ani_to_other_member"]].to_string(index=False))
print("\n[prior 15]\n", p15.to_string(index=False))
print("\n[nearest 4]\n", pd.concat([near, ap])[["scope", "mag", "rank", "acc", "ANI", "AF_mag", "strain", "in_final_set"]].to_string(index=False))
print(f"\n[info a] host-missing + gut keyword, ANI pass: {len(ia)} | GTDB-CheckM QC pass {int(ia.gtdb_qc_pass.sum())} | NCBI-CheckM QC pass {int(ia.ncbi_qc_pass.sum())} (NCBI value missing {int(ia.ncbi_checkm_comp.isna().sum())})")
print(f"   union (QC-passed human-gut {len(q)} + info-a GTDB-QC-pass {int(ia.gtdb_qc_pass.sum())}) = {len(u)} -> {n_union_clusters} clusters at 99.9")
print(f"   union with NCBI-CheckM QC instead = {len(un)} -> {n_union_clusters_ncbi} clusters at 99.9")
