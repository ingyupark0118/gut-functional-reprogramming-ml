#!/usr/bin/env python
"""vpr_15 - build the sensitivity set and the set/MAG tables.
Usage: LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python vpr_15_sens_set.py
Rule (fixed before running): main 11 + host-missing gut parvula_A genomes (T_V1_info_a_hostmissing.csv, 17) that pass
CheckM v1.2.4 completeness >=90 and contamination <5 (sens/checkm_sens17.tsv, run here), then single-linkage de-duplication of
the union at skani ANI >= 99.9 (vp_refs/ani_triangle_all.tsv); cluster representative = max(comp - 5*cont), tie = fewest contigs
(same rule as vp_refs/vp_40_final.py). NCBI-atypical GCA_963586375.1 is NOT excluded (no such pre-fixed rule) but flagged."""
import os, sys
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
W = "./refs/vp_reanalysis"; VR = "./refs/vp_refs"
outs = [f"{W}/sens/acc_sens.txt", f"{W}/sens/acc_sens_additions.txt", f"{W}/T_sens_flow.csv", f"{W}/T_sens_dedup_clusters.csv", f"{W}/T_set_members.csv", f"{W}/T_MAGs.csv", f"{W}/logs/LOG_sens_set.txt"]
if any(os.path.exists(f) for f in outs): sys.exit("output exists")
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
CDM = ["CDM-1_bin.1", "CDM-4_bin.2", "CDM-3_bin.12"]
d = pd.read_csv(f"{VR}/T_V1_candidates_all.csv").set_index("use_acc", drop=False)
main = open(f"{W}/main/acc_main11.txt").read().split(); add17 = open(f"{W}/sens/acc_hostmissing17.txt").read().split()
ck = pd.read_csv(f"{W}/sens/checkm_sens17.tsv", sep="\t").set_index("Bin Id"); assert set(ck.index) == set(add17), "CheckM output incomplete"
d.loc[add17, "checkm124_comp"] = ck.Completeness.reindex(add17).values; d.loc[add17, "checkm124_cont"] = ck.Contamination.reindex(add17).values
d.loc[add17, "checkm124_lineage"] = ck["Marker lineage"].reindex(add17).values
passed = [a for a in add17 if d.checkm124_comp[a] >= 90 and d.checkm124_cont[a] < 5]
union = main + passed
L = open(f"{VR}/ani_triangle_all.tsv").read().rstrip("\n").split("\n")[1:]
nm = [os.path.basename(l.split("\t")[0])[:-4] for l in L]
M = pd.DataFrame([[float(x) for x in l.split("\t")[1:]] for l in L], index=nm, columns=nm)
q = d.loc[union].copy(); q["qs"] = q.checkm124_comp - 5 * q.checkm124_cont
q["cluster_id"] = connected_components((M.loc[union, union].values >= 99.9).astype(int), directed=False)[1]
q = q.sort_values(["cluster_id", "qs", "n_contigs"], ascending=[True, False, True])
q["cluster_rep"] = q.groupby("cluster_id").use_acc.transform("first"); q["cluster_size"] = q.groupby("cluster_id").use_acc.transform("size")
q["max_ani_to_other_member"] = [M.loc[a, [b for b in union if b != a]].max() for a in q.use_acc]
q["in_main"] = q.use_acc.isin(main)
sens = sorted(q[q.use_acc == q.cluster_rep].use_acc)
dropped_main = [a for a in main if a not in sens]
flow = pd.DataFrame([("main set (pre-fixed rule)", len(main)), ("host-missing gut parvula_A candidates (T_V1_info_a_hostmissing.csv)", len(add17)),
    ("of these: CheckM v1.2.4 completeness>=90 & contamination<5", len(passed)), ("union main + passing additions", len(union)),
    ("clusters at ANI>=99.9 (single linkage) = sensitivity set", len(sens)), ("main-set genomes replaced by another cluster representative", len(dropped_main)),
    ("NCBI-atypical genomes kept in sensitivity set (flagged, not excluded)", int(d.loc[sens].is_atypical.astype(str).str.lower().eq("true").sum()))], columns=["step", "n_genomes"])
c = pd.read_csv(f"{VR}/ani_cdm_vs_public.tsv", sep="\t"); bn = lambda f: os.path.basename(f)[:-4]
c["mag"] = c.Query_file.map(bn); c["acc"] = c.Ref_file.map(bn)
for m in CDM:
    s = c[c.mag == m].set_index("acc"); d[f"ANI_{m}"] = d.use_acc.map(s.ANI); d[f"AFmag_{m}"] = d.use_acc.map(s.Align_fraction_query)
allm = sorted(set(main) | set(add17))
T = d.loc[allm, ["use_acc", "strain", "geo_loc_name", "ncbi_genome_category", "assembly_level", "n_contigs", "size", "host", "isolation_source", "habitat",
                 "checkm124_comp", "checkm124_cont", "gtdb_checkm_comp", "gtdb_checkm_cont", "is_atypical", "genome_notes"] + [f"{p}_{m}" for m in CDM for p in ("ANI", "AFmag")]].copy()
T.insert(1, "gtdb_r226_species", "Veillonella parvula_A")
T.insert(2, "in_main_set", T.use_acc.isin(main)); T.insert(3, "in_sensitivity_set", T.use_acc.isin(sens))
T.insert(4, "sens_note", [("main" if a in main else "added (host field empty)") if a in sens else ("CheckM fail" if a not in union else f"duplicate of {q.cluster_rep[a]} (ANI>=99.9)") for a in T.use_acc])
T["checkm_version"] = "v1.2.4 (lineage_wf); main: vp_refs/checkm_humangut.tsv, additions: vp_reanalysis/sens/checkm_sens17.tsv"
T["flag"] = np.where(T.is_atypical.astype(str).str.lower() == "true", "NCBI atypical (unverified source organism) - kept per pre-fixed rule", "")
LOWQ = open("data/sample_lists/excluded_low_quality_bin.txt").read().strip()   # sample of the V. parvula bin excluded for low completeness (name not distributed)
mq = pd.read_csv("./mag_qc/T_A5_mag_quality.csv"); mq = mq[mq["Bin Id"].str.startswith("V.parvula_CDM")]
rep = pd.read_csv(f"{VR}/T_V1_cdm_ani_to_rep.csv") if os.path.exists(f"{VR}/T_V1_cdm_ani_to_rep.csv") else None
G = pd.DataFrame(dict(MAG=mq["Bin Id"].str.replace("V.parvula_", ""), roary_column=mq["Bin Id"].str.replace("V.parvula_", ""), gtdb_r226_species=["Veillonella parvula_F" if LOWQ in b else "Veillonella parvula_A" for b in mq["Bin Id"]],
     completeness=mq.Completeness.round(2), contamination=mq.Contamination.round(2), quality_score_comp_minus_5cont=(mq.Completeness - 5 * mq.Contamination).round(2),
     genome_size=mq["Genome size"], n_contigs=mq["# contigs"], N50=mq["N50 (contigs)"]))
G["used_in_reanalysis"] = ~G.MAG.str.contains(LOWQ)
G["note"] = ["excluded (completeness 33%, GTDB parvula_F)" if LOWQ in m else ("contamination 7.11% >= 5%: included because quality score > 50 (manuscript MAG criterion); fails the reference criterion" if "CDM-4" in m else "") for m in G.MAG]
G["checkm_source"] = "original CheckM run quoted from mag_qc/T_A5_mag_quality.csv (not re-run here)"
open(outs[0], "w").write("\n".join(sens) + "\n"); open(outs[1], "w").write("\n".join(a for a in sens if a not in main) + "\n")
flow.to_csv(outs[2], index=False)
q[["cluster_id", "cluster_rep", "cluster_size", "use_acc", "in_main", "strain", "checkm124_comp", "checkm124_cont", "qs", "n_contigs", "max_ani_to_other_member"]].to_csv(outs[3], index=False)
T.to_csv(outs[4], index=False); G.sort_values("MAG").to_csv(outs[5], index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 38, "display.max_rows", 100)
P(flow.to_string(index=False)); P("\nCheckM v1.2.4 on the 17 additions:\n" + ck[["Marker lineage", "Completeness", "Contamination", "Strain heterogeneity"]].to_string())
P("\nfailed CheckM:", [a for a in add17 if a not in passed]); P("clusters with >1 member:\n" + q[q.cluster_size > 1][["cluster_id", "cluster_rep", "use_acc", "in_main", "qs", "n_contigs", "max_ani_to_other_member"]].to_string(index=False))
P("main genomes replaced:", dropped_main); P(f"\nsensitivity set n={len(sens)}:", sens)
P("\n" + T[["use_acc", "in_main_set", "in_sensitivity_set", "sens_note", "habitat", "checkm124_comp", "checkm124_cont"] + [f"ANI_{m}" for m in CDM] + ["flag"]].to_string(index=False))
P("\n" + G.to_string(index=False))
open(outs[6], "w").write("\n".join(log) + "\n")
