#!/usr/bin/env python
"""T19 T3: summarise skani (MAG vs public V. parvula pools).  Reads datasource/skani_4MAG_vs_D3_complex91.tsv + T19_vp_public_pool_members.csv (written by this session's skani run).
Writes datasource/vp_public_comparison_{per_pair,summary}.csv.  Pools: D2 = GTDB r226 parvula_A with local FASTA (n = 40; main), D3 = all local V. parvula-complex genomes (n = 91; auxiliary)."""
import pandas as pd, os
R = "."; O = f"{R}/work_d/vparvula_panels/datasource"
s = pd.read_csv(f"{O}/skani_4MAG_vs_D3_complex91.tsv", sep="\t"); pool = pd.read_csv(f"{O}/vp_public_pool_members.csv").set_index("accession")
s["acc"] = s.Ref_file.map(lambda p: next(a for a in pool.index if os.path.basename(p).startswith(a)))
assert s.acc.map(lambda a: pool.loc[a, "fasta"]).eq(s.Ref_file).all()
QN = {"CDM-1_bin.1.fna": "CDM-1_bin.1", "CDM-4_bin.2_clean.fa": "CDM-4_bin.2 (refined)", "CDM-3_bin.12.fna": "CDM-3_bin.12", "CDM-4_bin.2.fna": "CDM-4_bin.2 (original, before refinement)"}
s["mag"] = s.Query_file.map(lambda p: QN[os.path.basename(p)]); s["in_D2"] = s.acc.map(pool.pool_D2_parvulaA_n40)
s["species"] = s.acc.map(pool.gtdb_r226_cluster); s["is_MAG"] = s.acc.map(pool.MAG); s["habitat"] = s.acc.map(pool.habitat_rule_label); s["strain"] = s.acc.map(pool.strain)
assert len(s) == 4 * 91 and s.groupby("mag").size().eq(91).all()
s[["mag", "acc", "species", "in_D2", "ANI", "Align_fraction_ref", "Align_fraction_query", "ANI_5_percentile", "ANI_95_percentile", "is_MAG", "habitat", "strain"]].sort_values(["mag", "ANI"], ascending=[True, False]).to_csv(f"{O}/vp_public_comparison_per_pair.csv", index=False)
rows = []
for pool_name, sub in (("D2_parvula_A_n40 (main)", s[s.in_D2]), ("D3_complex_n91 (auxiliary)", s)):
    for m, g in sub.groupby("mag"):
        g = g.sort_values("ANI", ascending=False); b = g.iloc[0]
        rows.append(dict(pool=pool_name, n_pool=len(g), mag=m, nearest_accession=b.acc, nearest_ANI=b.ANI, nearest_AF_mag=b.Align_fraction_query, nearest_AF_public=b.Align_fraction_ref, nearest_is_MAG=b.is_MAG, nearest_species=b.species,
                         second_accession=g.iloc[1].acc, second_ANI=g.iloc[1].ANI, **{f"n_ge_{t}": int((g.ANI >= t).sum()) for t in (99.9, 99.7, 99.5, 99.0)}, n_AF_mag_lt_50=int((g.Align_fraction_query < 50).sum()), median_ANI=float(g.ANI.median())))
r = pd.DataFrame(rows); r.to_csv(f"{O}/vp_public_comparison_summary.csv", index=False)
pd.set_option("display.width", 250); print(r.drop(columns=["nearest_species"]).to_string())
