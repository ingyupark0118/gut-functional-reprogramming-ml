#!/usr/bin/env python
"""V1 step 1: GTDB r226 s__Veillonella parvula_A rows + NCBI datasets summary -> candidate table + FTP urls.
BS_KEYS/bsget are the same as p3_50_metadata.py. Stops if outputs exist."""
import json, os, sys
import pandas as pd
O = "./refs/vp_refs"
outs = [f"{O}/T_V1_ncbi_metadata.csv", f"{O}/T_V1_candidates_meta.csv", f"{O}/ftp_urls.txt"]
if any(os.path.exists(f) for f in outs):
    sys.exit("output exists: " + ", ".join(f for f in outs if os.path.exists(f)))
BS_KEYS = {
    "host": ["host", "specific_host", "host_scientific_name"],
    "isolation_source": ["isolation_source", "isolation-source", "env_medium", "source_type"],
    "body_site": ["host_body_site", "body_site", "isolation_site", "host_tissue_sampled",
                  "host_body_product", "env_local_scale"],
    "geo_loc_name": ["geo_loc_name", "geographic location (country and/or sea)", "country"],
    "collection_date": ["collection_date", "collection date"],
    "strain_bs": ["strain", "isolate"],
}
def bsget(attrs, keys):
    d = {a.get("name", "").lower(): a.get("value") for a in attrs or []}
    for k in keys:
        v = d.get(k.lower())
        if v is not None and str(v).strip() != "":
            return v
    return None
rows = []
import itertools
for line in itertools.chain(open(f"{O}/ncbi_summary.jsonl"), open(f"{O}/ncbi_summary2.jsonl")):
    r = json.loads(line)
    ai = r.get("assembly_info", {}) or {}; bs = ai.get("biosample", {}) or {}
    at = bs.get("attributes", []) or []; st = r.get("assembly_stats", {}) or {}
    cm = r.get("checkm_info", {}) or {}; ar = ai.get("atypical", {}) or {}
    org = r.get("organism", {}) or {}
    rows.append(dict(
        accession=r.get("accession"), current_accession=r.get("current_accession"),
        source_db=r.get("source_database", "").replace("SOURCE_DATABASE_", ""),
        paired_accession=r.get("paired_accession"),
        paired_status=((ai.get("paired_assembly") or {}).get("status")),
        assembly_status=ai.get("assembly_status"), assembly_name=ai.get("assembly_name"),
        organism=org.get("organism_name"),
        strain=(org.get("infraspecific_names", {}) or {}).get("strain")
               or (org.get("infraspecific_names", {}) or {}).get("isolate") or bsget(at, BS_KEYS["strain_bs"]),
        assembly_level=ai.get("assembly_level"),
        is_atypical=bool(ar.get("is_atypical")) if ar else False,
        atypical_warnings="; ".join(ar.get("warnings", []) or []) if ar else None,
        genome_notes="; ".join(ai.get("genome_notes", []) or []) or None,
        ncbi_checkm_comp=cm.get("completeness"), ncbi_checkm_cont=cm.get("contamination"),
        host=bsget(at, BS_KEYS["host"]), isolation_source=bsget(at, BS_KEYS["isolation_source"]),
        body_site=bsget(at, BS_KEYS["body_site"]), geo_loc_name=bsget(at, BS_KEYS["geo_loc_name"]),
        collection_date=bsget(at, BS_KEYS["collection_date"]), release_date=ai.get("release_date"),
        biosample=bs.get("accession"),
        biosample_all_attrs=" || ".join(f"{a.get('name')}={a.get('value')}" for a in at),
        size=st.get("total_sequence_length"), n_contigs=st.get("number_of_contigs"),
        contig_n50=st.get("contig_n50")))
n = pd.DataFrame(rows).drop_duplicates("accession")
n.to_csv(outs[0], index=False)
nd = n.set_index("accession")

g = pd.read_csv(f"{O}/gtdb_r226_parvulaA_rows.tsv", sep="\t", dtype=str)
assert (g.gtdb_taxonomy.str.split(";").str[-1] == "s__Veillonella parvula_A").all()
reps = g[g.gtdb_representative == "t"].accession.tolist()
assert len(reps) == 1 and set(g.gtdb_genome_representative) == {reps[0]}, reps
g["gtdb_acc"] = g.accession.str[3:]
g["gca"] = g.ncbi_genbank_assembly_accession
def pick(r):
    """Prefer RefSeq GCF when one exists and is current at NCBI today; else the GenBank GCA."""
    gcf = [r.gtdb_acc] if r.gtdb_acc.startswith("GCF_") else []
    if r.gca in nd.index and str(nd.loc[r.gca].paired_accession).startswith("GCF_"):
        gcf.append(nd.loc[r.gca].paired_accession)
    for c in gcf:
        if c in nd.index and nd.loc[c].assembly_status == "current":
            return c
    return r.gca if r.gca in nd.index else None
g["use_acc"] = g.apply(pick, axis=1)
keep = ["use_acc", "gtdb_acc", "gca", "gtdb_representative", "checkm_completeness", "checkm_contamination",
        "checkm2_completeness", "checkm2_contamination", "contig_count", "genome_size", "ncbi_genome_category",
        "ncbi_assembly_level", "ncbi_country", "ncbi_isolation_source", "ncbi_strain_identifiers", "ncbi_date"]
c = g[keep].rename(columns={"checkm_completeness": "gtdb_checkm_comp", "checkm_contamination": "gtdb_checkm_cont",
                            "checkm2_completeness": "gtdb_checkm2_comp", "checkm2_contamination": "gtdb_checkm2_cont",
                            "ncbi_country": "gtdb_ncbi_country", "ncbi_isolation_source": "gtdb_ncbi_isolation_source"})
c = c.merge(n.add_prefix("n_"), left_on="use_acc", right_on="n_accession", how="left")
c.columns = [x[2:] if x.startswith("n_") and x != "n_contigs" else x for x in c.columns]
c.to_csv(outs[1], index=False)
print("GTDB parvula_A rows:", len(g), "| representative:", reps[0])
print("with NCBI record:", c.accession.notna().sum(), "| GCF used:", c.use_acc.str.startswith("GCF_").sum(),
      "| GCA used:", c.use_acc.str.startswith("GCA_").sum(), "| none:", c.use_acc.isna().sum())
print("assembly_status:", c.assembly_status.value_counts(dropna=False).to_dict())
print("atypical:", int(c.is_atypical.sum()), "| NCBI CheckM present:", int(c.ncbi_checkm_comp.notna().sum()))
# FTP urls: candidates + 15 prior refs (GCF; fall back to GCA if GCF not in NCBI summary)
want = list(c.use_acc.dropna())
for a in open(f"{O}/acc_prior15.txt").read().split():
    b = a if a in nd.index else a.replace("GCF_", "GCA_")
    print("prior ref", a, "->", b if b in nd.index else "NOT FOUND AT NCBI",
          "" if b not in nd.index else nd.loc[b].assembly_status)
    if b in nd.index and b not in want: want.append(b)
with open(outs[2], "w") as fh:
    for a in want:
        nm = str(nd.loc[a].assembly_name).replace(" ", "_")
        d = a[4:13]
        fh.write(f"{a}\thttps://ftp.ncbi.nlm.nih.gov/genomes/all/{a[:3]}/{d[:3]}/{d[3:6]}/{d[6:9]}/{a}_{nm}/{a}_{nm}_genomic.fna.gz\n")
print("ftp urls:", len(want))
