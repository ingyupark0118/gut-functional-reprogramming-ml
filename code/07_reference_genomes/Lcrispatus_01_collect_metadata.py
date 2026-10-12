#!/usr/bin/env python
"""NCBI datasets JSON Lines -> metadata table (steps S1-S3, S6).

Input: lc_all.jsonl  (datasets summary genome taxon "Lactobacillus crispatus"
                      --assembly-source all --as-json-lines, 2026-10-01)
Output: T_P3_lc_all_metadata.csv   (all 2,065 records)
       T_P3_lc_typematerial.csv   (S2: type material records)
Log  : logs/LOG_P3_50_metadata.txt
Stops if an output file already exists.
"""
import json
import os
import sys

import pandas as pd

P3 = "./refs"
SRC = f"{P3}/lc_all.jsonl"
outs = [f"{P3}/T_P3_lc_all_metadata.csv", f"{P3}/T_P3_lc_typematerial.csv",
        f"{P3}/logs/LOG_P3_50_metadata.txt"]
if any(os.path.exists(f) for f in outs):
    sys.exit("output exists, back up and remove first: "
             + ", ".join(f for f in outs if os.path.exists(f)))
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


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


rows, tm_rows = [], []
for line in open(SRC):
    r = json.loads(line)
    ai = r.get("assembly_info", {}) or {}
    bs = ai.get("biosample", {}) or {}
    at = bs.get("attributes", []) or []
    st = r.get("assembly_stats", {}) or {}
    ani = r.get("average_nucleotide_identity", {}) or {}
    bam = ani.get("best_ani_match", {}) or {}
    cm = r.get("checkm_info", {}) or {}
    ar = ai.get("atypical", {}) or {}
    org = r.get("organism", {}) or {}
    tmat = r.get("type_material", {}) or {}
    row = dict(
        accession=r.get("accession"),
        source_db=r.get("source_database", "").replace("SOURCE_DATABASE_", ""),
        paired_accession=r.get("paired_accession"),
        organism=org.get("organism_name"),
        taxid=org.get("tax_id"),
        strain=(org.get("infraspecific_names", {}) or {}).get("strain") or bsget(at, BS_KEYS["strain_bs"]),
        type_material=tmat.get("type_display_text"),
        assembly_level=ai.get("assembly_level"),
        assembly_type=ai.get("assembly_type"),
        refseq_category=ai.get("refseq_category"),
        is_atypical=bool(ar.get("is_atypical")) if ar else False,
        atypical_warnings="; ".join(ar.get("warnings", []) or []) if ar else None,
        ncbi_ani=bam.get("ani"),
        ncbi_ani_type_assembly=bam.get("assembly"),
        ncbi_ani_category=bam.get("category"),
        taxonomy_check_status=ani.get("taxonomy_check_status"),
        match_status=ani.get("match_status"),
        ncbi_checkm_comp=cm.get("completeness"),
        ncbi_checkm_cont=cm.get("contamination"),
        host=bsget(at, BS_KEYS["host"]),
        isolation_source=bsget(at, BS_KEYS["isolation_source"]),
        body_site=bsget(at, BS_KEYS["body_site"]),
        geo_loc_name=bsget(at, BS_KEYS["geo_loc_name"]),
        collection_date=bsget(at, BS_KEYS["collection_date"]),
        release_date=ai.get("release_date"),
        biosample=bs.get("accession"),
        size=st.get("total_sequence_length"),
        n_contigs=st.get("number_of_contigs"),
        contig_n50=st.get("contig_n50"),
    )
    rows.append(row)
    if tmat:
        tm_rows.append(dict(accession=row["accession"], organism=row["organism"],
                            strain=row["strain"], type_material=tmat.get("type_display_text"),
                            label=tmat.get("type_label"), level=row["assembly_level"],
                            source_db=row["source_db"]))

d = pd.DataFrame(rows)
for c in ("size", "n_contigs", "contig_n50", "ncbi_ani", "ncbi_checkm_comp", "ncbi_checkm_cont"):
    d[c] = pd.to_numeric(d[c], errors="coerce")

P("=" * 78)
P("S1-S3/S6 | NCBI datasets 18.38.0 | taxon 'Lactobacillus crispatus' --assembly-source all")
P("=" * 78)
P(f"S1 total records      : {len(d)}")
P(f"   source_db          : {d.source_db.value_counts().to_dict()}")
P(f"   taxid values       : {d.taxid.nunique()} (check that sub-taxa are included)")
P(f"   with GCA/GCF pair  : {d.paired_accession.notna().sum()}")
P("\nS2 type material records:")
tm = pd.DataFrame(tm_rows)
for _, x in tm.iterrows():
    P(f"   {x.accession}  {x.source_db:8s} {str(x.strain):16s} {x.type_material}")
P(f"   assembly used as reference by NCBI ANI: {d.ncbi_ani_type_assembly.value_counts().head(3).to_dict()}")
P("\nS3 taxonomy_check_status:")
P(f"   {d.taxonomy_check_status.fillna('(none)').value_counts().to_dict()}")
P(f"   match_status       : {d.match_status.fillna('(none)').value_counts().to_dict()}")
P(f"\nS6 atypical=True      : {int(d.is_atypical.sum())}")
P(f"   with NCBI CheckM   : {int(d.ncbi_checkm_comp.notna().sum())} / {len(d)}")
P(f"   assembly_level     : {d.assembly_level.value_counts().to_dict()}")
P(f"   host stated        : {int(d.host.notna().sum())}  isolation_source stated: {int(d.isolation_source.notna().sum())}  body_site stated: {int(d.body_site.notna().sum())}")

d.to_csv(outs[0], index=False)
tm.to_csv(outs[1], index=False)
open(outs[2], "w").write("\n".join(log) + "\n")
P("[written] " + ", ".join(os.path.basename(f) for f in outs))
