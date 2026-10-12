#!/usr/bin/env python
"""Angle E step 1: rule-only habitat re-classification of set A (1,185 L. crispatus genomes).
Uses NCBI metadata only (lc_all.jsonl); NO ANI to the CDM MAGs is read or computed here.
Inputs : ../../lc_all.jsonl, ../../T_P3_50_candidates.csv (read-only)
Outputs: T_E_habitat_reclass.csv, T_E_habitat_confusion.csv, T_E_ontology_codes.csv
Rule text: DESIGN_E_prefixed.md section 2 (keyword lists below are the authoritative copy).
"""
import json, re, os, sys, collections
import pandas as pd

P3 = "./refs"
E = f"{P3}/rescue/E"
outs = [f"{E}/T_E_habitat_reclass.csv", f"{E}/T_E_habitat_confusion.csv", f"{E}/T_E_ontology_codes.csv"]
if any(os.path.exists(f) for f in outs):
    sys.exit("output exists (move to _discard first): " + ", ".join(f for f in outs if os.path.exists(f)))

# ---- fields
PRIMARY = ["isolation_source", "host_tissue_sampled", "host_body_product", "host_body_habitat",
           "host_sample", "misc_param: hmp body site"]
SECONDARY = ["sample_type", "env_broad_scale", "env_local_scale", "env_medium", "metagenome_source",
             "metagenomic source", "local environmental context", "urine_collect_meth",
             "samp_collect_device", "source_type"]
HOSTF = ["host", "specific_host", "host_scientific_name"]
NULL = {"", "none", "nan", "missing", "not applicable", "not collected", "not provided", "unknown", "na", "n/a"}

# ---- keyword lists (regex, applied to lower-cased text, '_' -> ' ')
KW = collections.OrderedDict([
    ("gut", r"\bfa?ec(es|al)\w*|\bstool\w*|\bgut\b|intestin|\bcolon(ic)?\b|\brect(al|um)\b|\bile(um|al)\b|\bca?ec(um|al|a)\b|duoden|jejun|\bdigesta\b|gastrointestinal"),
    ("vagina", r"vagin|cervi[cx]|menstrual|softcup|\bgenital tract"),
    ("urinary", r"\burin|bladder|urethr|catheter"),
    ("oral", r"\boral\b|saliva|tongue|dental|plaque|\bmouth\b|gingiv|buccal"),
    ("other_site", r"\beye\b|\bskin\b|milk|breast|nasal|sputum|\blung\b|\bblood\b|wound|\bear\b"),
])
ANIMAL = r"chicken|gallus|poultry|broiler|brolier|turkey|meleagris|\bpig\b|sus scrofa|\bboar\b|swine|equine|equus|bovine|\bcow\b|\bmouse\b|murine"
FOODENV = r"sourdough|bioreactor|sediment|stillage|sludge|ferment|\bfood\b|dairy|silage|\bsoil\b|\bwater\b|tidal"
HUMAN = r"\bhuman\b|homo sapiens|\bwom[ae]n\b|\binfant\b|newborn|\bfemale\b"
PKG = [("human-vaginal", "vagina"), ("human-gut", "gut"), ("human-oral", "oral"), ("human-skin", "other_site")]
LABEL = {"gut": "human gut", "vagina": "vagina", "urinary": "urinary", "oral": "oral",
         "other_site": "other human body site"}
CODE = re.compile(r"^(ENVO|UBERON|NCIT|PCO)[:_]\s?C?\d+$", re.I)
LAB_CODE = re.compile(r"([^\[\]|;]+?)'?\s*\[((?:ENVO|UBERON|NCIT|PCO)[:_]C?\d+)\]", re.I)

recs = [json.loads(l) for l in open(f"{P3}/lc_all.jsonl")]
# ontology dictionary resolvable from a file on disk = labelled occurrences "label [CODE]" in lc_all.jsonl itself
onto = {}
for r in recs:
    for a in ((r.get("assembly_info", {}).get("biosample") or {}).get("attributes") or []):
        for lab, code in LAB_CODE.findall(str(a.get("value", ""))):
            onto.setdefault(code.upper().replace("_", ":"), lab.strip().lower())

cand = pd.read_csv(f"{P3}/T_P3_50_candidates.csv")
A = cand[cand.pass_ani].set_index("accession")
OLD = {"vagina/cervix": "vagina/cervix", "food/environment/other": "food/environment/other", "poultry/animal": "poultry/animal",
       "urinary": "urinary", "not stated": "not stated", "gut/feces": "gut/feces", "oral": "oral",
       "other": "other", "breast milk": "breast milk"}
code_use = collections.Counter()


def resolve(v, acc):
    v = str(v).strip()
    if CODE.match(v):
        c = v.upper().replace("_", ":").replace(" ", "")
        code_use[(c, onto.get(c, ""))] += 1
        return onto.get(c, f"<unresolved {c}>")
    return v


def classify(d, pkg):
    host = " ; ".join(str(d[k]) for k in HOSTF if k in d and str(d[k]).strip().lower() not in NULL).lower()
    prim = {k: resolve(d[k], None) for k in PRIMARY if k in d and str(d[k]).strip().lower() not in NULL}
    sec = {k: resolve(d[k], None) for k in SECONDARY if k in d and str(d[k]).strip().lower() not in NULL}
    tp = " | ".join(prim.values()).lower().replace("_", " ")
    ts = " | ".join(sec.values()).lower().replace("_", " ")
    allt = tp + " | " + ts
    human = bool(re.search(r"homo sapiens|human", host)) or bool(re.search(HUMAN, allt)) or ("human" in (pkg or "").lower())
    host_nonhuman = bool(host) and not re.search(r"homo sapiens|human", host)
    unresolved = "<unresolved" in allt
    used = "; ".join([f"host={host}"] + [f"{k}={v}" for k, v in {**prim, **sec}.items()] + [f"package={pkg}"])
    if host_nonhuman or re.search(ANIMAL, allt):
        return "animal/poultry", "host non-human or animal keyword", human, used
    hits = [c for c, rx in KW.items() if re.search(rx, allt)]
    how = "one site class in pooled fields"
    if len(hits) > 1:
        hp = [c for c, rx in KW.items() if re.search(rx, tp)]
        if len(hp) == 1:
            hits, how = hp, f"conflict {'+'.join(hits)} resolved by primary fields"
        else:
            return ("human, site unknown" if human else "unknown"), f"ambiguous site keywords {'+'.join(hits)}", human, used
    if len(hits) == 1:
        c = hits[0]
        if c == "gut" and not human:
            return "gut, host not stated", "gut keyword without human evidence", human, used
        return LABEL[c], how, human, used
    if re.search(FOODENV, allt):
        return "food/environment", "food/environment keyword", human, used
    for key, c in PKG:
        if key in (pkg or "").lower():
            return LABEL[c], f"BioSample package {pkg} (no site keyword in attributes)", human, used
    if human:
        return "human, site unknown", "host human; no body site" + (" (unresolved ontology code only)" if unresolved else ""), human, used
    return "unknown", "no host, no source", human, used


rows = []
for r in recs:
    acc = r["accession"]
    if acc not in A.index:
        continue
    ai = r.get("assembly_info", {}) or {}
    bs = ai.get("biosample") or {}
    d = {a.get("name", "").lower(): a.get("value") for a in (bs.get("attributes") or [])}
    new, why, human, used = classify(d, bs.get("package"))
    src = str(d.get("isolation_source", "")).lower()
    detail = ("menstrual cup" if "menstrual" in src else "")
    cm = r.get("checkm_info") or {}
    rows.append(dict(accession=acc, strain=A.strain[acc], bioproject=ai.get("bioproject_accession"),
                     biosample=bs.get("accession"), assembly_level=ai.get("assembly_level"),
                     old_label_ko=A.habitat[acc], old_label=OLD[A.habitat[acc]], old_setB=bool(A.setB_eligible[acc]),
                     new_label=new, rule_fired=why, human_evidence=human, detail=detail,
                     human_gut_via_infant_keyword_only=False,
                     ncbi_checkm_comp=cm.get("completeness"), ncbi_checkm_cont=cm.get("contamination"),
                     n_contigs=A.n_contigs[acc], fields_used=used))
t = pd.DataFrame(rows).sort_values("accession")
# flag: human-gut calls whose only human evidence is infant/newborn (assumption that these are human)
for i, x in t[t.new_label == "human gut"].iterrows():
    txt = x.fields_used.lower()
    t.loc[i, "human_gut_via_infant_keyword_only"] = not re.search(r"human|homo sapiens|wom[ae]n|female", txt)
assert len(t) == 1185 and t.accession.is_unique
t.to_csv(outs[0], index=False)
conf = pd.crosstab(t.old_label, t.new_label, margins=True, margins_name="TOTAL")
conf.to_csv(outs[1])
pd.DataFrame([dict(code=c, resolved_label=l or "UNRESOLVED", n_field_values=n, resolved_from="lc_all.jsonl labelled values" if l else "")
              for (c, l), n in sorted(code_use.items())]).to_csv(outs[2], index=False)
pd.set_option("display.width", 250)
print(conf.to_string())
print(t.rule_fired.value_counts().to_string())
print(pd.read_csv(outs[2]).to_string())
