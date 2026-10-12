#!/usr/bin/env python
"""Selection of the representative n = 14 sub-sample network of each group and re-computation of network statistics.
 (a) representative sub-sample per group, chosen by a rule fixed beforehand: the sub-sample closest to the group medians of edge number, modularity and cross-phylum edge fraction
 (b) independent re-computation of edges, modularity and cross-phylum fraction from the stored correlation matrices for the full-group and representative networks
No SparCC estimation is run here."""
import sys, re, platform
sys.path.insert(0, __import__("os").path.dirname(__file__))
from sr_common import *
T = f"{OUT}/tables"; LOG = open(fresh(f"{OUT}/logs/LOG_s1_select_and_tables.txt"), "w", encoding="utf8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()
F_DIST = {a: f"{P9}/tables/T_sparcc_subsample_distribution_{a}.csv" for a in ("filtered", "unfiltered")}
F_RET = f"{P9}/tables/T_sparcc_filter_retention.csv"; F_C1M = f"{C1}/T_C1_main.csv"; F_OREC = f"{C1}/T_original_recomputed.csv"
F_X5 = f"{P9}/_crosscheck/X5_rerun_same_and_different_seed.csv"
OSUM = {"NC": "NCnon", "DM": "DMnon", "Cirrhosis": "NCcirr", "CDM": "DMcirr"}; F_OSUM = {g: f"{ORIG}/network2/{v}_network_summary_test0722.csv" for g, v in OSUM.items()}
INPUTS = [F_REP, F_TAX, F_DRAWS, *F_DIST.values(), F_RET, F_C1M, F_OREC, F_X5, *F_OSUM.values(), f"{ORIG}/sparcc_networkRcode.txt", f"{OUT}/DESIGN_prefixed.md",
          f"{C1}/inputs/species_s_counts_raw.csv", f"{REV}/nested_cv/labels.csv", __file__, __import__("sr_common").__file__]
log(f"s1_select_and_tables   {time.strftime('%F %T %Z')}   host={platform.node()}   python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__} networkx {nx.__version__}")
log("design:", open(f"{OUT}/logs/DESIGN_sha256.txt").read().strip()); assert sha(f"{OUT}/DESIGN_prefixed.md") in open(f"{OUT}/logs/DESIGN_sha256.txt").read()
log("inputs (sha256  path):"); [log(" ", sha(p), p) for p in INPUTS]
R = pd.read_csv(F_REP); DIST = {a: pd.read_csv(f).set_index(["metric", "group"]) for a, f in F_DIST.items()}
lab = pd.read_csv(f"{REV}/nested_cv/labels.csv", index_col=0)["Group"]; assert lab.value_counts().to_dict() == NEXP
draws = pd.read_csv(F_DRAWS)

# ------------------------------------------------------------------ (a) representative replicates
rep_rows, REPK = [], {}
for an in ("filtered", "unfiltered"):
    for g in SUBG:
        d = representative(R, an, g); b = d.iloc[0]; REPK[(an, g)] = int(b.draw)
        tie = int((np.isclose(d.distance, b.distance, rtol=0, atol=1e-12)).sum())
        ids = draws[(draws.group == g) & (draws.draw == b.draw)].iloc[0, 2:].tolist(); assert len(ids) == 14 and all(lab[i] == g for i in ids)
        row = dict(analysis=an, input="I-C raw 1,338 species + prevalence >= 3 filter" if an == "filtered" else "I-B raw 1,338 species, no filter", group=g, n_group=NEXP[g], n_subsample=14,
                   chosen_draw=int(b.draw), network_id=b.id, distance_z=b.distance, n_replicates_at_same_distance=tie,
                   second_draw=int(d.draw[1]), second_distance_z=d.distance[1], third_draw=int(d.draw[2]), third_distance_z=d.distance[2], median_distance_of_100=float(d.distance.median()))
        for s in STATS:
            x = R[(R.analysis == an) & (R.group == g) & (R.draw >= 0)][s].values; sd = x.std(ddof=1)
            row.update({f"{s}_chosen": b[s], f"{s}_median_of_100": np.median(x), f"{s}_difference": b[s] - np.median(x), f"{s}_difference_in_SD": (b[s] - np.median(x)) / sd,
                        f"{s}_percentile_of_chosen": 100 * ((x < b[s]).sum() + 0.5 * (x == b[s]).sum()) / len(x)})
            assert np.isclose(np.median(x), DIST[an].loc[(s, g), "p50"], rtol=1e-12)                     # medians = distribution table
        row.update(nodes_chosen=int(b.nodes), n_species_chosen=int(b.n_species), sample_ids=" ".join(ids)); rep_rows.append(row)
        log(f"representative {an:10s} {g:9s}: draw {int(b.draw):3d}  distance {b.distance:.4f} (2nd: draw {int(d.draw[1])} {d.distance[1]:.4f}; ties {tie})  "
            + "  ".join(f"{s} {b[s]:.4g} vs median {np.median(R[(R.analysis == an) & (R.group == g) & (R.draw >= 0)][s].values):.4g}" for s in STATS))
REP = pd.DataFrame(rep_rows); REP.to_csv(fresh(f"{T}/T_representative_replicates.csv"), index=False)

# ------------------------------------------------------------------ (b) independent re-computation from the stored matrices
chk = []
for an in ("filtered", "unfiltered"):
    for g, k in [(g, -1) for g in GROUPS] + [(g, REPK[(an, g)]) for g in SUBG]:
        st, ed, samples = network(an, g, k)
        s = R[(R.analysis == an) & (R.id == st["id"])].iloc[0]                                         # stored replicate table
        src2 = json.load(open(f"{NETDIR[an]}/{st['id']}.json"))                                        # stored per-network json (density, mean degree)
        e2 = pd.read_csv(f"{NETDIR[an]}/{st['id']}_edges.csv")
        assert st["edges"] == int(s.edges) == src2["edges"] == len(e2), (st["id"], st["edges"], s.edges)
        assert set(map(frozenset, zip(ed.a, ed.b))) == set(map(frozenset, zip(e2.taxon_a, e2.taxon_b)))  # same edge set as the stored edge list
        assert st["nodes"] == int(s.nodes) == src2["nodes"] and st["n_samples"] == int(s.n_samples) == (14 if k >= 0 or g == "CDM" else NEXP[g])
        if k >= 0: assert samples == draws[(draws.group == g) & (draws.draw == k)].iloc[0, 2:].tolist()
        dm, dc = abs(st["modularity"] - s.modularity), abs(st["cross_phylum_frac"] - s.cross_phylum_frac)
        assert dm < 1e-9 and dc < 1e-12 and abs(st["density"] - src2["density"]) < 1e-12 and abs(st["mean_degree"] - src2["mean_degree"]) < 1e-12, (st["id"], dm, dc)
        chk.append(dict(st, stored_edges=int(s.edges), stored_modularity=s.modularity, stored_cross_phylum_frac=s.cross_phylum_frac, abs_diff_modularity=dm, abs_diff_cross=dc,
                        role="full group" if k < 0 else "representative n = 14 sub-sample", n_species_in_analysis=int(s.n_species)))
        log(f"recomputed {st['id']:28s} nodes {st['nodes']:4d} edges {st['edges']:6d} (+{st['edges_pos']}/-{st['edges_neg']}) mod {st['modularity']:.6f} cross {st['cross_phylum_frac']:.6f} | stored: equal (|d mod| {dm:.1e}, |d cross| {dc:.1e})")
CHK = pd.DataFrame(chk); CHK.to_csv(fresh(f"{T}/T_selfcheck_recomputed_networks.csv"), index=False)
log(f"self-check: {len(CHK)} networks re-computed from npz with own code; edge sets, edge counts, nodes, density, mean degree, modularity, cross-phylum all equal to the stored values")

# ------------------------------------------------------------------ (c) Supplementary Table 3
ret = pd.read_csv(F_RET).set_index("group"); c1m = pd.read_csv(F_C1M); c1m = c1m[c1m.taxon_set == "all1338"].set_index("group"); orec = pd.read_csv(F_OREC).set_index("group")
def pct_mid(x, v): return 100 * ((x < v).sum() + 0.5 * (x == v).sum()) / len(x)
rows = []
for an, block, inp in [("filtered", "1 main analysis (I-C)", "raw MetaPhlAn species counts (1,338 species), species detected in >= 3 samples of the analysed sample set, SparCC re-estimated"),
                       ("unfiltered", "2 unfiltered companion (I-B)", "raw MetaPhlAn species counts (1,338 species), no prevalence filter")]:
    cdm = {s: float(R[(R.analysis == an) & (R.group == "CDM")][s].iloc[0]) for s in STATS}
    for g in GROUPS:
        c = CHK[(CHK.analysis == an) & (CHK.group == g) & (CHK.draw == -1)].iloc[0]
        nsp = int(ret.loc[g, "species_retained_full"]) if an == "filtered" else int(c1m.loc[g, "n_taxa_present"]); assert nsp == c.n_species_in_analysis
        if an == "filtered": assert (ret.loc[g, "nodes_full"], ret.loc[g, "edges_full"]) == (c.nodes, c.edges)
        else: assert (c1m.loc[g, "nodes"], c1m.loc[g, "edges"]) == (c.nodes, c.edges) and abs(c1m.loc[g, "modularity"] - c.modularity) < 1e-9 and abs(c1m.loc[g, "density"] - c.density) < 1e-12
        r = dict(block=block, input=inp, group=g, n_participants=NEXP[g], species_definition="retained by the filter" if an == "filtered" else "present (count > 0 in >= 1 sample)", n_species=nsp,
                 min_prevalence_percent_at_full_n=round(100 * 3 / NEXP[g], 1) if an == "filtered" else np.nan,
                 nodes=int(c.nodes), edges=int(c.edges), density=c.density, mean_degree=c.mean_degree, modularity=c.modularity, cross_phylum_fraction=c.cross_phylum_frac)
        for s, nm in zip(["modularity", "cross_phylum_frac", "edges"], ["modularity", "cross_phylum", "edges"]):
            if g == "CDM":
                r.update({f"sub14_{nm}_median": np.nan, f"sub14_{nm}_p2.5": np.nan, f"sub14_{nm}_p97.5": np.nan, f"CDM_percentile_in_group_{nm}": np.nan}); continue
            x = R[(R.analysis == an) & (R.group == g) & (R.draw >= 0)][s].values; d = DIST[an].loc[(s, g)]; assert len(x) == 100 == d.n_replicates
            v = dict(median=np.median(x), lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5), pc=pct_mid(x, cdm[s]))
            tol = 1e-12 * max(1, abs(d["p97.5"]))
            assert abs(v["median"] - d.p50) <= tol and abs(v["lo"] - d["p2.5"]) <= tol and abs(v["hi"] - d["p97.5"]) <= tol and np.isclose(v["pc"], d.CDM_percentile_midrank, rtol=1e-12) and np.isclose(d.CDM_observed, cdm[s], rtol=1e-12) and np.isclose(d.observed_full_n, c[s], rtol=1e-12)   # csv round trip: last bit
            r.update({f"sub14_{nm}_median": v["median"], f"sub14_{nm}_p2.5": v["lo"], f"sub14_{nm}_p97.5": v["hi"], f"CDM_percentile_in_group_{nm}": v["pc"]})
        r.update(sub14_replicates=100 if g != "CDM" else np.nan, published_nodes=np.nan, published_edges_double_counted=np.nan, published_mean_degree_double_counted=np.nan, published_density_double_counted=np.nan,
                 note="single network (n = 14); own sampling variability not captured" if g == "CDM" else ""); rows.append(r)
for g in GROUPS:                                               # block 3: the ORIGINAL published analysis (ConQuR 346-taxon input) and its correction
    o = orec.loc[g]; sv = pd.read_csv(F_OSUM[g]).iloc[0]
    assert (sv.Nodes, sv.Edges, sv.Avg_Degree, sv.Density) == (o.saved_Nodes, o.saved_Edges_doubled, o.saved_Avg_Degree, o.saved_Density) and o.metric_code_reproduces_saved
    assert sv.Edges == 2 * o.edges and sv.Nodes == o.nodes and round(2 * o.mean_degree, 2) == sv.Avg_Degree and round(2 * o.density, 4) == sv.Density and o.n_samples == NEXP[g]
    rows.append(dict(block="3 published values and their correction (I-A)", input="ConQuR-corrected table of 346 taxa (original analysis, 2025-07); no prevalence filter; no sub-sampling was run on this input",
                     group=g, n_participants=NEXP[g], species_definition="taxa in the table (present in the group: %d)" % o.n_taxa_present, n_species=int(o.n_taxa_input), nodes=int(o.nodes), edges=int(o.edges),
                     density=o.density, mean_degree=o.mean_degree, modularity=o.modularity, cross_phylum_fraction=o.cross_phylum_frac,
                     published_nodes=int(sv.Nodes), published_edges_double_counted=int(sv.Edges), published_mean_degree_double_counted=sv.Avg_Degree, published_density_double_counted=sv.Density,
                     note="modularity and cross-phylum fraction: no original value or code exists; values re-computed here with the definitions of this table (not published values)"))
S3 = pd.DataFrame(rows); S3.to_csv(fresh(f"{T}/TableS3_sparcc_networks.csv"), index=False)
m = S3[S3.block.str.startswith("1")].set_index("group"); u = S3[S3.block.str.startswith("2")].set_index("group"); a = S3[S3.block.str.startswith("3")].set_index("group")
# facts given in the task, verified here (a failure stops the script; nothing is adjusted to make them pass)
FACT = {"I-C nodes": (list(m.nodes), [320, 378, 161, 93]), "I-C edges": (list(m.edges), [498, 992, 391, 366]), "I-C modularity": ([round(v, 3) for v in m.modularity], [0.778, 0.708, 0.653, 0.467]),
        "I-C cross-phylum": ([round(v, 3) for v in m.cross_phylum_fraction], [0.187, 0.211, 0.340, 0.432]),
        "I-C CDM percentile modularity": (list(m.CDM_percentile_in_group_modularity[:3]), [93, 67, 47]), "I-C CDM percentile cross": (list(m.CDM_percentile_in_group_cross_phylum[:3]), [98, 85, 13]),
        "I-B edges": (list(u.edges), [2665, 5861, 3995, 6336]), "I-B CDM percentile modularity": (list(u.CDM_percentile_in_group_modularity[:3]), [29, 56, 24]),
        "I-B CDM percentile cross": (list(u.CDM_percentile_in_group_cross_phylum[:3]), [100, 100, 74]), "I-B CDM modularity / cross": ([round(u.modularity["CDM"], 3), round(u.cross_phylum_fraction["CDM"], 3)], [0.599, 0.509]),
        "I-A published edges": (list(a.published_edges_double_counted), [574, 904, 526, 1320]), "I-A edges once": (list(a.edges), [287, 452, 263, 660])}
for k, (got, exp) in FACT.items():
    log(f"fact check {k}: {got} {'==' if got == exp else '!='} expected {exp}"); assert got == exp, k
nc_hi = m.loc["NC", "sub14_cross_phylum_p97.5"]; margin = m.cross_phylum_fraction["CDM"] - nc_hi
x5 = pd.read_csv(F_X5); alt = x5[(x5.id == "prev3W__CDM__full") & (x5.seed_kind != "same")]; assert len(alt) == 10
sd_cross = alt.cross_phylum_frac.std(ddof=1); n_alt_le = int((alt.cross_phylum_frac <= nc_hi).sum()); sd_mod = alt.modularity.std(ddof=1)
log(f"cross-phylum: CDM {m.cross_phylum_fraction['CDM']:.6f} - NC 97.5th percentile {nc_hi:.6f} = {margin:.6f}; run-to-run SD of the CDM value over 10 other seeds {sd_cross:.6f} (range {alt.cross_phylum_frac.min():.4f}-{alt.cross_phylum_frac.max():.4f}); "
    f"{n_alt_le} of 10 re-runs at or below the NC bound; modularity SD {sd_mod:.4f}")
assert f"{margin:.3f}" == "0.006" and f"{sd_cross:.3f}" == "0.007" and margin < sd_cross
