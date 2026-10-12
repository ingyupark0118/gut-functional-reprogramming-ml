#!/usr/bin/env python
"""M6 step 7 (parameterised version of CDM-4/scripts/m6_07_tables.py): reference-bias audit + tables
T_M6_<S>_{steps,quality,refbias,novel_regions}.csv, same columns as CDM-4. Run from the sample folder: m6_07_tables.py <sample>.
Differences from CDM-4: nothing hard-coded (rRNA/tRNA/MIMAG computed from barrnap/ARAGORN output, DESIGN section 6);
does NOT rewrite ../T_M6_plan_remaining.csv; skani rows also list the other M6 targeted assemblies finished so far."""
import csv, collections, statistics, re, sys, os
AA = set('Ala Arg Asn Asp Cys Gln Glu Gly His Ile Leu Lys Met Phe Pro Ser Thr Trp Tyr Val'.split())
def rrna_summary(gff):
    c = collections.Counter()
    for l in open(gff):
        m = re.search(r'product=(5S|16S|23S) ribosomal RNA( \(partial\))?', l)
        if m: c[(m.group(1), 'partial' if m.group(2) else 'full')] += 1
    return c
def trna_summary(txt):
    t = re.findall(r'\btRNA-([A-Za-z]{3})\b', open(txt).read())
    return len(t), len(set(t) & AA)
def mimag(comp, cont, rr, ntypes):
    """DESIGN section 6: >90/<5 = high numeric level, strict high also needs full 5S+16S+23S and >=18 tRNA types; >=50/<10 medium; else low."""
    if comp > 90 and cont < 5:
        miss = [g for g in ('5S', '16S', '23S') if rr[(g, 'full')] == 0] + (['tRNA types<18'] if ntypes < 18 else [])
        return 'high (strict: full 5S+16S+23S, >=18 tRNA types)' if not miss else 'medium (completeness/contamination at high-quality level, but strict criterion not met: no full ' + ', '.join(miss) + ')'
    return 'medium' if comp >= 50 and cont < 10 else 'low'
_r = collections.Counter({('5S', 'full'): 1, ('16S', 'full'): 1, ('23S', 'partial'): 2})
assert mimag(99, 0.1, _r, 20).startswith('medium (') and mimag(99, 0.1, collections.Counter({(g, 'full'): 1 for g in ('5S', '16S', '23S')}), 18).startswith('high')
assert mimag(60, 9, _r, 5) == 'medium' and mimag(49, 0, _r, 20) == 'low' and mimag(95, 10, _r, 20) == 'low'
if __name__ == '__main__':
    S = sys.argv[1]; W = 'work/'
    C = list(csv.DictReader(open(f'T_M6_{S}_contigs.csv')))
    keep = {r['contig']: int(float(r['length'])) for r in C if r['decision'] == 'KEEP'}
    T = sum(keep.values())
    def ulen(x):
        x = sorted(x); t = e0 = 0
        for s, e in x:
            if e > e0: t += e - max(s - 1, e0); e0 = e
        return t
    def uncovered(L, x, minlen=1):
        x = sorted(x); gaps = []; e0 = 0
        for s, e in x:
            if s - 1 > e0: gaps.append((e0 + 1, s - 1))
            e0 = max(e0, e)
        if e0 < L: gaps.append((e0 + 1, L))
        return [g for g in gaps if g[1] - g[0] + 1 >= minlen]
    G = collections.defaultdict(lambda: collections.defaultdict(list))   # genome -> contig -> intervals
    for l in open(W + 'blast_r2.tsv'):
        p = l.split('\t')
        if p[0] not in keep: continue
        g = '__'.join(p[1].split('__')[:2]); a, b = sorted((int(p[4]), int(p[5]))); G[g][p[0]].append((a, b))
    def frac(gs): return sum(ulen([iv for g in gs for iv in G[g].get(c, [])]) for c in keep) / T
    pub = [g for g in G if g.startswith('LC__GC')]; mags = [g for g in G if g.startswith('LC__MAG')]
    single = {g: frac([g]) for g in pub}
    best = max(single, key=single.get); worst = min(single, key=single.get)
    f_m247 = single.get('LC__GCF_048541565.1', 0.0); f_pub = frac(pub); f_all = frac(pub + mags)
    r1 = collections.defaultdict(list)   # iterative gain: final bp not present (id>=95, len>=200) in round-1 contigs
    for l in open(W + 'blast_final_vs_r1.tsv'):
        p = l.split('\t'); a, b = sorted((int(p[4]), int(p[5]))); r1[p[0]].append((a, b))
    name2orig = {l[1:].split()[0]: re.search(r'orig=(\S+)', l).group(1) for l in open(f'{S}_Lcrispatus_targeted.fa') if l[0] == '>'}
    gain_bp = gain_nopub = 0
    for n, o in name2orig.items():
        gaps = uncovered(keep[o], r1.get(n, [])); gain_bp += sum(e - s + 1 for s, e in gaps)
        pubiv = sorted(iv for g in pub for iv in G[g].get(o, []))
        for s, e in gaps: gain_nopub += (e - s + 1) - ulen([(max(a, s), min(b, e)) for a, b in pubiv if b >= s and a <= e])
    nov = [(c, s, e) for c in keep for s, e in uncovered(keep[c], [iv for g in pub for iv in G[g].get(c, [])], 500)]
    novall = [(c, s, e) for c in keep for s, e in uncovered(keep[c], [iv for g in pub + mags for iv in G[g].get(c, [])], 500)]
    with open(f'T_M6_{S}_novel_regions.csv', 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['contig_orig', 'start', 'end', 'length', 'also_absent_from_patient_MAGs'])
        for c, s, e in sorted(nov, key=lambda x: -(x[2] - x[1])): w.writerow([c, s, e, e - s + 1, any(c == a and s <= d and e >= b for a, b, d in novall)])
    r1c = list(csv.DictReader(open(W + 'contigs_r1.csv'))); r2c = list(csv.DictReader(open(W + 'contigs_r2.csv')))
    unp = [r for r in C if r['decision'] == 'UNPLACED']
    sk = {r['Ref_file'].split('/')[-1].replace('.fna', '').replace('.fa', ''): r for r in csv.DictReader(open(W + 'skani_final_vs_refs.tsv'), delimiter='\t')}
    bp = lambda rows: sum(int(float(r['length'])) for r in rows)
    rb = [('assembly_bp_retained', T, f'{S}_Lcrispatus_targeted.fa, {len(keep)} contigs'),
     ('frac_matched_M247_Siena_GCF_048541565.1', round(f_m247, 4), 'blastn megablast id>=90 len>=200, union of HSPs / assembly bp'),
     ('frac_NOT_in_M247_Siena', round(1 - f_m247, 4), f'{round((1-f_m247)*T)} bp = what single-reference (M247) recruitment could not have templated'),
     ('best_single_public_reference', best.replace('LC__', ''), f'matched fraction {single[best]:.4f}'),
     ('frac_NOT_in_best_single_public_reference', round(1 - single[best], 4), f'{round((1-single[best])*T)} bp'),
     ('worst_single_public_reference', worst.replace('LC__', ''), f'matched fraction {single[worst]:.4f}'),
     ('median_single_public_reference_matched', round(statistics.median(single.values()), 4), f'n={len(single)} public genomes with >=1 hit (panel has 122)'),
     ('frac_matched_public_panel_122', round(f_pub, 4), ''),
     ('frac_NOT_in_public_panel', round(1 - f_pub, 4), f'{round((1-f_pub)*T)} bp; regions>=500bp: n={len(nov)}, {sum(e-s+1 for c,s,e in nov)} bp (T_M6_{S}_novel_regions.csv)'),
     ('frac_matched_public_panel_plus_2_patient_MAGs', round(f_all, 4), 'partly circular for CDM-1/CDM-3: their own MAG is in the panel (DESIGN section 5)'),
     ('frac_NOT_in_any_panel_genome', round(1 - f_all, 4), f'{round((1-f_all)*T)} bp; regions>=500bp: n={len(novall)}, {sum(e-s+1 for c,s,e in novall)} bp'),
     ('frac_matched_MAG_CDM1', round(frac(['LC__MAG_CDM1_CDM-1bin4']), 4), ''), ('frac_matched_MAG_CDM3', round(frac(['LC__MAG_CDM3_CDM3bin9']), 4), ''),
     ('skani_AF_query_vs_M247_Siena', sk.get('GCF_048541565.1', {}).get('Align_fraction_query', 'NA'), 'skani aligned fraction of the assembly (independent of blastn)'),
     ('round1_unfiltered_bp', bp(r1c), f'{len(r1c)} contigs'), ('round2_unfiltered_bp', bp(r2c), f'{len(r2c)} contigs'),
     ('iterative_gain_bp_in_retained_assembly', gain_bp, 'retained-assembly bp with no match (id>=95,len>=200) in round-1 contigs'),
     ('iterative_gain_bp_without_public_panel_match', gain_nopub, 'subset of the gain not matched by any public reference'),
     ('unplaced_contigs_not_in_genome', len(unp), f'{bp(unp)} bp, depth>=0.6x reference depth, <50% panel match, <50% decoy/other-bin match: taxonomy unresolved with on-server evidence (no remote BLAST)'),
     ('NOT_QUANTIFIED', 'strain-specific islands with no >=50-bp overlap to panel/round-1 contigs', f'cannot be recruited by design; needs untargeted assembly of {S}')]
    with open(f'T_M6_{S}_refbias.csv', 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['metric', 'value', 'note']); w.writerows(rb)
    # ---- quality
    ck = {r['Bin Id']: r for r in csv.DictReader(open(W + 'checkm_qa.tsv'), delimiter='\t')}
    snv = [l.split('\t') for l in open(W + 'snv_density.tsv')]; tot = [x for x in snv if x[0] == 'TOTAL'][0]
    core = [(float(r['depth']), float(r['length'])) for r in C if r['reason'] == 'core']; cd = [d for d, l in core]; big = [d for d, l in core if l >= 10000]
    cv = lambda x: round(statistics.pstdev(x) / statistics.mean(x), 3) if x else 'NA'
    seq = ''.join(l.strip() for l in open(f'{S}_Lcrispatus_targeted.fa') if l[0] != '>')
    lens = sorted(keep.values(), reverse=True); c = 0
    for x in lens:
        c += x
        if c >= T / 2: n50 = x; break
    q = []
    for b, lab in [(f'{S}_Lcrispatus_targeted', 'final (filtered)'), (f'{S}_round2_unfiltered', 'round-2 unfiltered'), (f'{S}_round1_unfiltered', 'round-1 unfiltered')]:
        r = ck[b]; comp, cont = float(r['Completeness']), float(r['Contamination'])
        q += [(lab, 'checkm_completeness', comp), (lab, 'checkm_contamination', cont), (lab, 'checkm_strain_heterogeneity', r['Strain heterogeneity']), (lab, 'quality_score_comp_minus_5cont', round(comp - 5 * cont, 2)), (lab, 'checkm_marker_lineage', r['Marker lineage'])]
    F = 'final (filtered)'; comp, cont = q[0][2], q[1][2]
    rr = rrna_summary(W + 'rrna.gff'); ntr, ntypes = trna_summary(W + 'trna.txt')
    q += [(F, 'size_bp', T), (F, 'contigs', len(keep)), (F, 'N50_bp', n50), (F, 'longest_contig_bp', lens[0]), (F, 'GC_pct', round(100 * (seq.count('G') + seq.count('C')) / len(seq), 2)),
     (F, 'prodigal_genes', open(W + 'final_gene_count.txt').read().strip()),
     (F, 'rRNA_barrnap', '; '.join(f'{g} {rr[(g,"full")]} full, {rr[(g,"partial")]} partial' for g in ('16S', '5S', '23S'))), (F, 'tRNA_aragorn', f'{ntr} genes, {ntypes} amino-acid types'),
     (F, 'manuscript_criterion_QS_gt_50', 'PASS' if comp - 5 * cont > 50 else 'FAIL'), (F, 'MIMAG', mimag(comp, cont, rr, ntypes)),
     (F, 'reference_depth_panel_matching_contigs', re.search(r'contigs\) (\S+)', open('logs/m6_05_filter.out').read()).group(1)),
     (F, 'core_contig_depth_median', round(statistics.median(cd), 1) if cd else 'NA'), (F, 'core_contig_depth_CV_all', cv(cd)),
     (F, 'core_contig_depth_CV_contigs_ge10kb', cv(big)), (F, 'core_contig_depth_range_ge10kb', f'{min(big):.1f}-{max(big):.1f}' if big else 'NA (no core contig >=10 kb)'),
     (F, 'polymorphic_sites', tot[2]), (F, 'sites_evaluated_depth_ge20', tot[1]),
     (F, 'polymorphic_sites_per_kb', tot[3].strip() if int(tot[1]) > 0 else 'not evaluable (no site with depth >= 20)'),
     (F, 'frac_assembly_bp_evaluated_for_SNV', round(int(tot[1]) / T, 4)),
     (F, 'minor_allele_freq_bins_0.1-0.2/0.2-0.3/0.3-0.4/0.4-0.5', '/'.join(x.strip() for x in snv[-1][1:]))]
    pubk = [k for k in sk if k.startswith('GC') and 'genomic' not in k]; bestp = max(pubk, key=lambda k: float(sk[k]['ANI'])) if pubk else None
    lab = [(bestp, f'best public by ANI ({bestp})')] if bestp else []
    lab += [('GCF_048541565.1', 'M247_Siena (U2_G reference)'), ('GCF_002088015.1', 'type strain ATCC 33820 (GCF_002088015.1)'), ('GCF_018987235.1', 'type strain ATCC 33820 complete (GCF_018987235.1)'),
            ('MAG_CDM1_CDM-1bin4', 'patient MAG CDM-1 (CDM-1 bin.4)'), ('MAG_CDM3_CDM3bin9', 'patient MAG CDM-3 (CDM-3 bin.9)'), ('GCF_000014425.1_ASM1442v1_genomic', 'L. gasseri ATCC 33323 (decoy)')]
    lab += [(f'{x}_Lcrispatus_targeted', f'M6 targeted assembly {x}') for x in ('CDM-4', 'Cirrhosis-1', 'CDM-3', 'CDM-2', 'CDM-1') if x != S]
    for k, l in lab:
        r = sk.get(k)   # absent = not finished yet when this sample ran (see ../T_M6_skani_allvsall.tsv) or no skani alignment
        q.append((F, f'skani_ANI | AF_assembly | AF_other :: {l}', f"{r['ANI']} | {r['Align_fraction_query']} | {r['Align_fraction_ref']}" if r else 'NA (not in this run: not yet assembled, or no alignment)'))
    with open(f'T_M6_{S}_quality.csv', 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['assembly', 'metric', 'value']); w.writerows(q)
    # ---- steps
    def sec(t):
        p = [float(x) for x in t.split(':')]; return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]
    st = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open('logs/steps_timing.tsv')}
    def kv(f): return dict(l.strip().split('=', 1) for l in open(f) if '=' in l and not l.startswith('hits_file'))
    a = kv('logs/r1_recruit_stats.txt'); b = kv('logs/r2_recruit_stats.txt'); bait = kv('logs/r2_bait_stats.txt')
    r2lc = re.search(r'r2_hits.tsv\tLC_best=(\d+)\tDC_best=(\d+)', open('logs/r2_recruit_stats.txt').read()).groups()
    def asm(rows):
        L = sorted((int(float(r['length'])) for r in rows), reverse=True); c = 0
        for x in L:
            c += x
            if c >= sum(L) / 2: return f'{len(L):,} contigs, {sum(L):,} bp, N50 {x:,}'
    ev = open('logs/ev_bt2.log').read(); al = sum(int(x) for x in re.findall(r'(\d+) \([\d.]+%\) aligned (?:exactly 1|>1) time', ev)); rate = re.search(r'([\d.]+)% overall', ev).group(1)
    D = [('inputs', 'sha256 + read-structure scan: ' + open('logs/input_read_structure.txt').read().strip(), a['total_reads']),
     ('r1_map', 'round 1 (shared panel+decoy index, reused): reads best-aligned to L. crispatus panel (decoy-best: %s)' % a['DC_reads_union'], a['LC_reads_union']),
     ('r1_recruit', 'round 1 recruited = direct %s + mate rescue %s (dropped decoy-best mates %s)' % (a['recruited_direct'], a['recruited_mate_rescue'], a['fragment_LC_but_read_decoy_dropped']), a['recruited']),
     ('r1_megahit', 'MEGAHIT v1.2.9 round 1: ' + asm(r1c), a['recruited']),
     ('r2_bait', 'bait = %s round-1 contigs (%s bp) with multi>=0.3x median (%s) + decoys; index' % (bait['bait_contigs'], bait['bait_bp'], bait['r1_lenweighted_median_multi']), ''),
     ('r2_map', 'round 2: ALL reads vs bait+decoy; LC-best %s, decoy-best %s' % r2lc, r2lc[0]),
     ('r2_recruit', 'round 2 recruited (union) = direct %s + mate rescue %s; newly added vs round 1 = %d' % (b['recruited_direct'], b['recruited_mate_rescue'], int(b['recruited']) - int(a['recruited'])), b['recruited']),
     ('r2_megahit', 'MEGAHIT v1.2.9 round 2 (final): ' + asm(r2c), b['recruited']),
     ('ev_map', f'recruited reads -> final contigs (end-to-end); aligned {al:,} ({rate} %)', al),
     ('ev_blast', 'blastn contigs vs panel+decoy (round 1 and 2; shared DB, local)', ''),
     ('ob_blast', 'blastn round-2 contigs vs other bins of the same sample (local): %d bin files' % sum(1 for l in open('logs/otherbins_list.txt') if l.strip()), ''),
     ('ctab', 'contig tables', ''),
     ('filter', f'PRE-FIXED contig filter -> {len(keep)} contigs, {T:,} bp retained; unplaced {len(unp)} ({bp(unp):,} bp)', ''),
     ('checkm', 'CheckM v1.2.4 lineage_wf, 3 assemblies in one run', ''),
     ('skani', f'skani 0.3.2 dist vs {len(sk)} genomes with a reported alignment', ''), ('prodigal', 'Prodigal gene calling', ''),
     ('rrna', 'barrnap 0.9 + ARAGORN', ''), ('snv', 'samtools mpileup SNV density', ''), ('gain_blast', 'final vs round-1 contigs', '')]
    rows = [(n, d, r, sec(st[n][1]), *st[n][2:]) for n, d, r in D if n in st]
    with open(f'T_M6_{S}_steps.csv', 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['step', 'description', 'reads', 'wall_s', 'peak_rss_kb', 'load1_start', 'load1_end']); w.writerows(rows)
        w.writerow(['TOTAL', 'sum of steps (panel index reused, not rebuilt; guard waiting time not included)', '', round(sum(float(r[3]) for r in rows)), max(int(r[4]) for r in rows if r[4]), '', ''])
    for r in rb + q: print(r)
