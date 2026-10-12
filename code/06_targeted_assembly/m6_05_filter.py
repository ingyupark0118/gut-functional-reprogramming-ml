#!/usr/bin/env python
"""M6 step 5, PRE-FIXED contig filter (DESIGN_prefixed_remaining.md section 3) -> <S>_Lcrispatus_targeted.fa
(+ unplaced / excluded FASTA, per-contig decision table). Parameterised copy of CDM-4/scripts/m6_05_filter.py.
usage (from the sample folder): m6_05_filter.py <sample>
  ref_depth   = length-weighted median depth of round-2 contigs with f_LC_all >= 0.5 (panel-matching contigs)
  depth_ratio = depth / ref_depth
  1 depth_ratio < 0.6  -> EXCLUDED low_depth
  2 f_decoy >= 0.5     -> EXCLUDED decoy_species
  3 f_otherbins >= 0.5 -> EXCLUDED other_bin_same_sample
  4 f_LC_all >= 0.5    -> KEEP (high_copy if depth_ratio > 1.6, else core)
  5 otherwise          -> UNPLACED no_panel_match_no_local_hit (separate file, NOT part of the genome)
No remote BLAST: local evidence only. Thresholds are fixed and must not be changed after seeing results."""
import csv, collections, sys
MIN_RATIO, MIN_PANEL, MAX_OTHER, HIGH_COPY = 0.6, 0.5, 0.5, 1.6
def decide(ratio, f_lc, f_dc, f_ob):
    if ratio < MIN_RATIO: return 'EXCLUDED', 'low_depth'
    if f_dc >= MAX_OTHER: return 'EXCLUDED', 'decoy_species'
    if f_ob >= MAX_OTHER: return 'EXCLUDED', 'other_bin_same_sample'
    if f_lc >= MIN_PANEL: return 'KEEP', 'high_copy' if ratio > HIGH_COPY else 'core'
    return 'UNPLACED', 'no_panel_match_no_local_hit'
# self-check of the rule (runs on every call)
assert decide(0.59, 1, 0, 0) == ('EXCLUDED', 'low_depth') and decide(0.6, 0.5, 0.49, 0.49) == ('KEEP', 'core')
assert decide(1.61, 0.9, 0, 0) == ('KEEP', 'high_copy') and decide(1.6, 0.9, 0, 0) == ('KEEP', 'core')
assert decide(1, 0.9, 0.5, 0) == ('EXCLUDED', 'decoy_species') and decide(1, 0.9, 0, 0.5) == ('EXCLUDED', 'other_bin_same_sample')
assert decide(1, 0.49, 0, 0) == ('UNPLACED', 'no_panel_match_no_local_hit') and decide(1, 0.49, 0.6, 0)[0] == 'EXCLUDED'
if __name__ == '__main__':
    S = sys.argv[1]; W = 'work/'
    rows = list(csv.DictReader(open(W + 'contigs_r2.csv')))
    for r in rows:
        for k in r:
            try: r[k] = float(r[k])
            except ValueError: pass
    iv = collections.defaultdict(lambda: collections.defaultdict(list))   # contig -> other bin -> intervals
    for l in open(W + 'blast_r2_otherbins.tsv'):
        p = l.split('\t'); a, b = sorted((int(p[4]), int(p[5]))); iv[p[0]][p[1].split('__')[1]].append((a, b))
    def ulen(x):
        x = sorted(x); t = e0 = 0
        for s, e in x:
            if e > e0: t += e - max(s - 1, e0); e0 = e
        return t
    pm = sorted((r for r in rows if r['f_LC_all'] >= MIN_PANEL), key=lambda r: r['depth'])
    if not pm: sys.exit(f'{S}: no panel-matching contig (f_LC_all >= {MIN_PANEL}) among {len(rows)} round-2 contigs - filter cannot be applied')
    Tp = sum(r['length'] for r in pm); c = 0
    for r in pm:
        c += r['length']
        if c >= Tp / 2: med = r['depth']; break
    if med <= 0: sys.exit(f'{S}: reference depth is 0')
    T = sum(r['length'] for r in rows); n = 0
    for r in sorted(rows, key=lambda r: -r['length']):
        G = iv.get(r['contig'], {})
        r['f_otherbins'] = round(ulen([x for g in G for x in G[g]]) / r['length'], 4)
        r['depth_ratio'] = round(r['depth'] / med, 3); r['remote_nt_top_hit'] = ''   # column kept for format compatibility; remote BLAST not used
        r['decision'], r['reason'] = decide(r['depth_ratio'], r['f_LC_all'], r['f_decoy'], r['f_otherbins'])
        if r['decision'] == 'KEEP': n += 1; r['new_name'] = f'{S}_Lcrispatus_targeted_c{n:04d}'
        else: r['new_name'] = ''
        r['best_otherbin'] = max(G, key=lambda g: ulen(G[g]), default='')
    if n == 0: sys.exit(f'{S}: 0 contigs kept by the pre-fixed filter')
    rows.sort(key=lambda r: -r['length'])
    with open(f'T_M6_{S}_contigs.csv', 'x', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    dec = {r['contig']: r for r in rows}
    outs = {'KEEP': open(f'{S}_Lcrispatus_targeted.fa', 'x'), 'UNPLACED': open(f'{S}_targeted_unplaced_NOT_genome.fa', 'x'), 'EXCLUDED': open(f'{S}_targeted_excluded_NOT_genome.fa', 'x')}
    for l in open(W + 'asm_r2/final.contigs.fa'):
        if l[0] == '>':
            r = dec[l[1:].split()[0]]; fo = outs[r['decision']]
            fo.write(f">{r['new_name']} targeted_assembly orig={r['contig']} depth={r['depth']:.1f} flag={r['reason']}\n" if r['decision'] == 'KEEP' else f">{r['contig']} {r['decision']} {r['reason']}\n")
        else: fo.write(l)
    for fo in outs.values(): fo.close()
    print('reference depth (length-weighted median of panel-matching contigs)', med, '| panel-matching contigs', len(pm), 'bp', int(Tp))
    cnt = collections.Counter(); bp = collections.Counter()
    for r in rows: cnt[(r['decision'], r['reason'])] += 1; bp[(r['decision'], r['reason'])] += int(r['length'])
    for k in sorted(cnt): print(k, cnt[k], bp[k])
    assert sum(bp.values()) == T
