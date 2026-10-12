#!/usr/bin/env python
"""E4 step 10: single-copy core genes -> per-gene alignment -> QC (DESIGN_E4_prefixed.md s2-5) -> concatenation. Logic copied from refs/rescue/F/F_10_gene_qc.py (22 genomes -> N).
Usage: nice -n 19 ionice -c3 python E4_10_gene_qc.py <main|sens>   (1 thread)
Outputs (suffix _<set>): T_E4_gene_qc, T_E4_qc_offenders, T_E4_alignment_summary, T_E4_per_gene_nearest (csv); E4_concat_<set>.fasta, _partitions.nex, _gene_coords.csv; work/genes_<set>/*.aln; work/arrays_<set>.npz"""
import csv, hashlib, itertools, os, re, subprocess, sys, time
import numpy as np
SET = sys.argv[1]; assert SET in ('main', 'sens')
REV = '.'; E = f'{REV}/work_a/E4_fig7B'; VP = f'{REV}/refs/vp_reanalysis'; RO = f'{VP}/{SET}/roary'
ANN = '../03_annotation/CDM'
MAFFT = 'mafft'
MIN_LEN, MIN_COV, MIN_ID, MAX_AMBIG = 300, 0.80, 0.90, 0.01
REN = {'CDM-1_bin.1': 'CDM-1_bin.1', 'CDM-4_bin.2': 'CDM-4_bin.2', 'CDM-3_bin.12': 'CDM-3_bin.12'}; MAGS = list(REN.values())
STOPS = {'TAA', 'TAG', 'TGA'}; t0 = time.time()
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def fasta(p):
    d, k = {}, None
    for l in open(p):
        if l[0] == '>': k = l[1:].split()[0]; d[k] = []
        else: d[k].append(l.strip())
    return {k: ''.join(v).upper() for k, v in d.items()}
os.makedirs(f'{E}/work/genes_{SET}', exist_ok=True)
for out in [f'T_E4_gene_qc_{SET}.csv', f'E4_concat_{SET}.fasta']: assert not os.path.exists(f'{E}/{out}'), f'{out} exists - refusing to overwrite'
print('### E4_10_gene_qc', SET, time.strftime('%F %T'), 'load', open('/proc/loadavg').read().split()[0])
print('python', sys.version.split()[0], 'numpy', np.__version__, '|', subprocess.run([MAFFT, '--version'], capture_output=True, text=True).stderr.strip())
gpa, core_aln, embl = f'{RO}/gene_presence_absence.csv', f'{RO}/core_gene_alignment.aln', f'{RO}/core_alignment_header.embl'
for p in [gpa, core_aln, embl, f'{E}/DESIGN_E4_prefixed.md', os.path.abspath(__file__)]: print('sha256', sha(p), p)
rows = list(csv.reader(open(gpa))); hdr, rows = rows[0], rows[1:]
cols = hdr[14:]; n = len(cols); assert n == (14 if SET == 'main' else 30) and set(REN) <= set(cols)
genomes = [REN.get(c, c) for c in cols]
ffn = {}
for c in cols:
    p = f'{ANN}/V.parvula_CDM{c[4:]}/{c}.ffn' if c in REN else f'{VP}/prokka/{c}/{c}.ffn'; print('sha256', sha(p), p); ffn.update(fasta(p))
core = {REN.get(k, k): v for k, v in fasta(core_aln).items()}; assert set(core) == set(genomes)
coords = {m.group(3): (int(m.group(1)) - 1, int(m.group(2))) for m in re.finditer(r'feature\s+(\d+)\.\.(\d+)\nFT\s+/label=(\S+)', open(embl).read())}
print('roary core alignment: length', len(next(iter(core.values()))), 'genes in embl header', len(coords))
nall = [r for r in rows if r[3] == str(n)]
sc = [r for r in nall if r[4] == str(n) and all(c and not re.search(r'[\t; ]', c) for c in r[14:])]
print(f'clusters total {len(rows)}; in {n}/{n} genomes {len(nall)}; single-copy core {len(sc)}; {n}/{n} but multi-copy {len(nall) - len(sc)}')
ACGT = np.frombuffer(b'ACGT', dtype='S1'); PAIRS = list(itertools.combinations(range(n), 2)); off = ~np.eye(n, dtype=bool)
qc, nearest, offend, passing, Dm, Nm, tagrows = [], [], [], [], [], [], []
n_roary = n_mafft = 0
for r in sorted(sc, key=lambda r: r[0]):
    name = r[0]; tags = dict(zip(genomes, r[14:])); raw = {g: ffn[tags[g]] for g in genomes}; aln = None; route = 'mafft_realigned(not in roary core aln)'
    if name in coords:
        a, b = coords[name]; sl = {g: core[g][a:b] for g in genomes}
        if all(sl[g].replace('-', '') == raw[g] for g in genomes): route, aln = 'roary_mafft_slice', sl
        else: route = 'mafft_realigned(slice!=ffn)'
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    if aln is None:
        tmp = f'{E}/work/genes_{SET}/{safe}.unaligned.fa'; open(tmp, 'w').write(''.join(f'>{g}\n{raw[g]}\n' for g in genomes))
        o = subprocess.run(['nice', '-n', '19', MAFFT, '--auto', '--quiet', '--thread', '1', tmp], capture_output=True, text=True, check=True).stdout
        d, k = {}, None
        for l in o.splitlines():
            if l[0] == '>': k = l[1:].strip(); d[k] = []
            else: d[k].append(l.strip())
        aln = {k: ''.join(v).upper() for k, v in d.items()}; n_mafft += 1
        assert all(aln[g].replace('-', '') == raw[g] for g in genomes)
    else: n_roary += 1
    open(f'{E}/work/genes_{SET}/{safe}.aln', 'w').write(''.join(f'>{g}\n{aln[g]}\n' for g in genomes))
    M = np.array([np.frombuffer(aln[g].encode(), dtype='S1') for g in genomes]); L = M.shape[1]
    valid = np.isin(M, ACGT); cov = (M != b'-').sum(1) / L
    V = valid.astype(np.float64); Nmat = V @ V.T; match = sum(((M == b) .astype(np.float64)) @ ((M == b).astype(np.float64)).T for b in ACGT)
    with np.errstate(invalid='ignore', divide='ignore'): ident = np.where(Nmat > 0, match / Nmat, np.nan)
    np.fill_diagonal(ident, 1.0)
    minid_per = np.array([np.nanmin(ident[i][off[i]]) if not np.isnan(ident[i][off[i]]).all() else np.nan for i in range(n)])
    medid_per = np.array([np.nanmedian(ident[i][off[i]]) if not np.isnan(ident[i][off[i]]).all() else np.nan for i in range(n)])
    codon_bad = [g for g in genomes if len(raw[g]) % 3 or any(raw[g][k:k + 3] in STOPS for k in range(0, len(raw[g]) - 3, 3))]
    ambig_bad = [g for g in genomes if sum(c not in 'ACGT' for c in raw[g]) / len(raw[g]) > MAX_AMBIG]
    fails = {}
    if L < MIN_LEN: fails['Q1_length<300'] = []
    if cov.min() < MIN_COV: fails['Q2_coverage<0.80'] = [g for g, c in zip(genomes, cov) if c < MIN_COV]
    if np.isnan(ident).any() or np.nanmin(ident) < MIN_ID:
        o = [g for g, m in zip(genomes, medid_per) if m < MIN_ID]
        fails['Q3_identity<0.90'] = o if o else ['NO_SINGLE_OFFENDER:' + '|'.join(sorted({genomes[k] for i, j in PAIRS if not ident[i, j] >= MIN_ID for k in (i, j)}))]
    if codon_bad: fails['Q4_codon'] = codon_bad
    if ambig_bad: fails['Q5_ambiguity>1%'] = ambig_bad
    for k, v in fails.items():
        for g in v: offend.append([name, k, g])
    lens = [len(raw[g]) for g in genomes]
    rec = dict(gene=name, annotation=r[2], route=route, aln_length=L, min_ungapped_len=min(lens), max_ungapped_len=max(lens), min_coverage=round(float(cov.min()), 4), min_coverage_genome=genomes[int(cov.argmin())],
               min_pairwise_identity=round(float(np.nanmin(ident)), 4), min_median_identity_genome=genomes[int(np.nanargmin(medid_per))])
    for m in MAGS: rec[f'{m}_min_identity'] = round(float(minid_per[genomes.index(m)]), 4); rec[f'{m}_coverage'] = round(float(cov[genomes.index(m)]), 4)
    rec.update(status='PASS' if not fails else 'FAIL', first_fail=next(iter(fails), ''), fail_reasons=';'.join(fails), offending_genomes=';'.join(f'{k}={",".join(v)}' for k, v in fails.items() if v))
    qc.append(rec)
    if not fails:
        passing.append((name, aln, L)); Dm.append(Nmat - match); Nm.append(Nmat); tagrows.append([name] + [tags[g] for g in genomes])
        for m in MAGS:
            i = genomes.index(m); d = 1 - ident[i]; d[i] = np.inf; mn = d.min(); near = [genomes[j] for j in range(n) if j != i and abs(d[j] - mn) < 1e-12]
            nearest.append(dict(gene=name, mag=m, aln_length=L, min_p_distance=round(float(mn), 6), n_nearest=len(near), nearest_genomes=';'.join(near), other_CDM_among_nearest=';'.join(x for x in near if x in MAGS)))
def wcsv(p, recs):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
wcsv(f'{E}/T_E4_gene_qc_{SET}.csv', qc); wcsv(f'{E}/T_E4_per_gene_nearest_{SET}.csv', nearest)
with open(f'{E}/T_E4_qc_offenders_{SET}.csv', 'w', newline='') as f: csv.writer(f).writerows([['gene', 'rule', 'offending_genome']] + offend)
with open(f'{E}/work/passing_locus_tags_{SET}.csv', 'w', newline='') as f: csv.writer(f).writerows([['gene'] + genomes] + tagrows)
print(f'alignment route: roary slice {n_roary}, mafft re-aligned {n_mafft}'); print(f'QC: pass {len(passing)}, fail {len(qc) - len(passing)}')
RULES = ['Q1_length<300', 'Q2_coverage<0.80', 'Q3_identity<0.90', 'Q4_codon', 'Q5_ambiguity>1%']
for rule in RULES: print(f'  {rule}: any {sum(rule in q["fail_reasons"].split(";") for q in qc)}, first-failing {sum(q["first_fail"] == rule for q in qc)}')
import collections
for rule in RULES:
    c = collections.Counter(g for _, k, g in offend if k == rule and not g.startswith('NO_SINGLE')); ns = sum(1 for _, k, g in offend if k == rule and g.startswith('NO_SINGLE'))
    print(f'  offenders {rule}:', dict(c.most_common()), f'| no single offender: {ns}' if rule.startswith('Q3') else '')
# genes failing ONLY because of one genome (it is the sole offender across all rules)
solo = collections.Counter()
for q in qc:
    if q['status'] == 'FAIL':
        o = {g for _, k, g in [x for x in offend if x[0] == q['gene']] if not g.startswith('NO_SINGLE')}
        if len(o) == 1 and 'Q1_length<300' not in q['fail_reasons'] and 'NO_SINGLE' not in q['offending_genomes']: solo[next(iter(o))] += 1
print('  genes whose only offender is one genome:', dict(solo.most_common()))
cat = {g: ''.join(a[g] for _, a, _ in passing) for g in genomes}
with open(f'{E}/E4_concat_{SET}.fasta', 'w') as f:
    for g in genomes: f.write(f'>{g}\n{cat[g]}\n')
with open(f'{E}/E4_concat_{SET}_partitions.nex', 'w') as f, open(f'{E}/E4_concat_{SET}_gene_coords.csv', 'w', newline='') as f2:
    f.write('#nexus\nbegin sets;\n'); w = csv.writer(f2); w.writerow(['gene', 'start_1based', 'end_1based']); p = 0
    for nm, _, L in passing: f.write(f'  charset {re.sub(r"[^A-Za-z0-9_]", "_", nm)} = {p + 1}-{p + L};\n'); w.writerow([nm, p + 1, p + L]); p += L
    f.write('end;\n')
M = np.array([np.frombuffer(cat[g].encode(), dtype='S1') for g in genomes]); L = M.shape[1]
cnt = np.array([(M == b).sum(0) for b in ACGT]); nstates = (cnt > 0).sum(0); pi = ((cnt >= 2).sum(0) >= 2)
summ = [dict(item='genomes', value=n), dict(item='clusters_total', value=len(rows)), dict(item='clusters_in_all_genomes', value=len(nall)), dict(item='single_copy_core_genes', value=len(sc)), dict(item='genes_passed', value=len(passing)), dict(item='genes_excluded', value=len(sc) - len(passing)),
        dict(item='alignment_route_roary_slice', value=n_roary), dict(item='alignment_route_mafft_realigned', value=n_mafft),
        dict(item='concat_length', value=L), dict(item='constant_columns', value=int((nstates <= 1).sum())), dict(item='variable_columns', value=int((nstates > 1).sum())),
        dict(item='parsimony_informative_columns', value=int(pi.sum())), dict(item='singleton_columns', value=int(((nstates > 1) & ~pi).sum())), dict(item='columns_with_any_gap', value=int((M == b'-').any(0).sum()))]
for rule in RULES: summ.append(dict(item=f'fail_any:{rule}', value=sum(rule in q["fail_reasons"].split(";") for q in qc))); summ.append(dict(item=f'fail_first:{rule}', value=sum(q["first_fail"] == rule for q in qc)))
for g, row in zip(genomes, M):
    summ.append(dict(item=f'gap_fraction:{g}', value=round(float((row == b'-').mean()), 6))); summ.append(dict(item=f'non_ACGT_non_gap:{g}', value=int((~np.isin(row, ACGT) & (row != b'-')).sum())))
wcsv(f'{E}/T_E4_alignment_summary_{SET}.csv', summ)
for s in summ[:14]: print(' ', s['item'], s['value'])
np.savez_compressed(f'{E}/work/arrays_{SET}.npz', D=np.array(Dm), N=np.array(Nm), genes=np.array([p[0] for p in passing]), G=np.array(genomes), length=np.array([p[2] for p in passing]))
print('sha256', sha(f'{E}/E4_concat_{SET}.fasta'), f'E4_concat_{SET}.fasta')
print(f'done {time.time() - t0:.0f}s', time.strftime('%F %T'), 'load', open('/proc/loadavg').read().split()[0])
