#!/usr/bin/env python
"""Angle F step 10: single-copy core genes -> per-gene alignment -> QC (DESIGN_F_prefixed.md s1-4) -> concatenation.
Inputs (read-only): rescue/A/roary/{gene_presence_absence.csv,core_gene_alignment.aln,core_alignment_header.embl}, rescue/A/prokka/*/*.ffn
Outputs: T_F_gene_qc.csv, T_F_alignment_summary.csv, T_F_per_gene_nearest.csv, T_F_qc_offenders.csv, F_concat.fasta, F_concat_partitions.nex, work/genes/*.aln
Run: nice -n 19 ionice -c3 python F_10_gene_qc.py   (1 thread; MAFFT 1 thread only for genes that need re-alignment)"""
import csv, hashlib, itertools, os, re, subprocess, sys, time
import numpy as np
A = sys.argv[1]  # set dir with roary/ and prokka/
F = sys.argv[1]
MAFFT = 'mafft'
MIN_LEN, MIN_COV, MIN_ID, MAX_AMBIG = 300, 0.80, 0.90, 0.01
MAGS = sys.argv[2].split(',')  # patient genomes (descriptive columns only)
STOPS = {'TAA', 'TAG', 'TGA'}
t0 = time.time()
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def fasta(p):
    d, k = {}, None
    for l in open(p):
        if l[0] == '>': k = l[1:].split()[0]; d[k] = []
        else: d[k].append(l.strip())
    return {k: ''.join(v).upper() for k, v in d.items()}
os.makedirs(f'{F}/work/genes', exist_ok=True)
for out in ['T_F_gene_qc.csv', 'F_concat.fasta']:
    assert not os.path.exists(f'{F}/{out}'), f'{out} exists - refusing to overwrite'
print('### F_10_gene_qc', time.strftime('%F %T'), 'load', open('/proc/loadavg').read().split()[0])
print('python', sys.version.split()[0], 'numpy', np.__version__, '|', subprocess.run([MAFFT, '--version'], capture_output=True, text=True).stderr.strip())
gpa, core_aln, embl = f'{A}/roary/gene_presence_absence.csv', f'{A}/roary/core_gene_alignment.aln', f'{A}/roary/core_alignment_header.embl'
for p in [gpa, core_aln, embl]: print('sha256', sha(p), p)
rows = list(csv.reader(open(gpa)))
hdr, rows = rows[0], rows[1:]
genomes = hdr[14:]; NG = len(genomes); assert NG == int(sys.argv[3]) and all(m in genomes for m in MAGS)
ffn = {}
for g in genomes:
    p = f'{A}/prokka/{g}/{g}.ffn'; print('sha256', sha(p), p); ffn.update(fasta(p))
core = fasta(core_aln); assert set(core) == set(genomes), 'core alignment taxa != 22 genomes'
coords = {}
txt = open(embl).read()
for m in re.finditer(r'feature\s+(\d+)\.\.(\d+)\nFT\s+/label=(\S+)', txt): coords[m.group(3)] = (int(m.group(1)) - 1, int(m.group(2)))
print('roary core alignment: length', len(next(iter(core.values()))), 'genes in embl header', len(coords))
# s1 gene set
n22 = [r for r in rows if r[3] == str(NG)]
sc = [r for r in n22 if r[4] == str(NG) and all(c and not re.search(r'[\t; ]', c) for c in r[14:])]
print(f'clusters total {len(rows)}; in 22/22 genomes {len(n22)}; single-copy core (22 isolates, 22 sequences, one locus tag each) {len(sc)}; 22/22 but multi-copy {len(n22) - len(sc)}')
ACGT = np.frombuffer(b'ACGT', dtype='S1')
qc, nearest, offend_rows, passing = [], [], [], []
n_roary = n_mafft = 0
for r in sorted(sc, key=lambda r: r[0]):
    name = r[0]; tags = dict(zip(genomes, r[14:])); raw = {g: ffn[tags[g]] for g in genomes}
    route, aln = 'mafft_realigned', None
    if name in coords:
        a, b = coords[name]; sl = {g: core[g][a:b] for g in genomes}
        if all(sl[g].replace('-', '') == raw[g] for g in genomes): route, aln = 'roary_mafft_slice', sl
        else: route = 'mafft_realigned(slice!=ffn)'
    else: route = 'mafft_realigned(not in roary core aln)'
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    if aln is None:
        tmp = f'{F}/work/genes/{safe}.unaligned.fa'
        open(tmp, 'w').write(''.join(f'>{g}\n{raw[g]}\n' for g in genomes))
        out = subprocess.run([MAFFT, '--auto', '--quiet', '--thread', '1', tmp], capture_output=True, text=True, check=True).stdout
        d, k = {}, None
        for l in out.splitlines():
            if l[0] == '>': k = l[1:].strip(); d[k] = []
            else: d[k].append(l.strip())
        aln = {k: ''.join(v).upper() for k, v in d.items()}; n_mafft += 1
        assert all(aln[g].replace('-', '') == raw[g] for g in genomes)
    else: n_roary += 1
    open(f'{F}/work/genes/{safe}.aln', 'w').write(''.join(f'>{g}\n{aln[g]}\n' for g in genomes))
    M = np.array([np.frombuffer(aln[g].encode(), dtype='S1') for g in genomes]); L = M.shape[1]
    nong = M != b'-'; valid = np.isin(M, ACGT)
    cov = nong.sum(1) / L
    ident = np.ones((NG, NG))
    for i, j in itertools.combinations(range(NG), 2):
        both = valid[i] & valid[j]; n = both.sum()
        ident[i, j] = ident[j, i] = ((M[i] == M[j]) & both).sum() / n if n else np.nan
    off = ~np.eye(NG, dtype=bool)
    minid_per = np.array([np.nanmin(ident[i][off[i]]) for i in range(NG)]); medid_per = np.array([np.nanmedian(ident[i][off[i]]) for i in range(NG)])
    codon_bad, ambig_bad = [], []
    for g in genomes:
        s = raw[g]
        if len(s) % 3 or any(s[k:k + 3] in STOPS for k in range(0, len(s) - 3, 3)): codon_bad.append(g)
        if sum(c not in 'ACGT' for c in s) / len(s) > MAX_AMBIG: ambig_bad.append(g)
    fails = {}
    if L < MIN_LEN: fails['Q1_length<300'] = []
    if cov.min() < MIN_COV: fails['Q2_coverage<0.80'] = [g for g, c in zip(genomes, cov) if c < MIN_COV]
    if np.nanmin(minid_per) < MIN_ID or np.isnan(ident).any():
        o = [g for g, m in zip(genomes, medid_per) if m < MIN_ID]
        fails['Q3_identity<0.90'] = o if o else ['NO_SINGLE_OFFENDER:' + '|'.join(sorted({genomes[k] for i, j in itertools.combinations(range(NG), 2) if not ident[i, j] >= MIN_ID for k in (i, j)}))]
    if codon_bad: fails['Q4_codon'] = codon_bad
    if ambig_bad: fails['Q5_ambiguity>1%'] = ambig_bad
    for k, v in fails.items():
        for g in v: offend_rows.append([name, k, g])
    lens = [len(raw[g]) for g in genomes]
    qc.append(dict(gene=name, annotation=r[2], route=route, aln_length=L, min_ungapped_len=min(lens), max_ungapped_len=max(lens),
                   min_coverage=round(cov.min(), 4), min_coverage_genome=genomes[int(cov.argmin())],
                   min_pairwise_identity=round(float(np.nanmin(minid_per)), 4), min_identity_genome=genomes[int(np.nanargmin(medid_per))],
                   **{f'{m}_min_identity': round(float(minid_per[genomes.index(m)]), 4) for m in MAGS},
                   **{f'{m}_coverage': round(float(cov[genomes.index(m)]), 4) for m in MAGS},
                   status='PASS' if not fails else 'FAIL', first_fail=next(iter(fails), ''), fail_reasons=';'.join(fails),
                   offending_genomes=';'.join(f'{k}={",".join(v)}' for k, v in fails.items() if v)))
    if not fails:
        passing.append((name, aln, L))
        for m in MAGS:
            i = genomes.index(m); d = 1 - ident[i]; d[i] = np.inf; mn = d.min()
            near = [genomes[j] for j in range(NG) if j != i and abs(d[j] - mn) < 1e-12]
            nearest.append(dict(gene=name, mag=m, aln_length=L, min_p_distance=round(float(mn), 6), n_nearest=len(near), nearest_genomes=';'.join(near)))
def wcsv(p, recs):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
wcsv(f'{F}/T_F_gene_qc.csv', qc); wcsv(f'{F}/T_F_per_gene_nearest.csv', nearest)
with open(f'{F}/T_F_qc_offenders.csv', 'w', newline='') as f: csv.writer(f).writerows([['gene', 'rule', 'offending_genome']] + offend_rows)
print(f'alignment route: roary slice {n_roary}, mafft re-aligned {n_mafft}')
print(f'QC: pass {len(passing)}, fail {len(qc) - len(passing)}')
for rule in ['Q1_length<300', 'Q2_coverage<0.80', 'Q3_identity<0.90', 'Q4_codon', 'Q5_ambiguity>1%']:
    print(f'  {rule}: any {sum(rule in q["fail_reasons"].split(";") for q in qc)}, first-failing {sum(q["first_fail"] == rule for q in qc)}')
# s4 concatenation (all columns kept)
cat = {g: ''.join(a[g] for _, a, _ in passing) for g in genomes}
with open(f'{F}/F_concat.fasta', 'w') as f:
    for g in genomes: f.write(f'>{g}\n{cat[g]}\n')
with open(f'{F}/F_concat_partitions.nex', 'w') as f:
    f.write('#nexus\nbegin sets;\n'); p = 0
    for n, _, L in passing: f.write(f'  charset {re.sub(r"[^A-Za-z0-9_]", "_", n)} = {p + 1}-{p + L};\n'); p += L
    f.write('end;\n')
with open(f'{F}/F_concat_gene_coords.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['gene', 'start_1based', 'end_1based']); p = 0
    for n, _, L in passing: w.writerow([n, p + 1, p + L]); p += L
M = np.array([np.frombuffer(cat[g].encode(), dtype='S1') for g in genomes]); L = M.shape[1]
cnt = np.array([(M == b).sum(0) for b in ACGT])
nstates = (cnt > 0).sum(0); pi = ((cnt >= 2).sum(0) >= 2)
summ = [dict(item='genomes', value=NG), dict(item='single_copy_core_genes', value=len(sc)), dict(item='genes_passed', value=len(passing)), dict(item='genes_excluded', value=len(sc) - len(passing)),
        dict(item='alignment_route_roary_slice', value=n_roary), dict(item='alignment_route_mafft_realigned', value=n_mafft),
        dict(item='concat_length', value=L), dict(item='constant_columns', value=int((nstates <= 1).sum())), dict(item='variable_columns', value=int((nstates > 1).sum())),
        dict(item='parsimony_informative_columns', value=int(pi.sum())), dict(item='singleton_columns', value=int(((nstates > 1) & ~pi).sum())),
        dict(item='columns_with_any_gap', value=int((M == b'-').any(0).sum()))]
for g, row in zip(genomes, M):
    summ.append(dict(item=f'gap_fraction:{g}', value=round(float((row == b'-').mean()), 6)))
    summ.append(dict(item=f'non_ACGT_non_gap:{g}', value=int((~np.isin(row, ACGT) & (row != b'-')).sum())))
wcsv(f'{F}/T_F_alignment_summary.csv', summ)
for s in summ[:12]: print(' ', s['item'], s['value'])
print('max gap fraction', max(s['value'] for s in summ if s['item'].startswith('gap_fraction')))
print(f'done {time.time() - t0:.0f}s', time.strftime('%F %T'), 'load', open('/proc/loadavg').read().split()[0])
