#!/usr/bin/env python
"""M6 step 4: per-contig table (length, GC, depth, panel/decoy covered fractions) for an assembly.
usage: m6_04_contig_table.py contigs.fa blast.tsv out.csv [samtools_coverage.tsv]
Covered fraction = union of blastn HSPs (megablast, identity>=90 %, length>=200 bp) on the contig, per genome / per genome set."""
import sys, csv, collections
fa, bl, out = sys.argv[1:4]
cov = sys.argv[4] if len(sys.argv) > 4 else None
M247 = 'GCF_048541565.1'
seqs, multi = {}, {}
for line in open(fa):
    if line[0] == '>':
        p = line[1:].split(); name = p[0]; seqs[name] = []
        multi[name] = float([x for x in p if x.startswith('multi=')][0][6:])
    else: seqs[name].append(line.strip())
seqs = {k: ''.join(v) for k, v in seqs.items()}
depth = {}
if cov:
    for r in csv.DictReader(open(cov), delimiter='\t'):
        depth[r['#rname']] = (float(r['meandepth']), float(r['coverage']))
def ulen(iv):
    iv.sort(); t = 0; e0 = 0
    for s, e in iv:
        if e > e0: t += e - max(s - 1, e0); e0 = e
    return t
per = collections.defaultdict(lambda: collections.defaultdict(list))   # contig -> genome -> intervals
best_dc = collections.defaultdict(float); best_lc = collections.defaultdict(float)
for line in open(bl):
    q, s, pid, ln, qs, qe = line.split('\t')[:6]
    kind, g = s.split('__')[:2]
    g = kind + '__' + g
    a, b = sorted((int(qs), int(qe)))
    per[q][g].append((a, b))
    d = best_dc if kind == 'DC' else best_lc
    d[q] = max(d[q], float(pid))
rows = []
for c, sq in seqs.items():
    L = len(sq); G = per.get(c, {})
    f = lambda keys: ulen([iv for k in keys for iv in G[k]]) / L if keys else 0.0
    pub = [k for k in G if k.startswith('LC__GC')]
    single = {k: ulen(list(G[k])) / L for k in pub}
    bs = max(single, key=single.get) if single else ''
    rows.append(dict(contig=c, length=L, gc=round(100 * (sq.count('G') + sq.count('C')) / L, 2), megahit_multi=multi[c],
        depth=depth.get(c, ('', ''))[0], breadth=depth.get(c, ('', ''))[1],
        f_M247=round(single.get('LC__' + M247, 0), 4), best_single_ref=bs.replace('LC__', ''), f_best_single=round(single.get(bs, 0), 4),
        f_public_panel=round(f(pub), 4),
        f_MAG_CDM1=round(f([k for k in G if k == 'LC__MAG_CDM1_CDM-1bin4']), 4),
        f_MAG_CDM3=round(f([k for k in G if k == 'LC__MAG_CDM3_CDM3bin9']), 4),
        f_LC_all=round(f([k for k in G if k.startswith('LC__')]), 4),
        f_decoy=round(f([k for k in G if k.startswith('DC__')]), 4),
        best_decoy=max((k for k in G if k.startswith('DC__')), key=lambda k: ulen(list(G[k])), default='').replace('DC__', ''),
        max_pid_LC=best_lc.get(c, ''), max_pid_decoy=best_dc.get(c, '')))
with open(out, 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
T = sum(r['length'] for r in rows)
assert all(0 <= r[k] <= 1 for r in rows for k in r if k.startswith('f_'))
print(f'{out}: contigs={len(rows)} bp={T}')
for k in ['f_M247', 'f_best_single', 'f_public_panel', 'f_MAG_CDM1', 'f_MAG_CDM3', 'f_LC_all', 'f_decoy']:
    print(f'  bp-weighted {k} = {sum(r[k]*r["length"] for r in rows)/T:.4f}')
