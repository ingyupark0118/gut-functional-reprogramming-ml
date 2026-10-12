#!/usr/bin/env python
"""Count polymorphic sites from samtools mpileup (stdin). Site evaluated if depth>=20 (base calls, indels ignored);
'polymorphic' if minor-allele frequency >= 0.10 and minor count >= 4; 'fixed difference vs consensus' if ref allele < 10 %.
usage: samtools mpileup ... | m6_snv.py keep_contigs.txt out.tsv"""
import sys, re, collections
keep = set(l.split()[0] for l in open(sys.argv[1]))
ev = collections.Counter(); poly = collections.Counter(); hist = collections.Counter()
rx = re.compile(r'\^.|\$|[+-](\d+)')
def strip(s):
    out = []; i = 0
    while i < len(s):
        c = s[i]
        if c == '^': i += 2; continue
        if c in '+-':
            j = i + 1
            while s[j].isdigit(): j += 1
            i = j + int(s[i+1:j]); continue
        if c not in '$*#': out.append(c)
        i += 1
    return out
for line in sys.stdin:
    p = line.split('\t')
    if p[0] not in keep: continue
    b = strip(p[4]); n = len(b)
    if n < 20: continue
    ev[p[0]] += 1
    c = collections.Counter('R' if x in '.,' else x.upper() for x in b)
    top = c.most_common(2)
    if len(top) > 1 and top[1][1] >= 4 and top[1][1] / n >= 0.10:
        poly[p[0]] += 1; hist[min(int(10 * top[1][1] / n), 4)] += 1
with open(sys.argv[2], 'w') as fo:
    fo.write('contig\tsites_evaluated\tpolymorphic_sites\tper_kb\n')
    for k in sorted(ev): fo.write(f'{k}\t{ev[k]}\t{poly[k]}\t{1000*poly[k]/ev[k]:.3f}\n')
    E, P = sum(ev.values()), sum(poly.values())
    fo.write(f'TOTAL\t{E}\t{P}\t{1000*P/max(E,1):.3f}\n')
    fo.write('#minor_allele_freq_bins(0.1-0.2,0.2-0.3,0.3-0.4,0.4-0.5)\t' + '\t'.join(str(hist[i]) for i in (1, 2, 3, 4)) + '\n')
