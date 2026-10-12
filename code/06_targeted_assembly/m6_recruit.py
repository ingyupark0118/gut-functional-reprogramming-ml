#!/usr/bin/env python
"""M6 read recruitment. FASTQ on stdin -> recruited FASTQ on stdout-file.
Rule: a read is recruited if its fragment (read id without /1,/2) has >=1 read whose reported bowtie2 alignment is on an
LC__ target in ANY given hits file (mate rescue), unless the read itself is best-aligned to a DC__ (decoy) target in any hits file.
usage: m6_recruit.py OUT.fq STATS.txt hits1.tsv [hits2.tsv ...]   (hits: read_id <tab> first 2 chars of RNAME)"""
import sys
out, stats, hits = sys.argv[1], sys.argv[2], sys.argv[3:]
lc_frag, lc_read, dc_read = set(), set(), set()
per = []
for h in hits:
    nl = nd = 0
    for line in open(h):
        rid, t = line.rstrip('\n').split('\t')
        if t == 'LC':
            lc_read.add(rid); lc_frag.add(rid[:-2]); nl += 1
        elif t == 'DC':
            dc_read.add(rid); nd += 1
    per.append((h, nl, nd))
n = w = direct = mate = dropped_dc = bases = 0
with open(out, 'w') as fo:
    it = iter(sys.stdin)
    for hdr in it:
        seq, plus, qual = next(it), next(it), next(it)
        n += 1
        rid = hdr[1:].split()[0]
        if rid[:-2] in lc_frag:
            if rid in dc_read:
                dropped_dc += 1; continue
            if rid in lc_read: direct += 1
            else: mate += 1
            w += 1; bases += len(seq) - 1
            fo.write(hdr + seq + plus + qual)
with open(stats, 'w') as fs:
    for h, nl, nd in per:
        fs.write(f'hits_file={h}\tLC_best={nl}\tDC_best={nd}\n')
    fs.write(f'total_reads={n}\nLC_reads_union={len(lc_read)}\nDC_reads_union={len(dc_read)}\n'
             f'recruited={w}\nrecruited_direct={direct}\nrecruited_mate_rescue={mate}\n'
             f'fragment_LC_but_read_decoy_dropped={dropped_dc}\nrecruited_bases={bases}\n')
assert w == direct + mate and direct <= len(lc_read)
