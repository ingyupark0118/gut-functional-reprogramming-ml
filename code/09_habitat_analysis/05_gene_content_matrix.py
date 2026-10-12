#!/usr/bin/env python
"""M5_ext step 3: presence matrices + metadata + ANI for one Roary run. Usage: m5ext_30_matrix.py <MAIN|S1|S2a|C0>
Same logic as work_b/M5/scripts/m5_30_matrix.py (filter >=2 and <=n-2 of ALL genomes of the run). Reference metadata
(habitat, asm_type, completeness, S_dedup flags) are taken unchanged from work_b/M5/work/meta_lc.csv."""
import sys, numpy as np, pandas as pd
run = sys.argv[1]; R = '.'; M5 = R + '/work_b/M5'; X = R + '/work_b/M5_ext'
g = pd.read_csv(M5 + '/work/meta_lc.csv')
assert len(g) == 77 and (g.role == 'reference').sum() == 75
q = pd.read_csv(R + '/work_b/M6/T_M6_all_samples_summary.csv'); q = q[q.section == 'per_sample'].pivot(index='entity', columns='metric', values='value')
GROUP = {'Cirrhosis-1': 'Cirrhosis', 'CDM-1': 'CDM', 'CDM-4': 'CDM', 'CDM-2': 'CDM', 'CDM-3': 'CDM'}
LIN = {'CDM-1': 'L1', 'CDM-2': 'L1', 'CDM-3': 'L2', 'Cirrhosis-1': 'L2', 'CDM-4': 'L3'}
new = []
for s in GROUP:
    assert q.loc[s, 'group'] == GROUP[s]
    for suf in ('', '_S1') if s == 'CDM-2' else ('',):
        a = f'{s}_Lcrispatus_targeted{suf}'
        new.append(dict(species='Lactobacillus_crispatus', genome=a, role='patient targeted assembly', habitat='patient targeted assembly',
                        selection='M6 targeted assembly' + (' minus 10 L. helveticus-like contigs' if suf else ''), asm_type='targeted',
                        n_contigs=sum(1 for l in open(f'{X}/work/fna/{a}.fa') if l[0] == '>'),
                        direct_checkm_comp=float(q.loc[s, 'checkm_completeness']), direct_checkm_cont=float(q.loc[s, 'checkm_contamination']),
                        completeness=float(q.loc[s, 'checkm_completeness']), checkm_source='work_b/M6 (full assembly; not re-measured for S1)', fasta=f'{X}/work/fna/{a}.fa',
                        sample=s, group=GROUP[s], lineage=LIN[s]))
g['sample'] = g.genome.map({'CDM-1_crispatus.4': 'CDM-1', 'CDM-3_crispatus.9': 'CDM-3'}); g['group'] = g['sample'].map(GROUP); g['lineage'] = g['sample'].map(LIN)
g = pd.concat([g, pd.DataFrame(new)], ignore_index=True)
g['status'] = g.genome.map({'CDM-1_crispatus.4': 'previous M5 MAG', 'CDM-3_crispatus.9': 'previous M5 MAG', 'Cirrhosis-1_Lcrispatus_targeted': 'PRIMARY', 'CDM-4_Lcrispatus_targeted': 'PRIMARY',
                            'CDM-2_Lcrispatus_targeted': 'PRIMARY', 'CDM-2_Lcrispatus_targeted_S1': 'PRIMARY (S1 copy)', 'CDM-1_Lcrispatus_targeted': 'TECHNICAL CONTROL', 'CDM-3_Lcrispatus_targeted': 'TECHNICAL CONTROL'}).fillna('reference')
rt = pd.read_csv(f'{X}/work/roary_{run}/out/gene_presence_absence.Rtab', sep='\t', index_col=0)
g = g[g.genome.isin(rt.columns)].reset_index(drop=True)
exp = {'MAIN': 82, 'S1': 82, 'S2a': 80, 'C0': 77, 'C0p4': 77, 'C0rep': 77}[run]
assert set(rt.columns) == set(g.genome) and len(g) == exp, (len(g), set(rt.columns) ^ set(g.genome))
assert (g.role == 'reference').sum() == 75
rt = (rt[list(g.genome)] > 0).astype(int); n = rt.shape[1]; k = rt.sum(1)
rt.to_csv(f'{X}/tables/M5_{run}_presence_full.csv.gz')
f = rt[(k >= 2) & (k <= n - 2)]; f.to_csv(f'{X}/tables/M5_{run}_presence_filtered.csv.gz')
g['n_gene_families_M5'] = g.n_gene_families; g['n_gene_families'] = rt.sum(0).reindex(g.genome).values
# ANI: skani 0.3.2 triangle on 77 M5 genomes + 6 new FASTA; the 77 x 77 part must equal M5's matrix
L = open(f'{X}/work/ani_tri.tsv').read().rstrip('\n').split('\n')[1:]
old = pd.read_csv(M5 + '/work/ani_lc.csv', index_col=0)
p2g = dict(zip(pd.read_csv(M5 + '/work/meta_lc.csv').fasta, pd.read_csv(M5 + '/work/meta_lc.csv').genome))
names = [p2g.get(x.split('\t')[0], x.split('\t')[0].split('/')[-1][:-3]) for x in L]
A = pd.DataFrame([[float(v) for v in x.split('\t')[1:]] for x in L], index=names, columns=names)
dmax = float(np.abs(A.loc[old.index, old.columns].values - old.values).max())
nz = A.values[A.values > 0]; n0 = int((A.values == 0).sum() / 2); A = A.where(A > 0, nz.min() - 1)
A.loc[old.index, old.columns] = old.values          # reference covariates = existing M5 values, unchanged
A.to_csv(f'{X}/work/ani_all83.csv'); A.loc[list(g.genome), list(g.genome)].to_csv(f'{X}/work/ani_{run}.csv')
g.to_csv(f'{X}/work/meta_{run}.csv', index=False)
print(f'{run}: genomes {n}; gene families total {len(rt)}, filtered (>=2, <=n-2) {len(f)}; core(all) {int((k == n).sum())}; singletons {int((k == 1).sum())}')
print('ANI: max |new skani - M5 matrix| over the 77x77 block =', dmax, '| pairs without skani output (83 set):', n0)
print(g[g.role != 'reference'][['genome', 'status', 'group', 'lineage', 'n_contigs', 'completeness', 'direct_checkm_cont', 'n_gene_families', 'n_gene_families_M5']].to_string())
print(g.groupby(['habitat', 'asm_type']).agg(n=('genome', 'size'), fam=('n_gene_families', 'median')).to_string())
