#!/usr/bin/env python
"""M5_ext step 7: summary tables (no new statistics except BH over the 3 primary P and descriptive neighbour counts)."""
import os, numpy as np, pandas as pd
R = '.'; X = R + '/work_b/M5_ext'; T = X + '/tables/'; M5 = R + '/work_b/M5'
meta = pd.read_csv(X + '/work/meta_MAIN.csv').set_index('genome')
runs = [r for r in ['MAIN', 'S1', 'S2a', 'S2b', 'C0', 'C0p4', 'C0rep'] if os.path.exists(T + f'T_M5ext_placement_{r}.csv')]
old = pd.read_csv(M5 + '/tables/T_M5_placement_lc.csv').rename(columns={'MAG': 'genome'}); old['run'] = 'M5_original'
P = pd.concat([old] + [pd.read_csv(T + f'T_M5ext_placement_{r}.csv').assign(run=r if r != 'S2b' else 'S2b') for r in runs], ignore_index=True)
P['genome0'] = P.genome.str.replace('_S1$', '', regex=True)
rows = []
for (run, g), d in P.groupby(['run', 'genome'], sort=False):
    c2 = d[d.method.str.endswith('1-2')].iloc[0]; c10 = d[d.method.str.endswith('1-10')].iloc[0]
    k5 = d[(d.method == 'kNN by Jaccard') & (d.k == 5)].iloc[0]; k10 = d[(d.method == 'kNN by Jaccard') & (d.k == 10)].iloc[0]
    r = dict(run=run, genome=g, centroid_axes12=c2.observed, centroid12_P=c2.P_perm, centroid12_P_strat=c2.P_perm_within_asm_type, centroid_axes1_10=c10.observed, centroid10_P=c10.P_perm, centroid10_P_strat=c10.P_perm_within_asm_type,
             k5_gut=int(k5.n_gut_among_k), k5_P=k5.P_perm, k5_P_strat=k5.P_perm_within_asm_type, k10_gut=int(k10.n_gut_among_k), k10_composition=k10.observed, k10_P=k10.P_perm, k10_P_strat=k10.P_perm_within_asm_type,
             meets_criterion=bool((c2.observed == 'human gut') and (k10.P_perm < 0.05)))
    if 'n_neighbours_ANI_ge_99_5' in d and run != 'M5_original':
        d1 = d[d.method.str.startswith('D1')].iloc[0]; d2 = d[d.method.str.startswith('D2')].iloc[0]
        r.update(k10_n_clonal_ANI995=int(k10.n_neighbours_ANI_ge_99_5), k10_n_clonal_gut=int(k10.n_gut_neighbours_ANI_ge_99_5), k10_neighbour_asm=k10.neighbours_asm,
                 D1_gut=int(d1.n_gut_among_k), D1_expected=d1.expected_gut, D1_P=d1.P_perm, D1_P_strat=d1.P_perm_within_asm_type, D2_gut=int(d2.n_gut_among_k), D2_expected=d2.expected_gut, D2_P=d2.P_perm, D2_P_strat=d2.P_perm_within_asm_type,
                 max_ANI_ref=d.max_ANI_ref.iloc[0], max_ANI=d.max_ANI.iloc[0], n_refs_ANI_ge_99_5=int(d.n_refs_ANI_ge_99_5.iloc[0]))
    rows.append(r)
S = pd.DataFrame(rows); g0 = S.genome.str.replace('_S1$', '', regex=True)
for c in ['status', 'sample', 'group', 'lineage', 'n_contigs', 'completeness', 'direct_checkm_cont', 'n_gene_families']: S.insert(2, c, g0.map(meta[c]).values)
# BH over the 3 primary genomes (MAIN)
m = S[(S.run == 'MAIN') & (S.status == 'PRIMARY')].sort_values('k10_P'); q = np.minimum.accumulate((m.k10_P.values * 3 / np.arange(1, 4))[::-1])[::-1]
S['k10_BH_q_primary3_MAIN'] = S.index.map(dict(zip(m.index, q)))
S.to_csv(T + 'T_M5ext_placement_summary.csv', index=False)
# lineage summary (MAIN)
L = []
for l, d in S[S.run == 'MAIN'].groupby('lineage'):
    n = int(d.meets_criterion.sum()); L.append(dict(lineage=l, members='; '.join(f'{g}({"meets" if v else "not"})' for g, v in zip(d.genome, d.meets_criterion)), n_members=len(d), n_meet=n,
                                             lineage_call='meets' if n == len(d) else ('does not meet' if n == 0 else 'inconsistent'), k10_gut_range=f'{d.k10_gut.min()}-{d.k10_gut.max()}', k10_P_range=f'{d.k10_P.min():.4f}-{d.k10_P.max():.4f}'))
L = pd.DataFrame(L); L.to_csv(T + 'T_M5ext_lineage_summary.csv', index=False)
# neighbour lists (MAIN), with ANI>=99.9 cluster of M5 and BioProject
D = pd.read_csv(T + 'M5_MAIN_jaccard.csv', index_col=0); A = pd.read_csv(X + '/work/ani_MAIN.csv', index_col=0); refs = meta.index[meta.role == 'reference']; N = []
for g in meta.index[meta.role != 'reference']:
    for i, a in enumerate(sorted(refs, key=lambda a: (D.loc[g, a], a))[:10], 1):
        N.append(dict(genome=g, lineage=meta.lineage[g], rank=i, neighbour=a, habitat=meta.habitat[a], asm_type=meta.asm_type[a], jaccard=round(D.loc[g, a], 4), ANI=A.loc[g, a], same_clone_ANI_ge_99_5=bool(A.loc[g, a] >= 99.5),
                      ref_cluster_ANI999=meta.cluster[a], ref_cluster_size=int(meta.cluster_size[a]), bioproject=meta.bioproject[a], strain=meta.strain[a]))
N = pd.DataFrame(N); N.to_csv(T + 'T_M5ext_neighbours_MAIN.csv', index=False)
ind = N.groupby('genome', sort=False).apply(lambda d: pd.Series(dict(n_gut=int((d.habitat == 'human gut').sum()), n_clonal=int(d.same_clone_ANI_ge_99_5.sum()), n_distinct_clusters999=d.ref_cluster_ANI999.nunique(),
        n_distinct_gut_clusters999=d[d.habitat == 'human gut'].ref_cluster_ANI999.nunique(), n_distinct_bioprojects_gut=d[d.habitat == 'human gut'].bioproject.nunique(), n_MAG=int((d.asm_type == 'MAG').sum()), ANI_range=f'{d.ANI.min():.2f}-{d.ANI.max():.2f}', jaccard_10th=d.jaccard.iloc[-1])))
ind.to_csv(T + 'T_M5ext_independence_MAIN.csv')
sets = {g: set(d.neighbour) for g, d in N.groupby('genome')}
ov = pd.DataFrame([[len(sets[a] & sets[b]) for b in sets] for a in sets], index=sets, columns=sets); ov.to_csv(T + 'T_M5ext_neighbour_overlap_MAIN.csv')
# PERMANOVA comparison
pm = [pd.read_csv(M5 + '/tables/T_M5_permanova_lc.csv').assign(matrix='M5_original')] + [pd.read_csv(T + f'T_M5_permanova_{r}.csv').assign(matrix=r) for r in runs if r != 'S2b']
pm = pd.concat(pm); pm.to_csv(T + 'T_M5ext_permanova_all.csv', index=False)
w = pm[pm.test.isin(['T1_unconditioned', 'T2_relatedness', 'T3_assembly', 'T4_all', 'betadisper'])].copy(); w['val'] = [f'{f:.2f} / {p:.4f}' if t == 'betadisper' else f'{r:.3f} / {p:.4f}' for t, r, f, p in zip(w.test, w.R2_habitat, w.F, w.P)]
cmpt = w.pivot_table(index=['set', 'coding', 'test'], columns='matrix', values='val', aggfunc='first')[[c for c in ['M5_original'] + runs if c != 'S2b']]; cmpt.to_csv(T + 'T_M5ext_permanova_compare.csv')
# figure coordinates
co = pd.read_csv(T + 'M5_MAIN_pcoa.csv'); eg = pd.read_csv(T + 'M5_MAIN_pcoa_eig.csv')
f = pd.DataFrame(dict(genome=co.genome, class_=np.where(co.role == 'reference', 'reference', co.genome.map(meta.status)), habitat=co.habitat, role=co.role, asm_type=co.asm_type,
                      sample=co.genome.map(meta['sample']), group=co.genome.map(meta.group), lineage=co.genome.map(meta.lineage),
                      label=co.genome.map(lambda g: '' if g.startswith('GC') else g.replace('_Lcrispatus_targeted', ' (targeted)').replace('_crispatus', ' MAG bin')),
                      meets_criterion_MAIN=co.genome.map(S[S.run == 'MAIN'].set_index('genome').meets_criterion)))
f = f.rename(columns={'class_': 'class'})
for i in range(1, 11): f[f'PC{i}'] = co[f'PCo{i}'].values
for i in range(1, 11): f[f'PC{i}_pct_explained'] = round(eg.pct_of_positive[i - 1], 3)
f.to_csv(T + 'M5ext_Lc_pcoa.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 300)
print(S[['run', 'genome', 'status', 'lineage', 'centroid_axes12', 'centroid12_P', 'k5_gut', 'k5_P', 'k10_gut', 'k10_P', 'k10_P_strat', 'meets_criterion', 'k10_BH_q_primary3_MAIN']].to_string(index=False))
print(L.to_string(index=False)); print(ind.to_string()); print(ov.to_string()); print(cmpt.to_string()); print('PCoA % axes 1-4 MAIN:', eg.pct_of_positive[:4].round(2).tolist())
