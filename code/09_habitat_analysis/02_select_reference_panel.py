#!/usr/bin/env python
"""M5 step 2: genome-set selection for both species (rules of DESIGN_M5_prefixed.md sections 1.1/1.2).
Uses accession, habitat label, quality values, contig counts, existing ANI clusters only -- no gene content.
Output: tables/T_M5_genome_set.csv, work/{lc,vp}_genomes.tsv (name, fasta, existing prokka gff or '-')."""
import os, sys, json, hashlib, subprocess, re, glob
import numpy as np, pandas as pd
R = '.'; P3 = R + '/refs'; M = R + '/work_b/M5'
SEED = 20261005
TARGET = {'vagina': 15, 'urinary': 10, 'animal/poultry': 10}   # 30/20/20 x 1/2, design amendment 1 (before any run)
out = M + '/tables/T_M5_genome_set.csv'
if os.path.exists(out): sys.exit('exists: ' + out)
def sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def seqmd5(f):
    seqs, cur = [], []
    for l in open(f):
        if l.startswith('>'):
            if cur: seqs.append(''.join(cur).upper()); cur = []
        else: cur.append(l.strip())
    if cur: seqs.append(''.join(cur).upper())
    return hashlib.md5('\n'.join(sorted(seqs)).encode()).hexdigest()
def qs_rep(df):  # representative rule
    return df.assign(qs=df.d_comp - 5 * df.d_cont).sort_values(['qs', 'n_contigs', 'accession'], ascending=[False, True, True]).accession.iloc[0]

# ---------------- L. crispatus ----------------
h = pd.read_csv(P3 + '/rescue/E/T_E_habitat_reclass.csv')
ck = pd.concat([pd.read_csv(f, sep='\t').assign(src=os.path.relpath(f, R)) for f in
                [P3 + '/checkmB/out.tsv', P3 + '/checkmSel/out.tsv', M + '/tables/checkm_b1.tsv']])
ck = ck.drop_duplicates('Bin Id').rename(columns={'Bin Id': 'accession', 'Completeness': 'd_comp', 'Contamination': 'd_cont', 'src': 'checkm_source'})[['accession', 'd_comp', 'd_cont', 'checkm_source']]
notes = {}
for l in open(P3 + '/lc_all.jsonl'):
    d = json.loads(l); ai = d['assembly_info']
    notes[d['accession']] = (ai.get('assembly_level'), ';'.join(ai.get('genome_notes') or []))
h['asm_level'] = h.accession.map(lambda a: notes[a][0]); h['genome_notes'] = h.accession.map(lambda a: notes[a][1])
h = h.merge(ck, on='accession', how='left')
rows = []
def add(acc, hab, how):
    r = h[h.accession == acc].iloc[0]
    rows.append(dict(species='Lactobacillus_crispatus', genome=acc, role='reference', habitat=hab, selection=how,
                     strain=r.strain, bioproject=r.bioproject, asm_level=r.asm_level, genome_notes=r.genome_notes,
                     n_contigs=r.n_contigs, ncbi_checkm_comp=r.ncbi_checkm_comp, ncbi_checkm_cont=r.ncbi_checkm_cont,
                     direct_checkm_comp=r.d_comp, direct_checkm_cont=r.d_cont, checkm_source=r.checkm_source,
                     fasta=f'{P3}/lc_fna/{acc}.fna'))
log = []
# gut
g = h[h.new_label == 'human gut'].copy()
assert len(g) == 36 and g.d_comp.notna().all()
g['pass'] = (g.d_comp >= 90) & (g.d_cont < 5)
log.append(f'LC human gut: 36 labelled, direct CheckM pass {int(g["pass"].sum())}, fail: ' +
           '; '.join(f'{r.accession}({r.d_comp}/{r.d_cont})' for r in g[~g["pass"]].itertuples()))
gp = g[g['pass']].copy(); gp['md5'] = [seqmd5(f'{P3}/lc_fna/{a}.fna') for a in gp.accession]
A_PROKKA = set(os.listdir(P3 + '/rescue/A/prokka'))
for md5, grp in gp.groupby('md5'):
    keep = sorted(grp.accession, key=lambda a: (a not in A_PROKKA, a))[0]
    if len(grp) > 1: log.append(f'LC gut sequence-identical records collapsed: {sorted(grp.accession)} -> keep {keep}')
    add(keep, 'human gut', 'all human-gut genomes passing direct CheckM v1.2.4 (>=90/<5)')
# random habitats from E S2 representatives
dc = pd.read_csv(P3 + '/rescue/E/T_E_dedup_clusters.csv')
rng = np.random.default_rng(SEED)
for hab in ['vagina', 'urinary', 'animal/poultry']:
    pool = sorted(dc[dc.rep_label == hab].dedup_rep)
    k = min(TARGET[hab], len(pool))
    pick = sorted(rng.choice(pool, size=k, replace=False)) if k < len(pool) else pool
    log.append(f'LC {hab}: pool (E S2 dedup reps, MAIN) {len(pool)}, target {TARGET[hab]}, taken {len(pick)}')
    for a in pick: add(a, hab, f'random seed {SEED} from {len(pool)} ANI>=99.9% cluster reps (angle E, MAIN)' if k < len(pool) else f'all {len(pool)} ANI>=99.9% cluster reps (pool < target)')
# oral, food/environment: all passing direct CheckM, dedup within habitat
for hab in ['oral', 'food/environment']:
    s = h[h.new_label == hab].copy(); assert s.d_comp.notna().all()
    p = s[(s.d_comp >= 90) & (s.d_cont < 5)]
    log.append(f'LC {hab}: labelled {len(s)}, direct CheckM pass {len(p)}, fail: ' + '; '.join(f'{r.accession}({r.d_comp}/{r.d_cont})' for r in s[~s.accession.isin(p.accession)].itertuples()))
    reps = list(p.accession)
    if len(p) > 1:
        lst = f'{M}/work/dedup_{hab.split("/")[0]}.list'; tri = f'{M}/work/dedup_{hab.split("/")[0]}_tri.tsv'
        open(lst, 'w').write(''.join(f'{P3}/lc_fna/{a}.fna\n' for a in sorted(p.accession)))
        subprocess.run(['nice', '-n', '19', f'{P3}/tools/skani', 'triangle', '-t', '4', '-l', lst, '--full-matrix', '-o', tri], check=True, stderr=subprocess.DEVNULL)
        L = open(tri).read().rstrip('\n').split('\n')[1:]
        names = [os.path.basename(x.split('\t')[0])[:-4] for x in L]
        Mx = np.array([[float(v) for v in x.split('\t')[1:]] for x in L])
        comp = {n: n for n in names}
        def find(x):
            while comp[x] != x: x = comp[x]
            return x
        for i in range(len(names)):
            for j in range(i):
                if Mx[i, j] >= 99.9: comp[find(names[i])] = find(names[j])
        cl = {}
        for n in names: cl.setdefault(find(n), []).append(n)
        reps = [qs_rep(p[p.accession.isin(m)]) for m in cl.values()]
        log.append(f'LC {hab}: ANI>=99.9% clusters within habitat {len(cl)}: ' + ' | '.join(','.join(m) for m in cl.values()))
    if len(reps) > 20: reps = list(rng.choice(sorted(reps), 20, replace=False))
    for a in sorted(reps): add(a, hab, 'all passing direct CheckM, 1 per ANI>=99.9% cluster within habitat')
lc = pd.DataFrame(rows)
MAGS = [('CDM-1_crispatus.4', R + '/mag_bins/bins2/CDM-1/bin.4.fa'), ('CDM-3_crispatus.9', R + '/mag_bins/bins2/CDM-3/bin.9.fa')]
q = pd.read_csv(P3 + '/T_P3_52_checkm_selection.csv').set_index('Bin Id')
lc = pd.concat([lc, pd.DataFrame([dict(species='Lactobacillus_crispatus', genome=n, role='patient MAG', habitat='patient MAG', selection='patient MAG',
        asm_level='MAG', genome_notes='CDM patient MAG', direct_checkm_comp=q.loc[n, 'Completeness'], direct_checkm_cont=q.loc[n, 'Contamination'],
        checkm_source='refs/T_P3_52_checkm_selection.csv', fasta=f) for n, f in MAGS])], ignore_index=True)
lc['asm_type'] = np.where(lc.role == 'patient MAG', 'MAG', np.where(lc.genome_notes.fillna('').str.contains('derived from metagenome'), 'MAG',
                 np.where(lc.asm_level.isin(['Complete Genome', 'Chromosome']), 'complete', 'draft')))
lc['completeness'] = lc.direct_checkm_comp
# existing Prokka 1.12 (rescue/A) reuse: sha256 must equal the one logged at annotation time
logged = dict(re.findall(r'exit=0 (\S+) \d+s CDS=\d+ sha256_fna=(\w+)', open(P3 + '/rescue/A/logs/LOG_prokka.txt').read()))
lc['fasta_sha256'] = lc.fasta.map(sha)
lc['prokka_reuse'] = [f'{P3}/rescue/A/prokka/{g}/{g}.gff' if logged.get(g) == s and os.path.exists(f'{P3}/rescue/A/prokka/{g}/{g}.gff') else '-' for g, s in zip(lc.genome, lc.fasta_sha256)]

# ---------------- V. parvula ----------------
v = pd.read_csv(P3 + '/vp_refs/T_V1_candidates_all.csv')
HAB = {'gut/feces': 'human gut', 'oral': 'oral', 'food/environment/other': 'food/environment/other', 'vagina/cervix': 'vagina'}
log.append('VP: 41 parvula_A; excluded pass_ani False: ' + ','.join(v[~v.pass_ani].use_acc) + '; excluded not stated: ' + ','.join(v[v.habitat == 'not stated'].use_acc))
vv = v[v.pass_ani & v.habitat.isin(HAB)].copy()
vp = pd.DataFrame(dict(species='Veillonella_parvula_A', genome=vv.use_acc, role='reference', habitat=vv.habitat.map(HAB),
     selection='all parvula_A with pass_ani and a habitat label (refs/vp_refs/T_V1_candidates_all.csv)', strain=vv.strain,
     asm_level=vv.assembly_level, genome_notes=vv.ncbi_genome_category.astype(str) + '; host=' + vv.host.astype(str) + '; isolation_source=' + vv.isolation_source.astype(str),
     n_contigs=vv.n_contigs, ncbi_checkm_comp=vv.ncbi_checkm_comp, ncbi_checkm_cont=vv.ncbi_checkm_cont,
     direct_checkm_comp=vv.checkm124_comp, direct_checkm_cont=vv.checkm124_cont, gtdb_checkm_comp=vv.gtdb_checkm_comp, gtdb_checkm_cont=vv.gtdb_checkm_cont,
     checkm_source='GTDB R226 checkm (covariate); checkm124 where present', fasta=[f'{P3}/vp_refs/fna/{a}.fna' for a in vv.use_acc]))
vp['asm_type'] = np.where(vv.ncbi_genome_category.values == 'derived from metagenome', 'MAG', np.where(vv.assembly_level.isin(['Complete Genome', 'Chromosome']).values, 'complete', 'draft'))
vp['completeness'] = vp.gtdb_checkm_comp
VM = [('CDM-1_bin.1', R + '/mag_fasta/V.parvula/V.parvula_CDM1_bin.1.fna', 95.89, 0.6), ('CDM-4_bin.2_clean', R + '/work_b/B1_bin2_refine/CDM-4_bin.2_clean.fa', 91.26, 2.89),
      ('CDM-3_bin.12', R + '/mag_fasta/V.parvula/V.parvula_CDM3_bin.12.fna', 97.6, 0.0)]
vp = pd.concat([vp, pd.DataFrame([dict(species='Veillonella_parvula_A', genome=n, role='patient MAG', habitat='patient MAG', selection='patient MAG', asm_level='MAG',
     genome_notes='CDM patient MAG', direct_checkm_comp=c, direct_checkm_cont=x, completeness=c, checkm_source='CheckM v1.2.4 (work_b/CheckM_recheck, work_b/B1)', fasta=os.path.normpath(f), asm_type='MAG') for n, f, c, x in VM])], ignore_index=True)
vp['fasta_sha256'] = vp.fasta.map(sha)
logged = dict(re.findall(r'exit=0 (\S+) \d+s CDS=\d+ sha256_fna=(\w+)', open(P3 + '/vp_reanalysis/logs/LOG_prokka.txt').read()))
vp['prokka_reuse'] = [f'{P3}/vp_reanalysis/prokka/{g}/{g}.gff' if logged.get(g) == s else '-' for g, s in zip(vp.genome, vp.fasta_sha256)]
b1sha = re.search(r'input sha256=(\w+)', open(R + '/work_b/B1_bin2_refine/logs/LOG_b1_10_prokka.txt').read()).group(1)
i = vp.genome == 'CDM-4_bin.2_clean'
if vp.loc[i, 'fasta_sha256'].iloc[0] == b1sha: vp.loc[i, 'prokka_reuse'] = R + '/work_b/B1_bin2_refine/prokka_clean/CDM-4_bin.2_clean/CDM-4_bin.2_clean.gff'

allg = pd.concat([lc, vp], ignore_index=True)
allg.to_csv(out, index=False)
for sp, d in allg.groupby('species'):
    d[['genome', 'fasta', 'prokka_reuse']].to_csv(f'{M}/work/{"lc" if "Lacto" in sp else "vp"}_genomes.tsv', sep='\t', index=False, header=False)
    log.append(f'{sp}: n={len(d)} | habitat ' + str(d.habitat.value_counts().to_dict()) + ' | asm_type x habitat ' + str(pd.crosstab(d.habitat, d.asm_type).to_dict('index')) + f' | prokka reuse {int((d.prokka_reuse != "-").sum())}, new {int((d.prokka_reuse == "-").sum())}')
open(M + '/logs/LOG_m5_02_select.txt', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))
