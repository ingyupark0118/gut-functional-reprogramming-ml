"""T17 calculation: chance probability of 'no vagina-derived reference among the 10 nearest neighbours'.
Candidate pool = the 75 references of the habitat analysis (gut 33, vagina 15, urinary 10, animal 10, oral 4, food/environment 3) - not the 1,185 public genomes.
Neighbour lists are taken from the cross-check table (run=MAIN). Null: the habitat labels of the 75 references are permuted jointly (same scheme as T13/T16); stratified = permuted within assembly type.
Post hoc descriptive statistic (not pre-specified)."""
import pandas as pd, numpy as np, hashlib
from math import comb
R='./'
xc=pd.read_csv(R+'work_b/_crosscheck_M5ext/xc_placement_all.csv'); m=xc[xc.run=='MAIN'].set_index('genome')
gs=pd.read_csv(R+'work_b/M5/tables/T_M5_genome_set.csv'); gs=gs[gs.species.str.contains('crispatus')&(gs.role=='reference')]
ids=list(gs.genome); hab=np.array(gs.habitat.astype(str)); asm=np.array(gs.asm_type.astype(str)); idx={g:i for i,g in enumerate(ids)}
vag=(hab=='vagina'); assert len(ids)==75 and vag.sum()==15
comp=pd.Series(hab).value_counts(); print(comp.to_string()); comp.rename('n').to_frame().assign(pct=lambda d:(100*d.n/75).round(1)).to_csv(R+'work_d/response_numbers/calc/ref75_habitat.csv')
print(pd.crosstab(hab,asm).to_string()); pd.crosstab(hab,asm).to_csv(R+'work_d/response_numbers/calc/ref75_habitat_by_asmtype.csv')
nn={g:[idx[x] for x in m.loc[g,'nn10'].split(';')] for g in m.index}
key={'CDM-1 (CDM-1)':'CDM-1','CDM-2 (CDM-2)':'CDM-2','Cirrhosis-1 (Cirrhosis-1)':'Cirrhosis-1','CDM-3 (CDM-3)':'CDM-3','CDM-4 (CDM-4)':'CDM-4','CDM-1 MAG':'CDM-1','CDM-3 MAG':'CDM-3'}
name={k:[g for g in m.index if v in g][0] for k,v in key.items()}
rng=np.random.default_rng(20261007); B=100000
perm=np.array([rng.permutation(vag) for _ in range(B)]); perm_s=np.tile(vag,(B,1))
for a in np.unique(asm):
    s=np.where(asm==a)[0]; perm_s[:,s]=rng.permuted(perm_s[:,s],axis=1)
rows=[]
p_hyp=comb(60,10)/comb(75,10)
for k,g in name.items():
    o=vag[nn[g]].sum(); rows.append(dict(set_=k,slots=10,distinct_refs=10,obs_vagina=int(o),expected=2.0,P_perm=(np.sum(perm[:,nn[g]].sum(1)<=o)+1)/(B+1),P_strat=(np.sum(perm_s[:,nn[g]].sum(1)<=o)+1)/(B+1),P_hypergeom=p_hyp))
def comb_test(lbl,ks):
    o=sum(vag[nn[name[k]]].sum() for k in ks); n=sum(perm[:,nn[name[k]]].sum(1) for k in ks); ns=sum(perm_s[:,nn[name[k]]].sum(1) for k in ks)
    u=set().union(*[nn[name[k]] for k in ks]); pu=comb(75-len(u),15)/comb(75,15)
    rows.append(dict(set_=lbl,slots=10*len(ks),distinct_refs=len(u),obs_vagina=int(o),expected=round(10*len(ks)*15/75,1),P_perm=(np.sum(n<=o)+1)/(B+1),P_strat=(np.sum(ns<=o)+1)/(B+1),P_hypergeom=pu))
five=list(key)[:5]
comb_test('5 targeted assemblies (50 slots; main figure)',five)
comb_test('7 genomes (70 slots)',list(key))
comb_test('3 lineage representatives CDM-1, Cirrhosis-1, CDM-4 (30 slots)',['CDM-1 (CDM-1)','Cirrhosis-1 (Cirrhosis-1)','CDM-4 (CDM-4)'])
comb_test('CDM-4 excluded: CDM-1 + Cirrhosis-1 (20 slots)',['CDM-1 (CDM-1)','Cirrhosis-1 (Cirrhosis-1)'])
comb_test('CDM-4 excluded: 4 targeted assemblies (40 slots)',five[:4])
d=pd.DataFrame(rows); pd.set_option('display.width',250); print(d.to_string()); d.to_csv(R+'work_d/response_numbers/calc/vagina_zero_tests.csv',index=False)
u=sorted(set().union(*[nn[name[k]] for k in five])); print('distinct refs in 50 slots:',len(u), pd.Series(hab[u]).value_counts().to_dict(), pd.Series(asm[u]).value_counts().to_dict())
for f in ['work_b/_crosscheck_M5ext/xc_placement_all.csv','work_b/M5/tables/T_M5_genome_set.csv']: print(hashlib.sha256(open(R+f,'rb').read()).hexdigest(),f)
