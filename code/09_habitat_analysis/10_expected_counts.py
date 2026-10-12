"""T17: re-computation of the expected counts for the habitat sensitivity analysis. Uses only the existing Jaccard distance matrix (M5_ext MAIN); no analysis is re-run.
(1) Panel without the 16 gut-derived MAG references (59 references, 17 gut): expected = 10 x 17/59. Observed / expected / hypergeometric P per genome and per lineage, label-permutation P.
(2) the original 75-reference panel, (3) comparison with the expected counts in the full pool of 1,185 (skani ANI), (4) the balanced panel of T7b aggregated by lineage."""
import pandas as pd, numpy as np, json, hashlib
from math import comb
R='./'; O=R+'panels/Fig6_final4/T17/'
J=pd.read_csv(R+'work_b/M5_ext/tables/M5_MAIN_jaccard.csv',index_col=0)
gs=pd.read_csv(R+'work_b/M5/tables/T_M5_genome_set.csv'); gs=gs[gs.species.str.contains('crispatus')&(gs.role=='reference')].set_index('genome')
refs=[g for g in J.index if g in gs.index]; hab=np.array(gs.loc[refs,'habitat'].astype(str).tolist(),dtype=object); asm=np.array(gs.loc[refs,'asm_type'].astype(str).tolist(),dtype=object)
pat={'CDM-1':'CDM-1_Lcrispatus_targeted','CDM-2':'CDM-2_Lcrispatus_targeted','Cirrhosis-1':'Cirrhosis-1_Lcrispatus_targeted','CDM-3':'CDM-3_Lcrispatus_targeted','CDM-4':'CDM-4_Lcrispatus_targeted'}
lin={'CDM-1':1,'CDM-2':1,'Cirrhosis-1':2,'CDM-3':2,'CDM-4':3}
D=J.loc[list(pat.values()),refs].values.astype(float); k=10
def hyp(x,K,N,n=10): return sum(comb(K,i)*comb(N-K,n-i) for i in range(x,n+1))/comb(N,n)
def panel(keep,name,B=100000,seed=20261008):
    keep=np.array(keep); h=hab[keep]; a=asm[keep]; d=D[:,keep]; N=len(keep); K=int((h=='human gut').sum()); V=int((h=='vagina').sum())
    idx=np.argsort(d,axis=1,kind='stable')[:,:k]; g=(h[idx]=='human gut').sum(1); v=(h[idx]=='vagina').sum(1)
    rng=np.random.default_rng(seed); isg=(h=='human gut'); perm=np.array([rng.permutation(isg) for _ in range(B)])
    rows=[]
    for j,lab in enumerate(pat):
        rows.append(dict(panel=name,N_refs=N,gut_refs=K,vagina_refs=V,unit='genome',label=lab,lineage=lin[lab],slots=10,observed_gut=int(g[j]),expected_gut=round(10*K/N,2),P_hypergeom=hyp(int(g[j]),K,N),P_perm=(np.sum(perm[:,idx[j]].sum(1)>=g[j])+1)/(B+1),observed_vagina=int(v[j]),expected_vagina=round(10*V/N,2),nn_asm=str(pd.Series(a[idx[j]]).value_counts().to_dict())))
    for L,members in {1:[0,1],2:[2,3],3:[4]}.items():
        t=int(g[members].sum()); n_=perm[:,idx[members].ravel()].sum(1); u=len(set(idx[members].ravel()))
        rows.append(dict(panel=name,N_refs=N,gut_refs=K,vagina_refs=V,unit='lineage (all genomes, post hoc sum)',label=f'Lineage {L}',lineage=L,slots=10*len(members),observed_gut=t,expected_gut=round(10*len(members)*K/N,2),P_hypergeom=np.nan,P_perm=(np.sum(n_>=t)+1)/(B+1),observed_vagina=int(v[members].sum()),expected_vagina=round(10*len(members)*V/N,2),nn_asm=f'distinct refs {u}'))
    return rows
allidx=np.arange(75); noMAGgut=np.array([i for i in allidx if not (hab[i]=='human gut' and asm[i]=='MAG')]); noMAG=np.array([i for i in allidx if asm[i]!='MAG'])
rows=panel(allidx,'A. original 75 references')+panel(noMAGgut,'B. gut MAG references removed (59)')+panel(noMAG,'C. all MAG references removed (54)')
d=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(d.drop(columns=['nn_asm','vagina_refs']).to_string()); d.to_csv(O+'T17_expected_recalc_panels.csv',index=False)
print('composition:'); 
for nm,ix in [('75',allidx),('59 (gut MAG removed)',noMAGgut),('54 (all MAG removed)',noMAG)]: print(nm,pd.Series(hab[ix]).value_counts().to_dict())
t7=pd.read_csv(R+'work_d/panelC_sensitivity/tables/T7_full1185_skani_top10_habitat.csv'); t7=t7[t7.k==10]
m={'CDM-1 (CDM-1)':'CDM-1','CDM-2 (CDM-2)':'CDM-2','Cirrhosis-1 (Cirrhosis-1)':'Cirrhosis-1','CDM-3 (CDM-3)':'CDM-3','CDM-4 (CDM-4)':'CDM-4'}
t7=t7[t7.genome.isin(m)].assign(label=lambda x:x.genome.map(m)); Np,Kg,Kv=1185,36,975
f=[]
for r in t7.itertuples():
    f.append(dict(label=r.label,lineage=lin[r.label],observed_gut=r.gut,expected_gut_36of1185=round(10*Kg/Np,2),P_gut_ge_obs=hyp(int(r.gut),Kg,Np),observed_vagina=r.vagina,expected_vagina=round(10*Kv/Np,2),P_vagina_ge_obs=hyp(int(r.vagina),Kv,Np),P_vagina_le_obs=1-hyp(int(r.vagina)+1,Kv,Np) if r.vagina<10 else 1.0,nearest_habitat=r.nearest_habitat,nearest_ANI=r.nearest_ANI))
f=pd.DataFrame(f); print(f.to_string()); f.to_csv(O+'T17_full1185_expected.csv',index=False)
b=pd.read_csv(R+'work_d/panelC_sensitivity/tables/T7b_balanced_panel_per_genome.csv'); b=b[b.genome.str.contains('CDM-|Cirrhosis-')&b.variant.isin(['main_random15','asmtype_matched_all_distinct'])]
b['label']=b.genome.str.replace(r' \(.*','',regex=True); b['lineage']=b.label.map(lin)
bb=b[['variant','lineage','label','gut75','median','p2_5','p97_5','expected','median_hypergeom_P','frac_P_lt_0.05','frac_vagina0']].sort_values(['variant','lineage']); print(bb.to_string()); bb.to_csv(O+'T17_T7b_by_lineage.csv',index=False)
for x in ['work_b/M5_ext/tables/M5_MAIN_jaccard.csv','work_b/M5/tables/T_M5_genome_set.csv','work_d/panelC_sensitivity/tables/T7_full1185_skani_top10_habitat.csv','work_d/panelC_sensitivity/tables/T7b_balanced_panel_per_genome.csv']: print(hashlib.sha256(open(R+x,'rb').read()).hexdigest()[:16],x)
