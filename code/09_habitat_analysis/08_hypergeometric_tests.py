"""T17 item 1 (a)(b)(c-check): habitat composition of the 1,185 public L. crispatus genomes, composition and selection of the 75 references,
independent hypergeometric check of the 'zero vagina neighbours' probabilities. Read-only on inputs; writes only datasource/ and log/."""
import pandas as pd, numpy as np, hashlib, shutil
from math import comb
R='./'; O=R+'work_d/response_numbers/datasource/'
E=R+'refs/rescue/E/'
rc=pd.read_csv(E+'T_E_habitat_reclass.csv'); assert len(rc)==1185
cl=pd.read_csv(E+'T_E_dedup_clusters.csv')                    # ANI>=99.9 single-linkage clusters of universe MAIN (748)
uc=pd.read_csv(E+'T_E_universe_counts.csv').set_index('universe')
gs=pd.read_csv(R+'work_b/M5/tables/T_M5_genome_set.csv'); gs=gs[gs.species.str.contains('crispatus')]
ref=gs[gs.role=='reference']; assert len(ref)==75
# ---- (a) 1,185 composition
order=['vagina','human gut','urinary','animal/poultry','food/environment','oral','other human body site','human, site unknown','unknown','gut, host not stated']
a=rc.new_label.value_counts().reindex(order).rename('n_1185').to_frame(); a['pct_1185']=(100*a.n_1185/1185).round(1)
a['n_MAIN_748_NCBIcheckm']=[int(uc.loc['MAIN',f'n[{k}]']) for k in order]
a['n_ANI999_cluster_reps_115']=[int(uc.loc['S2_dedup',f'n[{k}]']) for k in order]
r75=ref.habitat.value_counts(); a['n_ref75']=[int(r75.get(k,0)) for k in order]; a['pct_ref75']=(100*a.n_ref75/75).round(1)
oldmap={'vagina':795,'human gut':41,'urinary':44,'animal/poultry':74,'food/environment':179,'oral':5}   # STEP_50 old keyword labels (refs/STEP_50 section 9)
a['old_STEP50_label_n(not same classes)']=[oldmap.get(k,'') for k in order]
a.index.name='habitat_class_T_E_habitat_reclass'; a.to_csv(O+'T17_item1_habitat_1185.csv')
print(a.to_string()); print('sum',a.n_1185.sum(),a.n_ref75.sum())
amb=int(rc.new_label.isin(['unknown','human, site unknown','gut, host not stated']).sum()); print('ambiguous (unknown+human site unknown+gut host not stated)=',amb,'; other human body site=',int((rc.new_label=='other human body site').sum()))
print('rule_fired:',rc.rule_fired.value_counts().to_dict()); print('menstrual cup (->vagina):',int((rc.detail=='menstrual cup').sum()),'; infant-keyword-only human gut:',int(rc.human_gut_via_infant_keyword_only.sum()))
# ---- (b) how 75 were drawn
pool=cl.rep_label.value_counts().to_dict(); print('cluster reps by rep_label (MAIN):',pool, 'n clusters',len(cl))
sel=ref.selection.groupby(ref.habitat).first()
rows=[]
spec={'human gut':dict(labelled=36,after_filter=34,pool=33,rule='ALL human-gut genomes passing direct CheckM v1.2.4 (>=90/<5): 34 pass (2 fail contamination 5.84: GCA_060581075.1, GCA_060583275.1); one sequence-identical pair collapsed (GCF_902386155.1 = GCF_000497065.1) -> 33; no ANI de-duplication',draw='all (no random draw)'),
'vagina':dict(labelled=975,after_filter=618,pool=pool['vagina'],rule='MAIN (NCBI CheckM >=90/<5) = 618 genomes -> 48 ANI>=99.9% single-linkage cluster representatives (T_E_dedup_clusters, rep_label=vagina)',draw='random 15 of 48; numpy default_rng(20261005) choice(replace=False), first of three draws (vagina -> urinary -> animal)'),
'urinary':dict(labelled=53,after_filter=40,pool=pool['urinary'],rule='MAIN 40 -> 11 cluster reps',draw='random 10 of 11; same generator, second draw'),
'animal/poultry':dict(labelled=80,after_filter=52,pool=pool['animal/poultry'],rule='MAIN 52 -> 37 cluster reps',draw='random 10 of 37; same generator, third draw'),
'oral':dict(labelled=5,after_filter=4,pool=4,rule='direct CheckM pass 4 (GCF_002811165.1 fails 98.38/14.22); ANI>=99.9% clusters within habitat = 4',draw='all 4'),
'food/environment':dict(labelled=7,after_filter=5,pool=3,rule='direct CheckM pass 5 (2 fail completeness); ANI>=99.9% clusters within habitat = 3',draw='all 3')}
for k,v in spec.items():
    n=int(r75[k]); rows.append(dict(habitat=k,n_labelled_1185=v['labelled'],n_after_quality_filter=v['after_filter'],n_pool_after_dedup=v['pool'],n_selected=n,pct_of_ref75=round(100*n/75,1),pct_of_1185=round(100*v['labelled']/1185,1),selection_rule=v['rule'],draw=v['draw'],selection_column_in_T_M5_genome_set=sel[k]))
    assert n==(v['pool'] if k in('human gut','oral','food/environment') else {'vagina':15,'urinary':10,'animal/poultry':10}[k]),k
s=pd.DataFrame(rows); s.to_csv(O+'T17_item1_ref75_selection.csv',index=False); print(s[['habitat','n_labelled_1185','n_after_quality_filter','n_pool_after_dedup','n_selected','pct_of_ref75','pct_of_1185']].to_string())
# the 15 vagina refs really are cluster reps of the vagina pool
vag=set(ref[ref.habitat=='vagina'].genome); vp=set(cl[cl.rep_label=='vagina'].dedup_rep); assert vag<=vp, 'vagina refs not all in pool'
for h,lab in [('urinary','urinary'),('animal/poultry','animal/poultry')]: assert set(ref[ref.habitat==h].genome)<=set(cl[cl.rep_label==lab].dedup_rep)
# ---- (c) independent hypergeometric check
calc=pd.read_csv(R+'work_d/response_numbers/calc/vagina_zero_tests.csv')
chk=[]
def hyp0(n_distinct,N=75,K=15): return comb(N-n_distinct,K)/comb(N,K)        # P(none of K vagina labels falls in a fixed set of n_distinct refs)
tests=[('single genome (10 distinct refs)',10,'CDM-1 (CDM-1)'),('5 targeted assemblies (20 distinct refs)',20,'5 targeted assemblies (50 slots; main figure)'),('7 genomes (21 distinct)',21,'7 genomes (70 slots)'),('3 lineage reps (19 distinct)',19,'3 lineage representatives CDM-1, Cirrhosis-1, CDM-4 (30 slots)'),('CDM-4 excluded, 4 assemblies (20 distinct)',20,'CDM-4 excluded: 4 targeted assemblies (40 slots)'),('CDM-4 excluded, CDM-1+Cirrhosis-1 (19 distinct)',19,'CDM-4 excluded: CDM-1 + Cirrhosis-1 (20 slots)')]
for lab,nd,key in tests:
    own=hyp0(nd); alt=comb(60,nd)/comb(75,nd)   # own = C(75-nd,15)/C(75,15); alt = C(60,nd)/C(75,nd) (same probability by hypergeometric symmetry)
    tab=calc.loc[calc.set_==key,'P_hypergeom'].iloc[0]
    chk.append(dict(set_=lab,n_distinct_refs=nd,P_own=own,P_alt_form_C60_nd_over_C75_nd=alt,P_hypergeom_in_calc_table=tab,abs_diff=abs(own-tab),match=(abs(own-tab)<1e-9)and(abs(alt-tab)<1e-9)))
print('C(60,10)/C(75,10)=',comb(60,10)/comb(75,10),' C(55,15)/C(75,15)=',comb(55,15)/comb(75,15))
c=pd.DataFrame(chk); c.to_csv(O+'T17_item1c_hypergeom_check.csv',index=False); print(c.to_string()); assert c.match.all()
# the habitat labels 33/15/10/10/4/3 and MAG count per habitat (for 'no MAG among vagina refs')
print(pd.crosstab(ref.habitat,ref.asm_type).to_string())
