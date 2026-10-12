#!/usr/bin/env python
"""M5_ext step 5: placement of patient genomes. Usage: m5ext_50_placement.py <MAIN|S1|S2a|M5check>
Core = work_b/M5/scripts/m5_50_placement.py unchanged in logic (nearest centroid axes 1-2 / 1-10; kNN k=5,10 among
REFERENCES ONLY; 10,000 label permutations, default_rng(20261005) re-seeded per test; unstratified + within asm type).
Added (design section 5, descriptive): n clonal (ANI>=99.5) neighbours, D1 (clonal refs removed), D2 (non-MAG refs only).
MAIN also runs S2b = leave-others-out (75 refs + one patient genome, re-filtered from the MAIN full matrix, no re-clustering).
M5check = run the same code on the ORIGINAL M5 tables and assert the published numbers (self-check)."""
import sys, numpy as np, pandas as pd
run = sys.argv[1]; R = '.'; X = R + '/work_b/M5_ext'; SEED = 20261005; NP = 10000

def perms(lab, strata, stratified):
    rng = np.random.default_rng(SEED)
    for _ in range(NP):
        if not stratified: yield rng.permutation(lab)
        else:
            p = lab.copy()
            for s in np.unique(strata):
                i = np.where(strata == s)[0]; p[i] = lab[rng.permutation(i)]
            yield p

def knn(m, ref, dj, ani, k, tag):
    """kNN among the reference genomes in `ref` only; permutation of the labels of `ref`."""
    lab = ref.habitat.values; strata = ref.asm_type.values
    order = sorted(ref.index, key=lambda a: (dj[a], a)); nn = order[:k]; idx = [ref.index.get_loc(a) for a in nn]
    obs = int((lab[idx] == 'human gut').sum()); comp = pd.Series(lab[idx]).value_counts()
    r = dict(method=tag, k=k, observed='; '.join(f'{h} {n}' for h, n in comp.items()), n_gut_among_k=obs, statistic=obs,
             detail=' '.join(f'{a}[{ref.habitat[a]}|{ref.asm_type[a]}|J={dj[a]:.3f}|ANI={ani[a]:.2f}]' for a in nn),
             expected_gut=round(k * (lab == 'human gut').mean(), 2), n_candidates=len(ref), n_gut_candidates=int((lab == 'human gut').sum()),
             n_neighbours_ANI_ge_99_5=int(sum(ani[a] >= 99.5 for a in nn)), n_gut_neighbours_ANI_ge_99_5=int(sum(ani[a] >= 99.5 and ref.habitat[a] == 'human gut' for a in nn)),
             neighbours_asm=' '.join(f'{t} {n}' for t, n in pd.Series(strata[idx]).value_counts().items()))
    for st in (False, True):
        null = np.array([(p[idx] == 'human gut').sum() for p in perms(lab, strata, st)])
        r['P_perm_within_asm_type' if st else 'P_perm'] = (1 + (null >= obs).sum()) / (NP + 1)
    return r

def place(co, D, ani, extras=True):
    ref = co[co.role == 'reference']; pats = co.index[co.role != 'reference']
    lab = ref.habitat.values; strata = ref.asm_type.values
    levels = [h for h, n in ref.habitat.value_counts().items() if n >= 3]; assert 'human gut' in levels
    def cent_stat(x, Xr, l):
        d = {h: np.linalg.norm(x - Xr[l == h].mean(0)) for h in levels}
        return d['human gut'] - min(v for h, v in d.items() if h != 'human gut'), d
    rows = []
    for m in pats:
        dj = D.loc[m, ref.index]; am = ani.loc[m, ref.index]; order = sorted(ref.index, key=lambda a: (dj[a], a))
        base = dict(genome=m, n_ref=len(ref), n_gut_ref=int((lab == 'human gut').sum()), nearest_ref=order[0], nearest_ref_habitat=ref.habitat[order[0]],
                    nearest_ref_jaccard=round(dj[order[0]], 4), nearest_ref_ANI=am[order[0]], max_ANI_ref=am.idxmax(), max_ANI=am.max(),
                    n_refs_ANI_ge_99_5=int((am >= 99.5).sum()), refs_ANI_ge_99_5_habitats='; '.join(f'{h} {n}' for h, n in ref.habitat[am >= 99.5].value_counts().items()))
        for ax in (2, 10):
            cols = [f'PCo{i}' for i in range(1, ax + 1)]; Xr = ref[cols].values; x = co.loc[m, cols].values.astype(float)
            obs, d = cent_stat(x, Xr, lab)
            r = dict(base, method=f'nearest centroid, PCoA axes 1-{ax}', k='', observed=min(d, key=d.get), n_gut_among_k='',
                     detail='; '.join(f'{h}={v:.4f}' for h, v in sorted(d.items(), key=lambda t: t[1])), statistic=round(obs, 5), expected_gut='')
            for st in (False, True):
                null = np.array([cent_stat(x, Xr, p)[0] for p in perms(lab, strata, st)])
                r['P_perm_within_asm_type' if st else 'P_perm'] = (1 + (null <= obs + 1e-12).sum()) / (NP + 1)
            rows.append(r)
        for k in (5, 10): rows.append(dict(base, **knn(m, ref, dj, am, k, 'kNN by Jaccard')))
        if extras:
            rows.append(dict(base, **knn(m, ref[(am < 99.5).values], dj, am, 10, 'D1 kNN, refs with ANI>=99.5 removed')))
            rows.append(dict(base, **knn(m, ref[(ref.asm_type != 'MAG').values], dj, am, 10, 'D2 kNN, non-MAG refs only')))
    t = pd.DataFrame(rows); v = {}
    for m in pats:
        x = t[t.genome == m]; c2 = x[x.method.str.endswith('1-2')].observed.iloc[0]; p10 = x[(x.method == 'kNN by Jaccard') & (x.k == 10)].P_perm.iloc[0]
        v[m] = bool((c2 == 'human gut') and (p10 < 0.05))
    t['placed_with_gut'] = t.genome.map(v)
    assert all((t.P_perm > 0) & (t.P_perm <= 1))
    return t

def jaccard(pa):            # pa: families x genomes 0/1
    B = pa.values.T.astype(float); inter = B @ B.T; s = B.sum(1); return pd.DataFrame(1 - inter / (s[:, None] + s[None, :] - inter), index=pa.columns, columns=pa.columns)
def cmdscale(D, k=10):      # classical MDS = R cmdscale
    n = len(D); J = np.eye(n) - 1 / n; w, v = np.linalg.eigh(-0.5 * J @ (D.values ** 2) @ J); o = np.argsort(w)[::-1][:k]
    return pd.DataFrame(v[:, o] * np.sqrt(w[o]), index=D.index, columns=[f'PCo{i}' for i in range(1, k + 1)])
SHOW = ['genome', 'method', 'k', 'observed', 'n_gut_among_k', 'expected_gut', 'P_perm', 'P_perm_within_asm_type', 'n_neighbours_ANI_ge_99_5', 'placed_with_gut']
pd.set_option('display.width', 280); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 500)

if run == 'M5check':
    M = R + '/work_b/M5'; co = pd.read_csv(M + '/tables/M5_lc_pcoa.csv').set_index('genome'); D = pd.read_csv(M + '/tables/M5_lc_jaccard.csv', index_col=0); ani = pd.read_csv(M + '/work/ani_lc.csv', index_col=0)
    t = place(co, D, ani); old = pd.read_csv(M + '/tables/T_M5_placement_lc.csv')
    a = t[t.method.isin(['kNN by Jaccard']) | t.method.str.startswith('nearest')].reset_index(drop=True)
    assert len(a) == len(old) and (a.genome.values == old.MAG.values).all()
    assert np.allclose(a.P_perm, old.P_perm) and np.allclose(a.P_perm_within_asm_type, old.P_perm_within_asm_type) and (a.observed.values == old.observed.values).all()
    x = a[a.k == 10].set_index('genome'); assert x.n_gut_among_k.tolist() == [8, 9] and abs(x.P_perm.iloc[0] - 0.018898) < 1e-5 and abs(x.P_perm.iloc[1] - 0.0023) < 1e-5
    # own Jaccard + cmdscale vs vegan (used by S2b)
    pa = pd.read_csv(M + '/tables/M5_lc_presence_filtered.csv.gz', index_col=0); Dn = jaccard(pa); pc = cmdscale(Dn)
    assert np.abs(Dn.loc[D.index, D.columns].values - D.values).max() < 1e-5
    assert np.allclose(np.abs(pc.loc[co.index].values), np.abs(co[pc.columns].values), atol=1e-4), 'cmdscale mismatch'
    t.to_csv(X + '/tables/T_M5ext_placement_M5check.csv', index=False)
    print('M5check OK: same code reproduces all 8 rows of work_b/M5 T_M5_placement_lc.csv (P identical); numpy Jaccard/cmdscale = vegan/cmdscale')
    print(t[SHOW].to_string()); sys.exit()

co = pd.read_csv(f'{X}/tables/M5_{run}_pcoa.csv').set_index('genome'); D = pd.read_csv(f'{X}/tables/M5_{run}_jaccard.csv', index_col=0); ani = pd.read_csv(f'{X}/work/ani_{run}.csv', index_col=0)
t = place(co, D, ani); t.insert(0, 'run', run); t.to_csv(f'{X}/tables/T_M5ext_placement_{run}.csv', index=False)
print(t[SHOW].to_string())
print(t[['genome', 'nearest_ref', 'nearest_ref_habitat', 'nearest_ref_jaccard', 'max_ANI_ref', 'max_ANI', 'n_refs_ANI_ge_99_5', 'refs_ANI_ge_99_5_habitats']].drop_duplicates().to_string())
for r in t[t.k == 10].itertuples(): print(r.genome, '|', r.method, '|', r.detail)
if run == 'MAIN':   # S2b leave-others-out
    full = pd.read_csv(f'{X}/tables/M5_MAIN_presence_full.csv.gz', index_col=0); meta = co[['habitat', 'role', 'asm_type']]
    refs = list(co.index[co.role == 'reference']); out = []
    for m in co.index[co.role != 'reference']:
        pa = full[refs + [m]]; k = pa.sum(1); pa = pa[(k >= 2) & (k <= pa.shape[1] - 2)]
        Dn = jaccard(pa); c = cmdscale(Dn).join(meta); x = place(c, Dn, ani); x.insert(0, 'n_families_filtered', len(pa)); out.append(x)
    s = pd.concat(out); s.insert(0, 'run', 'S2b_leave_others_out'); s.to_csv(f'{X}/tables/T_M5ext_placement_S2b.csv', index=False)
    print('\n##### S2b leave-others-out'); print(s[SHOW + ['n_families_filtered']].to_string())
    for r in s[s.k == 10].itertuples(): print(r.genome, '|', r.method, '|', r.detail)
