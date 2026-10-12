#!/bin/bash
cd ./refs/vp_refs
[ -e checkm_humangut.tsv ] && { echo "output exists"; exit 1; }
export PATH=$CONDA_ENVS/checkm_env/bin:$PATH; export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
nice -n 19 checkm lineage_wf -x fna -t 8 --tab_table -f checkm_humangut.tsv checkm_bins checkm_run > checkm_run.log 2>&1
echo "exit=$?" >> checkm_run.log
