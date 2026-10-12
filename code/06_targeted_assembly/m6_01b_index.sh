#!/bin/bash
# M6 step 1b: bowtie2 index of panel+decoy (first attempt in m6_01_panel.sh failed with rc=127: `python` not on PATH; PATH fixed here).
set -euo pipefail
export PATH=$CONDA_ENVS/mapping_env/bin:$PATH
O=./work_b/M6/CDM-4; W=$O/work
nice -n 19 ionice -c3 bowtie2-build --threads 4 --seed 20261005 $W/panel_decoy.fa $W/panel_decoy > $O/logs/m6_01b_bt2build.log 2>&1
