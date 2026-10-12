#!/bin/bash
# M6 step 0 (parameterised copy of CDM-4/scripts/m6_00_inputs.sh): input checksum + read structure of the host-filtered FASTQ (read-only input).
# usage: m6_00_inputs.sh <sample>
set -euo pipefail
source ./work_b/M6/scripts/m6_lib.sh "$1"
step inputs "$N sha256sum $FQ > $O/logs/input_sha256.txt;
 zcat $FQ | $N awk 'NR%4==1{n++; if(\$0~/\/1\$/)a++; else if(\$0~/\/2\$/)b++; else c++; id=\$0; sub(/\/[12]\$/,\"\",id); if(id in s){p++; delete s[id]} else s[id]=1}
 NR%4==2{L=length(\$0); t+=L; if(L>mx)mx=L; if(mn==\"\"||L<mn)mn=L}
 END{printf \"reads=%d suffix1=%d suffix2=%d other=%d ids_seen_twice(pairs)=%d bases=%.0f meanlen=%.1f min=%d max=%d\n\",n,a,b,c,p,t,t/n,mn,mx}' > $O/logs/input_read_structure.txt"
# mate rescue (m6_recruit.py) assumes every read id ends in /1 or /2
grep -q ' other=0 ' $O/logs/input_read_structure.txt || { echo "read ids without /1,/2 suffix: $(cat $O/logs/input_read_structure.txt)" >&2; echo inputs_readname_check > $O/logs/current_step; exit 1; }
