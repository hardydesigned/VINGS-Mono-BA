#!/bin/bash
# Lücken-Nachfüller: fährt einzelne, am VRAM-Spike gestorbene Segmente erneut mit
# robusteren Settings (num_keyframe 2 = kleinerer Densification-Spike, mehr Retries)
# und OHNE finalen Merge. Danach scripts/merge_survey.sh aufrufen, um den Survey neu
# zu bauen. Dünner Wrapper um run_interval1_survey.sh -- gleiche Messung (CSV-Zeile pro
# Segment). Default-Lücken: 1000 1600 4600.
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
STARTS=("$@"); [ "${#STARTS[@]}" -eq 0 ] && STARTS=(1000 1600 4600)
NUMKF=2 MAXTRY=3 MERGE=0 bash scripts/run_interval1_survey.sh "${STARTS[@]}"
echo "Fertig. Survey neu bauen mit: bash scripts/merge_survey.sh"
