#!/bin/bash
# Mergt alle GT-verankerten AGZ-Survey-Segmente (output/exp_agz_survey/sN_gps.ply) zum
# finalen Deliverable. Analog zu merge_survey.sh, aber: AGZ-Segmente liegen bereits im
# GT-Posen-Frame (sim3_unwarp --gt-poses), also wird DIREKT gegen agz_poses_w2c.txt
# gecroppt (clean_ply --gt-poses berechnet C=-R^T t selbst) -- kein synthetisches
# gps_traj noetig wie bei interval1.
# Schritte: detilt_gps (Boden-Leveling) -> clean_ply (Floater + Footprint-Crop).
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
PY=/home/philipp/anaconda3/envs/vings/bin/python
GT=/home/philipp/Dokumente/datasets/agz/agz_0_10000/agz_poses_w2c.txt
OUT=${OUT:-output/exp_agz_survey}
mapfile -t GPSPLYS < <(ls "$OUT"/s*_gps.ply 2>/dev/null | sed -E 's/.*\/s([0-9]+)_gps\.ply/\1 &/' | sort -n | cut -d' ' -f2)
[ "${#GPSPLYS[@]}" -eq 0 ] && { echo "Keine sN_gps.ply in $OUT -- nichts zu mergen"; exit 1; }
echo "Merge ${#GPSPLYS[@]} Segmente: ${GPSPLYS[*]##*/}"

$PY scripts/eval/detilt_gps.py "${GPSPLYS[@]}" --out "$OUT/survey_raw.ply" \
   --cell 25 --clip-scale 2.0 --zband 120
$PY scripts/eval/clean_ply.py "$OUT/survey_raw.ply" --gt-poses "$GT" \
   --max-dist 160 --max-scale 1.5 --opacity-min 0.4 --max-z-spread 60 \
   --out "$OUT/survey_complete.ply" && rm -f "$OUT/survey_raw.ply"
df -h . | tail -1
echo "=========== FERTIG -> $OUT/survey_complete.ply ==========="
