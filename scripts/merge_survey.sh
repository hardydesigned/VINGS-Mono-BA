#!/bin/bash
# Mergt ALLE vorhandenen GPS-unwarpten Survey-Segmente (output/exp_interval1_survey/sN_gps.ply)
# zum finalen Deliverable -- unabhängig davon, welche Segmente gerade neu gefahren wurden.
# Schritte: GPS-Bahn-Datei (für korrekten Spatial-Crop) -> detilt_gps (GPS-Boden-Leveling)
# -> clean_ply (Floater). Siehe docs/INTERVAL1_LIDAR_PIPELINE.md "Update 2026-06-04 (II)".
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
PY=/home/philipp/anaconda3/envs/vings/bin/python
RTK=/home/philipp/Dokumente/datasets/interval1_AMtown03/rtk_positions_raw.csv
OUT=${OUT:-output/exp_interval1_survey}
# alle Segmente in Flugreihenfolge (numerisch nach Start sortiert)
mapfile -t GPSPLYS < <(ls "$OUT"/s*_gps.ply 2>/dev/null | sed -E 's/.*\/s([0-9]+)_gps\.ply/\1 &/' | sort -n | cut -d' ' -f2)
[ "${#GPSPLYS[@]}" -eq 0 ] && { echo "Keine sN_gps.ply in $OUT -- nichts zu mergen"; exit 1; }
echo "Merge ${#GPSPLYS[@]} Segmente: ${GPSPLYS[*]##*/}"

# GPS-Bahn-Datei für clean_ply-Spatial-Crop (GT-Posen liegen in ANDEREM Frame als die GPS-Survey!)
$PY - "$RTK" "$OUT/gps_traj_w2c.txt" <<'PYEOF'
import sys, numpy as np
gp=np.genfromtxt(sys.argv[1],delimiter=",",names=True); o=np.argsort(gp["headerstamp"])
t=gp["headerstamp"][o]; e=gp["easting"][o]-gp["easting"][o][0]
n=gp["northing"][o]-gp["northing"][o][0]; al=gp["alt"][o]-gp["alt"][o][0]
Z=np.zeros_like(t); O=np.ones_like(t)
np.savetxt(sys.argv[2],np.stack([t,-e,-n,-al,Z,Z,Z,O],1),fmt="%.6f")
PYEOF

$PY scripts/eval/detilt_gps.py "${GPSPLYS[@]}" --out "$OUT/survey_raw.ply" \
   --cell 25 --clip-scale 2.0 --zband 120
$PY scripts/eval/clean_ply.py "$OUT/survey_raw.ply" --gt-poses "$OUT/gps_traj_w2c.txt" \
   --max-dist 160 --max-scale 1.5 --opacity-min 0.4 --max-z-spread 60 \
   --out "$OUT/survey_complete.ply" && rm -f "$OUT/survey_raw.ply"
df -h . | tail -1
echo "=========== FERTIG -> $OUT/survey_complete.ply ==========="
