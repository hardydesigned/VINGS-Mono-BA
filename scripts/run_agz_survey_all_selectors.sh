#!/bin/bash
# Faehrt den KOMPLETTEN AGZ-Survey fuer JEDEN Frame-Selektor nacheinander (Analog zu
# scripts/run_survey_all_selectors.sh fuer interval1). Pro Selektor: ein voller AGZ-Survey
# (run_agz_survey.sh) -> eine survey_complete.ply -> eine Summary-Zeile -> Aufraeumen.
#
# Selektor-Params kommen aus dem AGZ-s2675_400f-Sweep-Baum (1:1-Mirror der amtown-
# s1000_400f-Gewinner-Varianten, siehe project_agz_amtown_port). gen_opt_cfg uebernimmt
# per --selector-from den frame_selector/gate_a/gate_b-Block der Referenz-Config.
# HINWEIS: Das sind die amtown-Gewinner-VARIANTEN auf AGZ portiert -- ein eigener
# AGZ-Sweep zur Gewinner-Bestimmung wurde NICHT gefahren. Bei Bedarf WIN-Map anpassen.
#
# WICHTIG:
#  - SEHR LANG + SERIELL: ~10 Selektoren x voller AGZ-Survey (0-9500) = ~1-2 Tage.
#    Jeder Selektor unabhaengig; bei Abbruch behalten fertige ihre ply + Summary-Zeile.
#    Fortsetzen: SELECTORS="<rest>" bash scripts/run_agz_survey_all_selectors.sh
#  - Zum Verkuerzen: S0/S1 einschraenken (z.B. nur Cruise 7000+) oder SELECTORS filtern.
#  - Survey laeuft non-metrisch (--no-ext); Metrik via sim3_unwarp --gt-poses (kein RTK).
#  - Selektor-Schwellen sind amtown-getunt -> Akzeptanzrate auf AGZ ggf. abweichend.
#
# Env: SELECTORS  SUMMARY  BASE  + alle Env von run_agz_survey.sh (FRAMES STEP S0 S1 NUMKF CKPT MAXTRY)
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
SUMMARY=${SUMMARY:-docs/results/agz_survey_selectors_summary.csv}
BASE=${BASE:-output/agz_survey_selectors}
SWEEP=configs/local/agz/s2675_400f
mkdir -p "$BASE" "$(dirname "$SUMMARY")"

# Selektor -> Referenz-Config ("" = none/Baseline). Mirror der amtown-s1000-Gewinner.
declare -A WIN=(
  [none]=""
  [vista]="$SWEEP/vista/agz_s2675_400f_vista_g020.yaml"
  [mm3dgs]="$SWEEP/mm3dgs/agz_s2675_400f_mm3dgs_gap10.yaml"
  [adaptive_kf]="$SWEEP/adaptive_kf/agz_s2675_400f_adaptive_kf_sens3.yaml"
  [nurbs]="$SWEEP/nurbs/agz_s2675_400f_nurbs_orb400.yaml"
  [game]="$SWEEP/game/agz_s2675_400f_game_eta05.yaml"
  [orbslam]="$SWEEP/orbslam/agz_s2675_400f_orbslam_max15.yaml"
  [coko]="$SWEEP/coko/agz_s2675_400f_coko_st010.yaml"
  [aim]="$SWEEP/aim/agz_s2675_400f_aim.yaml"
  [two_gate_v2]="$SWEEP/two_gate_v2/agz_s2675_400f_two_gate_v2_a3_loose.yaml"
)
SELECTORS=${SELECTORS:-"none vista mm3dgs adaptive_kf nurbs game orbslam coko aim two_gate_v2"}

echo "############################################################"
echo "# AGZ SURVEY x SELEKTOREN: $SELECTORS"
echo "# Summary-CSV: $SUMMARY"
echo "# Root: $BASE"
echo "############################################################"

for SEL in $SELECTORS; do
  if [ -z "${WIN[$SEL]+x}" ]; then echo "!! unbekannter Selektor '$SEL' -- skip"; continue; fi
  CFG="${WIN[$SEL]}"
  if [ -n "$CFG" ] && [ ! -f "$CFG" ]; then
    echo "!! Referenz-Config fehlt: $CFG -- skip $SEL"; continue
  fi
  OUTDIR="$BASE/$SEL"
  echo
  echo "============================================================"
  echo "= SELEKTOR: $SEL   cfg=${CFG:-<none>}   -> $OUTDIR"
  echo "============================================================"

  OUT="$OUTDIR" SELECTOR_FROM="$CFG" SURVEY_NAME="$SEL" SUMMARY_CSV="$SUMMARY" \
     CSV="$OUTDIR/segments.csv" \
     bash scripts/run_agz_survey.sh

  if [ -f "$OUTDIR/survey_complete.ply" ]; then
    find "$OUTDIR" -mindepth 1 ! -name survey_complete.ply -delete 2>/dev/null
    SZ=$(du -h "$OUTDIR/survey_complete.ply" | cut -f1)
    echo "# [$SEL] aufgeraeumt -> nur $OUTDIR/survey_complete.ply ($SZ)"
  else
    echo "# [$SEL] WARN: keine survey_complete.ply -- $OUTDIR NICHT aufgeraeumt (Diagnose behalten)"
  fi
  df -h . | tail -1
done

echo
echo "=========== ALLE AGZ-SELEKTOREN FERTIG ==========="
echo "Summary (1 Zeile/Selektor): $SUMMARY"
ls -la "$BASE"/*/survey_complete.ply 2>/dev/null || echo "  (keine)"
