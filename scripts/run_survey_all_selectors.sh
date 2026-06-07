#!/bin/bash
# Faehrt den KOMPLETTEN interval1-Cruise-Survey (mono+GPS) fuer JEDEN Frame-Selektor
# nacheinander, mergt pro Selektor eine survey_complete.ply, schreibt EINE Summary-Zeile
# je Selektor in eine gemeinsame CSV und raeumt nach jedem Selektor alles ausser der
# gemergten ply weg.
#
# Selektor-Params kommen aus den s1000_400f-Sweep-Gewinnern (beste psnr_ho, siehe
# docs/results/s1000_400f_results.csv). Pro Selektor uebernimmt gen_opt_cfg per
# --selector-from den frame_selector/gate_a-Block dieser Gewinner-Config.
#
# WICHTIG:
#  - SERIELL und LANG: ~9 Selektoren x voller Survey. Jeder Selektor ist unabhaengig;
#    bei Abbruch behalten fertige Selektoren ihre survey_complete.ply + Summary-Zeile.
#    Fortsetzen: SELECTORS="<rest>" scripts/run_survey_all_selectors.sh
#  - Survey laeuft non-metrisch (--no-ext, DROID-Frame). Die Selektor-Schwellen wurden
#    auf dem s1000-Slice getunt; bei abweichender Akzeptanzrate ggf. nachskalieren.
#  - aim_slam ist im s1000-Sweep an der VRAM-Wand gestorben (rc=137); hier faengt die
#    Retry-/Teil-ply-Logik das ab, das Ergebnis kann aber sparse/partiell sein.
#
# Env: SELECTORS="vista mm3dgs ..."  (Default: none + die 8 registrierten + two_gate_v2)
#      SUMMARY (Summary-CSV)  BASE (Root-Ordner)  + alle Env von run_interval1_survey.sh
#      (FRAMES STEP S0 S1 NUMKF CKPT MAXTRY)
set -u
cd /home/philipp/Dokumente/Github/VINGS-Mono-BA
SUMMARY=${SUMMARY:-docs/results/survey_selectors_summary.csv}
BASE=${BASE:-output/survey_selectors}
SWEEP=configs/local/amtown03/s1000_400f
mkdir -p "$BASE" "$(dirname "$SUMMARY")"

# Selektor -> Gewinner-Config ("" = none/Baseline, kein frame_selector)
declare -A WIN=(
  [none]=""
  [vista]="$SWEEP/vista/amtown03_s1000_400f_vista_g020.yaml"
  [mm3dgs]="$SWEEP/mm3dgs/amtown03_s1000_400f_mm3dgs_gap10.yaml"
  [adaptive_kf]="$SWEEP/adaptive_kf/amtown03_s1000_400f_adaptive_kf_sens3.yaml"
  [nurbs]="$SWEEP/nurbs/amtown03_s1000_400f_nurbs_orb400.yaml"
  [game]="$SWEEP/game/amtown03_s1000_400f_game_eta05.yaml"
  [orbslam]="$SWEEP/orbslam/amtown03_s1000_400f_orbslam_max15.yaml"
  [coko]="$SWEEP/coko/amtown03_s1000_400f_coko_st010.yaml"
  [aim]="$SWEEP/aim/amtown03_s1000_400f_aim.yaml"
  # Gesamtsieger des s1000-Sweeps (psnr_ho 15.22 / ate 4.82). Sein gate_a(A3)-GPS-Gate
  # greift jetzt, da die Survey-Base interval1 GPS pro Frame liefert (dataset.gps_csv).
  [two_gate_v2]="$SWEEP/two_gate_v2/amtown03_s1000_400f_two_gate_v2_a3_loose.yaml"
)
SELECTORS=${SELECTORS:-"none vista mm3dgs adaptive_kf nurbs game orbslam coko aim two_gate_v2"}

echo "############################################################"
echo "# SURVEY x SELEKTOREN: $SELECTORS"
echo "# Summary-CSV: $SUMMARY"
echo "# Root: $BASE"
echo "############################################################"

for SEL in $SELECTORS; do
  if [ -z "${WIN[$SEL]+x}" ]; then echo "!! unbekannter Selektor '$SEL' -- skip"; continue; fi
  CFG="${WIN[$SEL]}"
  if [ -n "$CFG" ] && [ ! -f "$CFG" ]; then
    echo "!! Gewinner-Config fehlt: $CFG -- skip $SEL"; continue
  fi
  OUTDIR="$BASE/$SEL"
  echo
  echo "============================================================"
  echo "= SELEKTOR: $SEL   cfg=${CFG:-<none>}   -> $OUTDIR"
  echo "============================================================"

  # Voller Survey fuer diesen Selektor. run_interval1_survey.sh erbt OUT/SELECTOR_FROM/
  # SURVEY_NAME/SUMMARY_CSV + die per-Segment-CSV liegt IM OUT-Ordner (wird mit aufgeraeumt).
  OUT="$OUTDIR" SELECTOR_FROM="$CFG" SURVEY_NAME="$SEL" SUMMARY_CSV="$SUMMARY" \
     CSV="$OUTDIR/segments.csv" \
     bash scripts/run_interval1_survey.sh

  # Aufraeumen: ALLES im OUT-Ordner weg ausser der gemergten ply.
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
echo "=========== ALLE SELEKTOREN FERTIG ==========="
echo "Summary (1 Zeile/Selektor): $SUMMARY"
echo "Gemergte plys:"
ls -la "$BASE"/*/survey_complete.ply 2>/dev/null || echo "  (keine)"
