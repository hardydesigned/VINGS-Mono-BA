# Überholte Skripte (interval1 per-Chunk-Sim3)

Diese Skripte gehörten zur **Option-C-Pipeline** (jeden Chunk non-metrisch im eigenen
DROID-Frame rekonstruieren, dann per-Chunk-Sim3 ins metrische Welt-Frame schieben).
Das per-Chunk-Sim3 WAR die Ursache der Nähte (separate DROID-Läufe = unabhängige
Gauges). Ersetzt durch die durchgehende mono+GPS-Pipeline:

  scripts/run_interval1_survey.sh   (durchgehende Segmente, sim3_unwarp --gps-csv)
  scripts/merge_survey.sh           (detilt_gps GPS-Boden-Leveling + clean)

Siehe docs/INTERVAL1_LIDAR_PIPELINE.md ("Update 2026-06-04 (II)").

| Datei | War | Ersetzt durch |
|---|---|---|
| run_interval1_optC.sh | per-Chunk-Sim3-Orchestrator | run_interval1_survey.sh |
| chunk_postfix.py | Sim3+Crop+Gate je Chunk | sim3_unwarp.py |
| chain_chunks.py | Rotations-Chaining der Chunks | (entfällt: ein durchgehender Lauf) |
| detilt_chain.py | sequentielles Overlap-De-Tilt (akkumuliert Fehler) | detilt_gps.py (globales GPS-Leveling) |
