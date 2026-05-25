  Was integriert ist

  - scripts/vings_utils/frame_selector.py — FrameSelector + FrameSelectorConfig mit from_config(...). Kaputter Header in deinem Original (_pose_is_redundant)
  repariert. Smoke-Test bestätigt: Run 1 = 26/60 akzeptiert, Replay = 0/60.
  - scripts/run.py — Import gesetzt, Selector in Runner.__init__ aus cfg['frame_selector'] (off by default), per-Frame-Decision an run.py:259 ersetzt
  mapper_kf_skip durch den Selector wenn enabled: true. PhaseTimer-Phase frame_select ist verdrahtet.
  - configs/local/vista/ — Basis-YAML + zwei Sweep-Varianten (g020, g040).
  - scripts/run_vista_experiments.sh — Sweep-Skript analog zu run_mapskip_experiments.sh.
