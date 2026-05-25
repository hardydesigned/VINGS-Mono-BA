Baseline mit skip n-tes frame generell und n-tes frame beim mapping


  ┌─────────────────┬────────┬─────────────┬──────────┐
  │     Config      │ Status │ KFs gemappt │ GPU-Peak │
  ├─────────────────┼────────┼─────────────┼──────────┤
  │ mapskip10       │ OK     │ 181         │ ~3 GB    │
  ├─────────────────┼────────┼─────────────┼──────────┤
  │ nofilter_skip10 │ FAIL   │ 148 (Crash) │ 9.6 GB   │
  ├─────────────────┼────────┼─────────────┼──────────┤
  │ mapskip9        │ OK     │ 201         │ ~3 GB    │
  ├─────────────────┼────────┼─────────────┼──────────┤
  │ skip9           │ OK     │ 201         │ 5.3 GB   │
  ├─────────────────┼────────┼─────────────┼──────────┤
  │ skip7           │ OK     │ 258         │ 5.7 GB   │
  ├─────────────────┼────────┼─────────────┼──────────┤
  │ skip6           │ FAIL   │ 209 (Crash) │ 9.7 GB   │
  └─────────────────┴────────┴─────────────┴──────────┘

  Gleiche Anzahl KFs, aber frame_skip braucht 2–3× so viel VRAM wie mapskip!

  Warum: Bei frame_skip=N sind aufeinanderfolgende Tracker-KFs N Frames auseinander — riesige Baseline-Sprünge,
  ständig neue Sichtbereiche. Folge:
  1. Storage Manager cpu2gpu holt nicht zurück (Posen weit voneinander) — wirkt eigentlich gut
  2. ABER: gpu2cpu Offload basiert auf Distanz zum aktuellen Frame. Bei sprunghaften Posen ist „aktuell" ständig
  woanders → andere KFs sind oft NICHT distanz>threshold weil sie zufällig wieder „nah" rotieren
  3. Hauptproblem: accum_thresh=0.92 Filter — bei großen Pose-Sprüngen ist pred_accum für neue Pixel niedrig → fast
  alle 40k Init-Punkte werden gesampelt → Gaussian-Wachstum explodiert 

rsync -av --remove-source-files --exclude='*sec0_5*' /home/philipp/Dokumente/Github/VINGS-Mono-BA/output/ /media/philipp/USB_STICK/ba/vings/output