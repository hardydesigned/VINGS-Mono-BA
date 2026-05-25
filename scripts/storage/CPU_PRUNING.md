# CPU-seitiges Gaussian-Pruning im StorageManager

## Problem

Der `StorageManager` verschiebt Gaussians, die weiter als `distance_threshold` von der
aktuellen Kameraposition entfernt sind, vom GPU-Mapper auf die CPU. Dieser Transfer ist
einseitig akkumulierend: Gaussians kommen auf der CPU hinzu, werden dort aber nie wieder
entfernt. Über einen langen Run (z. B. 800 Frames) wächst der CPU-RAM daher kontinuierlich,
bis der Linux-OOM-Killer den Prozess mit SIGKILL (exit code 137) beendet.

---

## GPU-seitiges Pruning als Referenz: `storage_control()`

Auf der GPU-Seite existiert in `gaussian_model.py` die Methode `storage_control()`, die
alle **4 Keyframes** aufgerufen wird. Sie funktioniert so:

1. **Re-Rendering aller aktuellen Keyframes** mit dem aktuellen Mapper-Zustand.
2. **Gradienten von `_zeros`** (den Score-Tensoren) werden als frische Importance-Scores
   akkumuliert: `temp_importance_scores`.
   - Diese Scores messen, wie stark jeder Gaussian zum Rendering-Fehler beigetragen hat.
   - Hoher Score → Gaussian ist sichtbar und beeinflusst das Ergebnis stark.
   - Score ≈ 0 → Gaussian ist unsichtbar oder redundant.
3. **Prune-Maske:**
   ```python
   prune_mask = (temp_importance_scores > 0.05) & (temp_importance_scores < 0.8) & (~stable_mask)
   ```
   - **`~stable_mask`**: Nur unstabile Gaussians werden entfernt. Stabile Gaussians
     (als konvergiert markiert) sind immer geschützt.
   - **`> 0.05`**: Völlig unsichtbare Gaussians (`≤ 0.05`) werden **bewusst behalten** —
     auf der GPU können sie bei künftigen Kamerabewegungen sichtbar werden.
   - **`< 0.8`**: Sehr wichtige Gaussians (`≥ 0.8`) werden ebenfalls behalten.
   - **Mittlerer Bereich** (0.05–0.8): unstabile Gaussians mit mittlerer Sichtbarkeit
     sind wahrscheinlich rauschig oder redundant → Prune.

---

## CPU-seitige Implementierung: `prune_cpu_gaussians()`

### Warum ist eine direkte Übertragung nicht möglich?

| Aspekt | GPU (`storage_control`) | CPU (`prune_cpu_gaussians`) |
|---|---|---|
| Optimizer-State | Adam-State muss synchron gepruned werden (`prune_tensors_from_optimizer`) | Kein Optimizer, kein `nn.Parameter` → einfaches Boolean-Indexing |
| Importance-Scores | Frisch per Re-Rendering berechnet | Kein Rendering möglich → andere Metrik nötig |
| Stable-Gate | Stabile Gaussians nie anfassen | Kein stable-Gate — stabile unsichtbare Gaussians sind auf CPU ebenso nutzlos |

### Warum `_global_scores` nicht funktioniert

Ein naheliegender Proxy wäre `_global_scores[:, 0]` (akkumulierter Gradient über die gesamte
GPU-Lebenszeit). Das Problem: dieser Wert akkumuliert über **Tausende von Iterationen**
(50 Iters/KF × viele KFs). Schon ein kaum sichtbarer Gaussian mit einem Score von 1e-4 pro
Iteration hat nach 1000 Iterationen `_global_scores[:, 0] = 0.1` — deutlich über jedem
sinnvollen Threshold. In der Praxis liegt **kein einziger** CPU-Gaussian unter 0.05, der
Prune entfernt 0 Gaussians und hilft nicht gegen RAM-Erschöpfung.

### Prune-Kriterium: `sigmoid(_opacity)`

```python
opacity_activated = torch.sigmoid(self._opacity.squeeze(-1))
prune_mask = opacity_activated < threshold
```

Der Adam-Optimizer auf der GPU senkt implizit die Opacity von Gaussians, die beim
Rendering nichts beitragen: negatives Opacity-Gradient → sinkender Logit → `sigmoid → 0`.
Dieser Mechanismus ist **iterationsunabhängig** und liefert einen direkten Wert in [0, 1]:

- `sigmoid(logit) < 0.05`: Gaussian ist für < 5% Deckkraft zuständig → praktisch unsichtbar
- Das Kriterium gilt unabhängig von `stable_mask` — stabile Gaussians mit niedriger Opacity
  sind auf CPU genauso tot wie instabile

Konfiguration: `storage_manager.cpu_prune_opacity_threshold` (default `0.05`).

Der Schwellwert `threshold` (default **0.05**, entspricht der unteren Grenze von
`storage_control`) ist über die Config konfigurierbar:

```yaml
storage_manager:
  distance_threshold: 30.0
  cpu_prune_score_threshold: 0.05   # optional, default 0.05
```

### Kadenz

`prune_cpu_gaussians()` wird aus `run()` aufgerufen, das pro Keyframe einmal ausgeführt
wird. Der interne Zähler `_run_call_counter` sorgt dafür, dass das Pruning nur **alle
4 Aufrufe** stattfindet — dieselbe Kadenz wie `storage_control()` auf der GPU-Seite.

---

## Qualitäts-Trade-offs

| `cpu_prune_score_threshold` | Effekt |
|---|---|
| `0.0` | Kein Pruning (nur Gaussians mit Score exakt 0) — sehr konservativ |
| `0.05` (default) | Entfernt Gaussians, die auf GPU ebenfalls als "untere Grenze" galten |
| `0.1`–`0.5` | Aggressiver — mehr RAM-Einsparung, potenziell leichte Qualitätsverluste in Randbereichen |

Stabile Gaussians (`stable_mask = True`) sind bei **jedem** Schwellwert geschützt und
werden niemals durch dieses Pruning entfernt. Das größte Qualitätsrisiko liegt bei
unstabilen Gaussians mit niedrigem, aber nicht-null Score — diese könnten bei sehr
aggressivem Threshold fälschlicherweise entfernt werden, obwohl sie noch sichtbar sind.

---

## Zusammenfassung der Änderungen in `storage_manage.py`

| Neu | Beschreibung |
|---|---|
| `_run_call_counter` | Zähler in `__init__`, inkrementiert in `run()` |
| `_apply_keep_mask(keep)` | Wendet Boolean-Maske auf alle CPU-Tensoren an (CPU-Äquivalent zu `prune_tensors_from_optimizer`) |
| `prune_cpu_gaussians()` | Führt das Pruning durch — CPU-Pendant zu `storage_control()` |
| `run()` STEP 4 | Ruft `prune_cpu_gaussians()` alle 4 Iterationen auf |
