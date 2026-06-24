# Tiefenmodelle: Factory + `scale_align`

Config-schaltbares Single-Image-Tiefenmodell und ein Schalter, der bestimmt, ob
die Metric3D-Tiefe die DROID-BA-Tiefe **ersetzt** oder nur deren **Skala** setzt.
Alle Felder sind optional; ohne sie ist das Verhalten exakt wie vorher.

## Ziel-Config

```yaml
use_metric: true
use_metric_for_mapper: true

depth_model:
  kind: metric3d          # Registry-Key; aktuell einziges Backend, erweiterbar
  variant: v2-S           # v2-S | v2-L | v2-g  (Backbone-Größe)
  checkpoint: ckpts/metric_depth_vit_small_800k.pth

metric_cov: 1.0           # Prior-Vertrauen (weight = 1/cov). Kleiner = härter.
scale_align: false        # false = Replace (alt) | true = nur Skala auf DROID
scale_align_min_depth: 0.2  # Gültigkeits-Untergrenze fürs Skalen-Median (m)
```

Beispiel-Config: `configs/local/depth_model/smallcity_scale_align.yaml`.

## Factory-Pattern

`scripts/metric/depth_factory.py` spiegelt `scripts/vings_utils/selector_factory.py`:

- `_REGISTRY: dict[str, Callable]` — `kind -> builder(cfg, u_scale, v_scale)`.
- `@register_depth_model("name")` — Dekorator zum Registrieren eines Backends.
- `make_depth_model(cfg, u_scale=None, v_scale=None)` — liest
  `cfg['depth_model']['kind']` (Default `metric3d`, falls Block fehlt) und baut die
  Instanz. Unbekanntes `kind` → `ValueError` mit `known_depth_kinds()`.
- `known_depth_kinds() -> list[str]` — für Fehlermeldungen/Diagnose.

Alle Run-Entrypoints (`run.py`, `run_tracking.py`, `run_multiprocess*.py`) gehen
über `make_depth_model(cfg)` statt direkt über `Metric_Model(cfg)`.

**Gemeinsame Schnittstelle** jedes Backends:

```python
predict(img) -> torch.Tensor   # (H, W), float32, auf img.device
```

Mehr braucht der Rest des Codes nicht: der Cache in `run.py` und der Mapper-
Injection-Block hängen nur an dieser `(H, W)`-Signatur.

### Variant-Switch (v2-S → v2-L)

`scripts/metric/metric_model.py` liest `variant` und `checkpoint` aus
`cfg['depth_model']` (Defaults `v2-S` / `ckpts/metric_depth_vit_small_800k.pth`).
Sobald der L-Checkpoint in `ckpts/` liegt, zieht man v2-S → v2-L allein per Config
(billiger Schärfe-Test, kein Code-Change).

### Neues Backend registrieren

```python
# scripts/metric/my_backend.py
class MyDepthModel:
    def __init__(self, cfg, u_scale=None, v_scale=None): ...
    def predict(self, img): ...   # -> torch.Tensor (H, W)

# scripts/metric/depth_factory.py
@register_depth_model("my_backend")
def _build_my_backend(cfg, u_scale, v_scale):
    from metric.my_backend import MyDepthModel
    return MyDepthModel(cfg, u_scale, v_scale)
```

Vorgemerkt als nächste Backends: **UniDepth V2** und **Depth-Anything-V2**
(beide liefern metrische Single-Image-Tiefe und passen auf dieselbe `predict`-
Signatur).

## `scale_align`-Mathe

Im Mapper-Injection-Pfad (`scripts/run.py`) wird pro Keyframe die gecachte
Metric3D-Tiefe `m_d` mit der DROID-BA-Tiefe `d_droid` (= das, was schon in
`viz_out['depths']` steht) verrechnet:

- **`scale_align: false` (Default, Replace):** `viz_out['depths'] = m_d`
  (Sky-Pixel, `rgb==0`, bleiben bei 0). Historisches Verhalten.
- **`scale_align: true` (Skala only):**
  ```
  valid = ¬sky ∧ (m_d > min_d) ∧ (d_droid > min_d)
  s     = median( m_d[valid] / d_droid[valid] )
  viz_out['depths'] = d_droid · s        (Sky bleibt 0)
  ```
  `min_d = scale_align_min_depth` (Default 0.2 m). **Failsafe:** < 100 gültige
  Pixel → kein Re-Scale, DROID-Tiefe (und ihre Kovarianz) bleiben unangetastet.

In **beiden** Pfaden wird `depths_cov` aus `metric_cov` gesetzt — außer im
Failsafe-Skip, wo nichts geschrieben wird.

### Warum DROID-Struktur scharf ist

DROID-BA schätzt Tiefe aus **Multi-View-Bundle-Adjustment** (Geometrie über mehrere
Frames). Metric3D schätzt aus einem **einzelnen Bild**. Bei Aerial-Nadir-Szenen auf
flachem Boden ist die Single-Image-Tiefe verrauscht und platziert Gaussians falsch
(CLAUDE.md Aerial-Erkenntnis #1: `use_metric: false` ist dort ein +2–3 dB Hebel).
`scale_align` ist der Mittelweg: behalte die scharfe, multi-view-konsistente
DROID-Struktur, übernimm nur den fehlenden **metrischen Maßstab** von Metric3D.

## `metric_cov`-Wirkung

`depths_cov` geht als `weight = 1 / cov` in den `weighted_l1`-Term von `get_loss()`.

| `metric_cov` | weight | Effekt |
|---|---|---|
| 0.01 | 100 | Tiefen-Prior dominiert den RGB-Loss |
| 1.0 (Default) | ~1 | Tiefe und RGB etwa gleich gewichtet |
| groß | klein | Tiefe weich, RGB dominiert |

`metric_cov` war schon vor diesem Change config-steuerbar (`run.py` liest
`cfg.get('metric_cov', 1.0)`); hier nur dokumentiert.

## Rückwärtskompatibilität

Configs ohne `depth_model`-Block und ohne `scale_align`:
- `make_depth_model` → `kind='metric3d'`,
- `Metric_Model` → `v2-S` + small-Checkpoint,
- Injection → Replace-Pfad.

→ Bit-identisches Verhalten zu vor dem Change.

## Smoketest

```bash
python -c "from metric.depth_factory import make_depth_model, known_depth_kinds; print(known_depth_kinds())"
# -> ['metric3d']
```
