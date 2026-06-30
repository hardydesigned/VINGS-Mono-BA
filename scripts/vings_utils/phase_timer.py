"""Profiling-Infrastruktur fuer den VINGS-Run-Loop.

Aus scripts/run.py ausgelagert. Enthaelt den hierarchischen Sub-Timer
(`PhaseTimer`) mit cuda.synchronize() vor/nach jeder gemessenen Phase, die
zwei Hilfs-Wrapper (`_PhaseCtx`, `_CallableProxy`) sowie den atomaren
profiling.json-Dump (`write_profiling_json`).
"""

import os
import json
import time
import statistics

import torch


class PhaseTimer:
    """Sammelt Sub-Timer mit cuda.synchronize() vor/nach time.time()."""
    def __init__(self, sync=True):
        self.records = {}
        self.sync = sync and torch.cuda.is_available()

    def time(self, name):
        return _PhaseCtx(self, name)

    def add(self, name, dt):
        self.records.setdefault(name, []).append(dt)

    def last(self, name):
        rec = self.records.get(name)
        return rec[-1] if rec else 0.0

    def patch(self, obj, attr, name):
        orig = getattr(obj, attr)
        timer = self
        def wrapper(*args, **kwargs):
            with timer.time(name):
                return orig(*args, **kwargs)
        setattr(obj, attr, wrapper)

    def patch_callable(self, obj, attr, name):
        orig = getattr(obj, attr)
        proxy = _CallableProxy(orig, self, name)
        setattr(obj, attr, proxy)

    def summary(self, total_wall=None):
        if not self.records:
            print("(no timing records)")
            return
        rows = []
        for name, vals in self.records.items():
            n = len(vals)
            tot = sum(vals)
            mean = tot / n
            med = statistics.median(vals)
            p95 = sorted(vals)[max(0, int(0.95 * n) - 1)] if n > 0 else 0.0
            rows.append((name, n, tot, mean, med, p95))
        rows.sort(key=lambda r: -r[2])
        denom = total_wall if total_wall else max(r[2] for r in rows)
        print(f"{'phase':<28} {'n':>6} {'total[s]':>10} {'mean[ms]':>10} "
              f"{'med[ms]':>10} {'p95[ms]':>10} {'%':>7}")
        print("-" * 86)
        for name, n, tot, mean, med, p95 in rows:
            pct = 100.0 * tot / denom if denom > 0 else 0.0
            print(f"{name:<28} {n:>6} {tot:>10.2f} {mean*1000:>10.1f} "
                  f"{med*1000:>10.1f} {p95*1000:>10.1f} {pct:>6.1f}%")


class _PhaseCtx:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        if self.timer.sync:
            torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer.sync:
            torch.cuda.synchronize()
        self.timer.add(self.name, time.time() - self.t0)
        return False


class _CallableProxy:
    """Transparent-Proxy: leitet Attribut-Zugriffe ans Original weiter,
    misst aber jeden __call__ in einer Phase."""
    __slots__ = ('_target', '_timer', '_name')

    def __init__(self, target, timer, name):
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_timer', timer)
        object.__setattr__(self, '_name', name)

    def __call__(self, *args, **kwargs):
        with self._timer.time(self._name):
            return self._target(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._target, attr)

    def __setattr__(self, attr, value):
        setattr(self._target, attr, value)


def write_profiling_json(timer, cfg, *, n_keyframes, n_mapped, n_processed,
                         n_frames, last_idx, frame_skip, mapper_kf_skip,
                         wall_t0, partial):
    """Atomic profiling.json dump. Survives SIGKILL/OOM mid-run."""
    try:
        out_path = os.path.join(cfg['output']['save_dir'], 'profiling.json')
        tmp_path = out_path + '.tmp'
        payload = {
            'wall_total_s': time.time() - wall_t0,
            'n_keyframes': n_keyframes,
            'n_mapped': n_mapped,
            'n_processed': n_processed,
            'n_frames': n_frames,
            'last_idx': last_idx,
            'frame_skip': frame_skip,
            'mapper_kf_skip': mapper_kf_skip,
            'partial': partial,
            'records': timer.records,
        }
        with open(tmp_path, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp_path, out_path)
    except Exception as e:
        print(f"profiling.json write failed: {e}")
