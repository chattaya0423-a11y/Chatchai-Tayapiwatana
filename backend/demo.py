"""Synthetic microscopy frame generator for the demo dataset.

Produces real RGB frames (green target blobs, red NK blobs, some NK orbiting
targets to create persistent contact episodes) that are run through the exact
analysis pipeline. Output is genuine analysis, not fabricated numbers.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

H, W = 260, 340


def _disk(frame: np.ndarray, cy: float, cx: float, radius: int, channel: int, value: int) -> None:
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    frame[mask, channel] = np.maximum(frame[mask, channel], value)


def _generate(seed: int, n_frames: int, n_targets: int, n_nk: int, engage_fraction: float) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    # Static-ish target cells (green), slow drift.
    targets = np.column_stack([
        rng.uniform(40, H - 40, n_targets),
        rng.uniform(40, W - 40, n_targets),
    ])
    target_drift = rng.normal(0, 0.6, targets.shape)

    # NK cells (red). A subset is assigned to orbit a specific target (engaging).
    nk = np.column_stack([
        rng.uniform(30, H - 30, n_nk),
        rng.uniform(30, W - 30, n_nk),
    ])
    n_engaging = int(n_nk * engage_fraction)
    assigned_target = rng.integers(0, n_targets, n_engaging)
    nk_vel = rng.normal(0, 3.0, nk.shape)

    frames: List[np.ndarray] = []
    for _ in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        targets = np.clip(targets + target_drift, 20, [H - 20, W - 20])
        for cy, cx in targets:
            _disk(frame, cy, cx, 8, 1, 210)  # green target, area ~200

        for i in range(n_nk):
            if i < n_engaging:
                tgt = targets[assigned_target[i]]
                # Orbit close to the assigned target to stay within contact range.
                direction = tgt - nk[i]
                nk[i] = nk[i] + 0.35 * direction + rng.normal(0, 1.2, 2)
            else:
                nk[i] = nk[i] + nk_vel[i] + rng.normal(0, 1.0, 2)
                for d in range(2):
                    lo, hi = (30, H - 30) if d == 0 else (30, W - 30)
                    if nk[i, d] < lo or nk[i, d] > hi:
                        nk_vel[i, d] *= -1
                nk[i] = np.clip(nk[i], [30, 30], [H - 30, W - 30])
            _disk(frame, nk[i, 0], nk[i, 1], 5, 0, 220)  # red NK, area ~80
        frames.append(frame)
    return frames


def demo_conditions() -> List[Tuple[str, str, List[np.ndarray]]]:
    """Return (label, source_name, frames) for each demo condition."""
    return [
        ("S1 · +Antibody", "demo_S1_HEK293T_NK_HuM6-1B9.avi", _generate(1, 20, 14, 22, 0.85)),
        ("S2 · Control", "demo_S2_HEK293T_NK_No-Antibody.avi", _generate(7, 20, 14, 18, 0.18)),
    ]
