"""
CytoSoftware-RPQ core routines.

This module provides a notebook-independent implementation of the RPQ algorithm:
- Read MP4 frames
- Enhance red signal
- Detect blobs using Laplacian-of-Gaussian
- Export counts vs time, per-blob features, annotated MP4, and run_config.json

It is derived from the Colab/Jupyter workflow but avoids any widget/Colab-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Tuple
import json
import math

import numpy as np
import pandas as pd

import cv2
from skimage import util
from skimage.feature import blob_log
from skimage.morphology import disk, opening


@dataclass
class RPQConfig:
    # Performance
    process_every_n_frames: int = 1
    max_frames_to_analyze: Optional[int] = None

    # Red enhancement / background
    bg_blur_sigma: float = 8.0
    background_method: str = "gaussian"  # "gaussian" or "rolling_ball"
    rolling_ball_radius: int = 25

    # Filtering
    snr_min: float = 0.0
    red_dom_ratio: float = 1.6
    red_min_abs: float = 0.08

    # Blob detection
    log_sigma_min: float = 1.0
    log_sigma_max: float = 4.5
    log_num_sigmas: int = 8
    log_thresh: float = 0.03
    robust_k: float = 3.0

    # Area filtering (pixels)
    min_particle_area: int = 6
    max_particle_area: int = 260


def iter_video_frames(path: str | Path,
                     every_n: int = 1,
                     max_frames: Optional[int] = None) -> Tuple[float, int, Iterator[Tuple[int, int, np.ndarray]]]:
    """
    Iterate RGB frames from a video.

    Returns:
        fps, frame_count, iterator yielding (orig_index, used_index, frame_rgb)
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if (not fps) or fps < 1:
        fps = 25.0  # fallback

    def _gen() -> Iterator[Tuple[int, int, np.ndarray]]:
        kept_i = 0
        orig_i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if orig_i % max(1, every_n) == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield orig_i, kept_i, frame_rgb
                kept_i += 1
                if max_frames is not None and kept_i >= max_frames:
                    break
            orig_i += 1
        cap.release()

    return fps, frame_count, _gen()


def _estimate_background(enh: np.ndarray, *, method: str, bg_blur_sigma: float, rolling_ball_radius: int) -> np.ndarray:
    if method == "rolling_ball":
        # morphological opening approximates rolling ball background
        selem = disk(int(max(1, rolling_ball_radius)))
        bg = opening(enh, selem)
        # mild smoothing to avoid blocky background
        if bg_blur_sigma and bg_blur_sigma > 0:
            bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), float(bg_blur_sigma))
        return bg.astype(np.float32)
    # default gaussian blur background
    return cv2.GaussianBlur(enh.astype(np.float32), (0, 0), float(bg_blur_sigma))


def red_enhance(frame_rgb: np.ndarray, cfg: RPQConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = util.img_as_float32(frame_rgb)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    enh = r - 0.5 * (g + b)
    enh = np.clip(enh, 0, 1)
    bg = _estimate_background(enh, method=cfg.background_method,
                              bg_blur_sigma=cfg.bg_blur_sigma,
                              rolling_ball_radius=cfg.rolling_ball_radius)
    enh2 = np.clip(enh - bg, 0, 1)
    return enh2, r, g, b, bg


def robust_threshold(img: np.ndarray, base: float, k: float) -> float:
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med))) + 1e-6
    return max(float(base), med + float(k) * mad)


def detect_red_blobs(frame_rgb: np.ndarray, cfg: RPQConfig) -> List[Dict[str, Any]]:
    enh, r, g, b, _bg = red_enhance(frame_rgb, cfg)
    thr = robust_threshold(enh, base=cfg.log_thresh, k=cfg.robust_k)

    sigma_min = max(0.5, float(cfg.log_sigma_min))
    sigma_max = float(cfg.log_sigma_max)
    if sigma_max < sigma_min:
        sigma_max = sigma_min
    num_sigma = int(cfg.log_num_sigmas) if int(cfg.log_num_sigmas) >= 1 else 1

    blobs = blob_log(enh, min_sigma=sigma_min, max_sigma=sigma_max, num_sigma=num_sigma, threshold=thr)
    accepted: List[Dict[str, Any]] = []
    for (y, x, s) in blobs:
        y, x = int(round(y)), int(round(x))
        rad = max(2, int(round(math.sqrt(2) * float(s))))
        y0, y1 = max(0, y - rad), min(enh.shape[0], y + rad + 1)
        x0, x1 = max(0, x - rad), min(enh.shape[1], x + rad + 1)

        patch_enh = enh[y0:y1, x0:x1]
        patch_r = r[y0:y1, x0:x1]
        patch_g = g[y0:y1, x0:x1]
        patch_b = b[y0:y1, x0:x1]

        if float(patch_enh.max()) < float(cfg.red_min_abs):
            continue

        mean_r = float(patch_r.mean())
        mean_g = float(patch_g.mean()) + 1e-6
        mean_b = float(patch_b.mean()) + 1e-6
        dom_rg = mean_r / mean_g
        dom_rb = mean_r / mean_b
        if not (dom_rg >= float(cfg.red_dom_ratio) and dom_rb >= float(cfg.red_dom_ratio)):
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= rad ** 2
        area = int(mask.sum())
        if area < int(cfg.min_particle_area) or area > int(cfg.max_particle_area):
            continue

        core_vals = patch_enh[mask]
        bg_vals = patch_enh[~mask]
        if bg_vals.size < 5:
            bg_vals = patch_enh.ravel()
        bg_mean = float(bg_vals.mean())
        bg_std = float(bg_vals.std()) + 1e-6
        peak = float(core_vals.max()) if core_vals.size else float(patch_enh.max())
        snr = float((peak - bg_mean) / bg_std)
        if float(cfg.snr_min) and snr < float(cfg.snr_min):
            continue

        accepted.append({
            "y": y, "x": x, "sigma": float(s), "radius": int(rad),
            "area": area,
            "mean_r": mean_r, "mean_g": float(mean_g - 1e-6), "mean_b": float(mean_b - 1e-6),
            "dom_rg": dom_rg, "dom_rb": dom_rb,
            "peak_enh": peak,
            "bg_mean_enh": bg_mean, "bg_std_enh": bg_std,
            "snr": snr,
        })
    return accepted


def overlay_blobs(frame_rgb: np.ndarray, blobs: List[Dict[str, Any]]) -> np.ndarray:
    out = frame_rgb.copy()
    for blob in blobs:
        y = int(blob["y"]); x = int(blob["x"]); rad = int(blob["radius"])
        cv2.circle(out, (x, y), rad, (255, 0, 0), 2)  # RGB (red circle)
    return out


def analyze_video(video_path: str | Path,
                  out_dir: str | Path,
                  cfg: Optional[RPQConfig] = None) -> Dict[str, Any]:
    """
    Run RPQ analysis and write outputs into out_dir.

    Outputs:
      - red_particles_by_time.csv
      - red_blob_features.csv
      - annotated_red_particles.mp4
      - run_config.json
    """
    cfg = cfg or RPQConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps, frame_count, frame_iter = iter_video_frames(video_path, every_n=cfg.process_every_n_frames,
                                                     max_frames=cfg.max_frames_to_analyze)

    csv_path = out_dir / "red_particles_by_time.csv"
    blobs_csv_path = out_dir / "red_blob_features.csv"
    annotated_mp4_path = out_dir / "annotated_red_particles.mp4"

    results: List[Dict[str, Any]] = []
    blob_rows: List[Dict[str, Any]] = []

    writer = None
    annotated_written = False

    for orig_i, used_i, frame_rgb in frame_iter:
        time_sec = (orig_i / fps) if fps else float(orig_i) / 25.0
        blobs = detect_red_blobs(frame_rgb, cfg)

        for bi, blob in enumerate(blobs):
            blob_rows.append({
                "frame_index_original": orig_i,
                "frame_index_used": used_i,
                "time_sec": time_sec,
                "blob_id_in_frame": bi,
                **blob,
            })

        results.append({
            "frame_index_original": orig_i,
            "frame_index_used": used_i,
            "time_sec": time_sec,
            "n_red_particles": len(blobs),
        })

        annotated = overlay_blobs(frame_rgb, blobs)

        if writer is None:
            h, w = annotated.shape[:2]
            out_fps = fps / max(1, cfg.process_every_n_frames)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_mp4_path), fourcc, out_fps, (w, h))

        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()
        annotated_written = True

    if len(results) == 0:
        raise RuntimeError("No frames were analyzed. Check the video path and settings.")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    df_blobs = pd.DataFrame(blob_rows)
    df_blobs.to_csv(blobs_csv_path, index=False)

    run_config = {
        "video_path": str(video_path),
        "video_frame_count": frame_count,
        "video_fps": fps,
        **asdict(cfg),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    return {
        "df": df,
        "df_blobs": df_blobs,
        "out_dir": str(out_dir),
        "csv_path": str(csv_path),
        "blobs_csv_path": str(blobs_csv_path),
        "annotated_mp4_path": str(annotated_mp4_path),
        "annotated_video_written": annotated_written,
        "run_config_path": str(out_dir / "run_config.json"),
    }
