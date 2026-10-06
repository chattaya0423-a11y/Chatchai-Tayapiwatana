"""Persistent NK-proximal target occupation analysis.

Faithful port of the NK_Target_Persistence_Analysis Colab notebook.
Segmentation, nearest-centroid tracking, proximity contact detection and
persistent-episode classification are reproduced exactly; only the video
decoder is swapped to imageio-ffmpeg so it runs as a hosted service.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi


def segment_frame(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer label images for red NK objects and green target objects."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Each frame must be an RGB image with three channels.")

    red = ndi.gaussian_filter(rgb[:, :, 0].astype(float), 1)
    green = ndi.gaussian_filter(rgb[:, :, 1].astype(float), 1)

    # Red NK segmentation: red dominance, object-size, peak-intensity and border filters.
    red_signal = red - 0.5 * green
    raw, n = ndi.label(red_signal > 35)
    area = np.bincount(raw.ravel(), minlength=n + 1)
    peak = ndi.maximum(red_signal, raw, np.arange(n + 1))
    keep = (area >= 35) & (area <= 650) & (peak >= 80)
    keep[0] = False
    border_labels = np.unique(
        np.r_[raw[:3, :].ravel(), raw[-3:, :].ravel(), raw[:, :3].ravel(), raw[:, -3:].ravel()]
    )
    keep[border_labels] = False
    nk_labels = ndi.label(keep[raw])[0]

    # Green target segmentation: green dominance, hole filling and area filter.
    green_mask = ndi.binary_fill_holes(green - 0.5 * red > 20)
    raw, n = ndi.label(green_mask)
    area = np.bincount(raw.ravel(), minlength=n + 1)
    keep = area >= 80
    keep[0] = False
    target_labels = ndi.label(keep[raw])[0]
    return nk_labels, target_labels


def component_centers(label_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.arange(1, int(label_image.max()) + 1)
    if len(ids) == 0:
        return ids, np.empty((0, 2), dtype=float)
    centers = np.asarray(ndi.center_of_mass(label_image > 0, label_image, ids), dtype=float)
    return ids, centers


def assign_tracks(
    previous_centers: np.ndarray,
    current_centers: np.ndarray,
    previous_ids: Sequence[int],
    next_id: int,
    maximum_distance: float = 100.0,
) -> Tuple[List[int], int]:
    """Greedy one-to-one nearest-centroid assignment with a displacement limit."""
    if len(current_centers) == 0:
        return [], next_id
    if len(previous_centers) == 0:
        return list(range(next_id, next_id + len(current_centers))), next_id + len(current_centers)

    distances = np.sqrt(
        ((current_centers[:, None, :] - previous_centers[None, :, :]) ** 2).sum(axis=2)
    )
    assigned = [-1] * len(current_centers)
    used_previous = set()
    for i in np.argsort(distances.min(axis=1)):
        j = int(distances[i].argmin())
        if distances[i, j] <= maximum_distance and j not in used_previous:
            assigned[i] = int(previous_ids[j])
            used_previous.add(j)
    for i in range(len(assigned)):
        if assigned[i] < 0:
            assigned[i] = next_id
            next_id += 1
    return assigned, next_id


def read_video_rgb(video_path: str | Path) -> List[np.ndarray]:
    """Read all encoded frames; each encoded frame is one biological time point."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: List[np.ndarray] = []
    for frame in iio.imiter(video_path, plugin="FFMPEG"):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        frames.append(arr[:, :, :3].astype(np.uint8))
    if not frames:
        raise ValueError(f"No frames were decoded from: {video_path}")
    return frames


def analyze_frames(
    label: str,
    frames: List[np.ndarray],
    contact_px: float = 10.0,
    stable_frames: int = 4,
    frame_interval_min: float = 5.0,
    max_target_displacement_px: float = 100.0,
) -> Tuple[List[dict], List[dict], int, int]:
    """Core analysis over a list of RGB frames.

    Returns (frame_rows, episode_rows, eligible_tracks, persistent_tracks).
    """
    frame_rows: List[dict] = []
    episode_rows: List[dict] = []
    previous_centers = np.empty((0, 2), dtype=float)
    previous_ids: List[int] = []
    next_id = 0
    track_seen: Dict[int, int] = {}
    open_episodes: Dict[int, Tuple[int, int]] = {}

    for frame_number, rgb in enumerate(frames, start=1):
        nk_labels, target_labels = segment_frame(rgb)
        target_component_ids, target_centers = component_centers(target_labels)
        target_ids, next_id = assign_tracks(
            previous_centers,
            target_centers,
            previous_ids,
            next_id,
            maximum_distance=max_target_displacement_px,
        )
        for target_id in target_ids:
            track_seen[target_id] = track_seen.get(target_id, 0) + 1

        distance_from_nk = ndi.distance_transform_edt(~(nk_labels > 0))
        active_target_ids = set()
        for i, component in enumerate(target_component_ids):
            minimum_distance = float(np.min(distance_from_nk[target_labels == component]))
            if minimum_distance <= contact_px:
                active_target_ids.add(target_ids[i])

        for target_id in list(open_episodes):
            if target_id not in active_target_ids:
                start, last = open_episodes.pop(target_id)
                episode_rows.append(
                    {
                        "label": label,
                        "target_track_id": int(target_id),
                        "start_frame": int(start),
                        "end_frame": int(last),
                        "n_frames": int(last - start + 1),
                        "duration_min": float((last - start) * frame_interval_min),
                        "sampled_occupancy_min": float((last - start + 1) * frame_interval_min),
                    }
                )

        for target_id in active_target_ids:
            if target_id in open_episodes:
                open_episodes[target_id] = (open_episodes[target_id][0], frame_number)
            else:
                open_episodes[target_id] = (frame_number, frame_number)

        frame_rows.append(
            {
                "label": label,
                "frame": int(frame_number),
                "time_min": float((frame_number - 1) * frame_interval_min),
                "target_regions_detected": int(len(target_ids)),
                "target_regions_engaged": int(len(active_target_ids)),
                "engaged_target_percent": (
                    float(100.0 * len(active_target_ids) / len(target_ids)) if target_ids else None
                ),
                "nk_objects_detected": int(nk_labels.max()),
            }
        )
        previous_centers, previous_ids = target_centers, target_ids

    for target_id, (start, last) in open_episodes.items():
        episode_rows.append(
            {
                "label": label,
                "target_track_id": int(target_id),
                "start_frame": int(start),
                "end_frame": int(last),
                "n_frames": int(last - start + 1),
                "duration_min": float((last - start) * frame_interval_min),
                "sampled_occupancy_min": float((last - start + 1) * frame_interval_min),
            }
        )

    for row in episode_rows:
        row["persistent"] = bool(row["n_frames"] >= stable_frames)

    eligible = int(sum(1 for n in track_seen.values() if n >= stable_frames))
    persistent = int(len({r["target_track_id"] for r in episode_rows if r["persistent"]}))
    return frame_rows, episode_rows, eligible, persistent


def summarize(
    label: str,
    source_name: str,
    frame_rows: List[dict],
    episode_rows: List[dict],
    eligible: int,
    persistent: int,
) -> dict:
    engaged_vals = [r["engaged_target_percent"] for r in frame_rows if r["engaged_target_percent"] is not None]
    target_vals = [r["target_regions_detected"] for r in frame_rows]
    persistent_eps = [r for r in episode_rows if r["persistent"]]
    mean_dur = (
        float(np.mean([r["duration_min"] for r in persistent_eps])) if persistent_eps else 0.0
    )
    return {
        "label": label,
        "video_path": source_name,
        "frames": len(frame_rows),
        "mean_target_regions": float(np.mean(target_vals)) if target_vals else 0.0,
        "mean_engaged_target_percent": float(np.mean(engaged_vals)) if engaged_vals else 0.0,
        "all_contact_episode_count": len(episode_rows),
        "persistent_target_tracks": persistent,
        "eligible_target_tracks_observed_4plus_frames": eligible,
        "persistent_target_fraction_percent": (
            float(100.0 * persistent / eligible) if eligible else None
        ),
        "persistent_episode_count": len(persistent_eps),
        "mean_persistent_duration_min": mean_dur,
    }


def build_config(params: dict, videos: Dict[str, str]) -> dict:
    return {
        "contact_px": params["contact_px"],
        "stable_frames": params["stable_frames"],
        "frame_interval_min": params["frame_interval_min"],
        "max_target_displacement_px": params["max_target_displacement_px"],
        "red_channel": "R - 0.5G; threshold 35; area 35-650; peak >=80; border exclusion 3 px",
        "green_channel": "G - 0.5R; threshold 20; hole filling; area >=80",
        "videos": videos,
        "interpretation": "Persistent NK-proximal target occupation; not direct ultrastructural membrane contact.",
    }
