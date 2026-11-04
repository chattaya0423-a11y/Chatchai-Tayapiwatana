import json, os

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md("""# NK–Target Persistence Analysis

**Persistent NK-proximal target occupation** from two-colour live-cell microscopy movies
(**red = NK / effector cells**, **green = target cells**).

This notebook is the Colab version of the NK-Track web app. It reproduces the exact analysis:
Gaussian denoising -> channel-contrast segmentation -> nearest-centroid target tracking ->
Euclidean proximity contact detection -> persistent-episode classification.

**How to use**
1. Run **Setup** once.
2. Set **Parameters**.
3. Choose **Demo** (synthetic) or **Upload** your `.avi/.mp4/.mov` movies.
4. Run **Analyze**, then **Visualize** and **Download**.

> Primary endpoint: **persistent-target fraction (%)** = 100 x persistent / eligible target tracks.
""")

md("## Setup — install & imports (run once)")

code('''#@title Install dependencies
!pip -q install imageio imageio-ffmpeg scipy pandas matplotlib numpy
print("Dependencies ready.")
''')

code('''#@title Analysis engine (faithful port of the web-app backend)
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import imageio.v3 as iio
import numpy as np
from scipy import ndimage as ndi


def segment_frame(rgb):
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Each frame must be an RGB image with three channels.")
    red = ndi.gaussian_filter(rgb[:, :, 0].astype(float), 1)
    green = ndi.gaussian_filter(rgb[:, :, 1].astype(float), 1)

    red_signal = red - 0.5 * green
    raw, n = ndi.label(red_signal > 35)
    area = np.bincount(raw.ravel(), minlength=n + 1)
    peak = ndi.maximum(red_signal, raw, np.arange(n + 1))
    keep = (area >= 35) & (area <= 650) & (peak >= 80)
    keep[0] = False
    border = np.unique(np.r_[raw[:3, :].ravel(), raw[-3:, :].ravel(),
                             raw[:, :3].ravel(), raw[:, -3:].ravel()])
    keep[border] = False
    nk_labels = ndi.label(keep[raw])[0]

    green_mask = ndi.binary_fill_holes(green - 0.5 * red > 20)
    raw, n = ndi.label(green_mask)
    area = np.bincount(raw.ravel(), minlength=n + 1)
    keep = area >= 80
    keep[0] = False
    target_labels = ndi.label(keep[raw])[0]
    return nk_labels, target_labels


def component_centers(label_image):
    ids = np.arange(1, int(label_image.max()) + 1)
    if len(ids) == 0:
        return ids, np.empty((0, 2), dtype=float)
    centers = np.asarray(ndi.center_of_mass(label_image > 0, label_image, ids), dtype=float)
    return ids, centers


def assign_tracks(previous_centers, current_centers, previous_ids, next_id, maximum_distance=100.0):
    if len(current_centers) == 0:
        return [], next_id
    if len(previous_centers) == 0:
        return list(range(next_id, next_id + len(current_centers))), next_id + len(current_centers)
    distances = np.sqrt(((current_centers[:, None, :] - previous_centers[None, :, :]) ** 2).sum(axis=2))
    assigned = [-1] * len(current_centers)
    used = set()
    for i in np.argsort(distances.min(axis=1)):
        j = int(distances[i].argmin())
        if distances[i, j] <= maximum_distance and j not in used:
            assigned[i] = int(previous_ids[j]); used.add(j)
    for i in range(len(assigned)):
        if assigned[i] < 0:
            assigned[i] = next_id; next_id += 1
    return assigned, next_id


def read_video_rgb(video_path):
    video_path = Path(video_path)
    frames = []
    for frame in iio.imiter(video_path, plugin="FFMPEG"):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        frames.append(arr[:, :, :3].astype(np.uint8))
    if not frames:
        raise ValueError(f"No frames were decoded from: {video_path}")
    return frames


def analyze_frames(label, frames, contact_px=10.0, stable_frames=4,
                   frame_interval_min=5.0, max_target_displacement_px=100.0):
    frame_rows, episode_rows = [], []
    previous_centers = np.empty((0, 2), dtype=float)
    previous_ids, next_id = [], 0
    track_seen, open_episodes = {}, {}

    for frame_number, rgb in enumerate(frames, start=1):
        nk_labels, target_labels = segment_frame(rgb)
        target_component_ids, target_centers = component_centers(target_labels)
        target_ids, next_id = assign_tracks(previous_centers, target_centers, previous_ids,
                                             next_id, maximum_distance=max_target_displacement_px)
        for tid in target_ids:
            track_seen[tid] = track_seen.get(tid, 0) + 1

        distance_from_nk = ndi.distance_transform_edt(~(nk_labels > 0))
        active = set()
        for i, comp in enumerate(target_component_ids):
            if float(np.min(distance_from_nk[target_labels == comp])) <= contact_px:
                active.add(target_ids[i])

        for tid in list(open_episodes):
            if tid not in active:
                start, last = open_episodes.pop(tid)
                episode_rows.append({"label": label, "target_track_id": int(tid),
                                     "start_frame": int(start), "end_frame": int(last),
                                     "n_frames": int(last - start + 1),
                                     "duration_min": float((last - start) * frame_interval_min),
                                     "sampled_occupancy_min": float((last - start + 1) * frame_interval_min)})
        for tid in active:
            if tid in open_episodes:
                open_episodes[tid] = (open_episodes[tid][0], frame_number)
            else:
                open_episodes[tid] = (frame_number, frame_number)

        frame_rows.append({"label": label, "frame": int(frame_number),
                           "time_min": float((frame_number - 1) * frame_interval_min),
                           "target_regions_detected": int(len(target_ids)),
                           "target_regions_engaged": int(len(active)),
                           "engaged_target_percent": (float(100.0 * len(active) / len(target_ids)) if target_ids else None),
                           "nk_objects_detected": int(nk_labels.max())})
        previous_centers, previous_ids = target_centers, target_ids

    for tid, (start, last) in open_episodes.items():
        episode_rows.append({"label": label, "target_track_id": int(tid),
                             "start_frame": int(start), "end_frame": int(last),
                             "n_frames": int(last - start + 1),
                             "duration_min": float((last - start) * frame_interval_min),
                             "sampled_occupancy_min": float((last - start + 1) * frame_interval_min)})
    for row in episode_rows:
        row["persistent"] = bool(row["n_frames"] >= stable_frames)

    eligible = int(sum(1 for n in track_seen.values() if n >= stable_frames))
    persistent = int(len({r["target_track_id"] for r in episode_rows if r["persistent"]}))
    return frame_rows, episode_rows, eligible, persistent


def summarize(label, source_name, frame_rows, episode_rows, eligible, persistent):
    engaged = [r["engaged_target_percent"] for r in frame_rows if r["engaged_target_percent"] is not None]
    targets = [r["target_regions_detected"] for r in frame_rows]
    pers = [r for r in episode_rows if r["persistent"]]
    return {"label": label, "video_path": source_name, "frames": len(frame_rows),
            "mean_target_regions": float(np.mean(targets)) if targets else 0.0,
            "mean_engaged_target_percent": float(np.mean(engaged)) if engaged else 0.0,
            "all_contact_episode_count": len(episode_rows),
            "persistent_target_tracks": persistent,
            "eligible_target_tracks_observed_4plus_frames": eligible,
            "persistent_target_fraction_percent": (float(100.0 * persistent / eligible) if eligible else None),
            "persistent_episode_count": len(pers),
            "mean_persistent_duration_min": (float(np.mean([r["duration_min"] for r in pers])) if pers else 0.0)}


print("Analysis engine loaded.")
''')

code('''#@title Synthetic demo generator (optional)
H, W = 260, 340

def _disk(frame, cy, cx, radius, channel, value):
    yy, xx = np.ogrid[:H, :W]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    frame[m, channel] = np.maximum(frame[m, channel], value)

def _generate(seed, n_frames, n_targets, n_nk, engage_fraction):
    rng = np.random.default_rng(seed)
    targets = np.column_stack([rng.uniform(40, H - 40, n_targets), rng.uniform(40, W - 40, n_targets)])
    drift = rng.normal(0, 0.6, targets.shape)
    nk = np.column_stack([rng.uniform(30, H - 30, n_nk), rng.uniform(30, W - 30, n_nk)])
    n_eng = int(n_nk * engage_fraction)
    assigned = rng.integers(0, n_targets, n_eng)
    vel = rng.normal(0, 3.0, nk.shape)
    frames = []
    for _ in range(n_frames):
        f = np.zeros((H, W, 3), np.uint8)
        targets = np.clip(targets + drift, 20, [H - 20, W - 20])
        for cy, cx in targets:
            _disk(f, cy, cx, 8, 1, 210)
        for i in range(n_nk):
            if i < n_eng:
                nk[i] = nk[i] + 0.35 * (targets[assigned[i]] - nk[i]) + rng.normal(0, 1.2, 2)
            else:
                nk[i] = nk[i] + vel[i] + rng.normal(0, 1.0, 2)
                for d in range(2):
                    lo, hi = (30, H - 30) if d == 0 else (30, W - 30)
                    if nk[i, d] < lo or nk[i, d] > hi:
                        vel[i, d] *= -1
                nk[i] = np.clip(nk[i], [30, 30], [H - 30, W - 30])
            _disk(f, nk[i, 0], nk[i, 1], 5, 0, 220)
        frames.append(f)
    return frames

def demo_conditions():
    return [("S1 - +Antibody", "demo_S1_HEK293T_NK_HuM6-1B9.avi", _generate(1, 20, 14, 22, 0.85)),
            ("S2 - Control",   "demo_S2_HEK293T_NK_No-Antibody.avi", _generate(7, 20, 14, 18, 0.18))]

print("Demo generator ready.")
''')

md("## Step 1 — Parameters")

code('''#@title Set analysis parameters { run: "auto" }
contact_px = 10 #@param {type:"number"}
stable_frames = 4 #@param {type:"integer"}
frame_interval_min = 5 #@param {type:"number"}
max_target_displacement_px = 100 #@param {type:"number"}
#@markdown **Threshold sensitivity** — tick any thresholds to also re-run the analysis at those proximities for comparison:
sensitivity_2px = False #@param {type:"boolean"}
sensitivity_5px = False #@param {type:"boolean"}
sensitivity_10px = False #@param {type:"boolean"}

PARAMS = dict(contact_px=float(contact_px), stable_frames=int(stable_frames),
              frame_interval_min=float(frame_interval_min),
              max_target_displacement_px=float(max_target_displacement_px))
SENS_THRESHOLDS = [t for t, on in [(2, sensitivity_2px), (5, sensitivity_5px), (10, sensitivity_10px)] if on]
print("Parameters:", PARAMS)
print("Sensitivity thresholds:", SENS_THRESHOLDS or "none")
''')

md("## Step 2 — Choose input (demo or upload)")

code('''#@title Load videos
#@markdown Pick a data source. For **Upload**, a file picker opens; you may optionally give
#@markdown comma-separated labels (one per file, in the order selected) e.g. `S1, S2`.
data_source = "Use demo (synthetic)" #@param ["Use demo (synthetic)", "Upload my videos"]
custom_labels = "" #@param {type:"string"}

CONDITIONS = []  # list of (label, source_name, frames)
if data_source == "Use demo (synthetic)":
    CONDITIONS = demo_conditions()
    print("Loaded 2 synthetic demo conditions.")
else:
    from google.colab import files
    uploaded = files.upload()
    names = list(uploaded.keys())
    labels = [s.strip() for s in custom_labels.split(",") if s.strip()]
    for i, name in enumerate(names):
        with open(name, "wb") as fh:
            fh.write(uploaded[name])
        label = labels[i] if i < len(labels) else f"S{i+1}"
        print(f"Decoding {name} ...")
        frames = read_video_rgb(name)
        CONDITIONS.append((label, name, frames))
    print(f"Loaded {len(CONDITIONS)} uploaded condition(s).")

for lbl, src, fr in CONDITIONS:
    print(f"  {lbl}: {len(fr)} frames  ({src})")
''')

md("## Step 3 — Run analysis")

code('''#@title Analyze
import pandas as pd
from IPython.display import display

summary_rows, per_frame_rows, episode_rows = [], [], []
for label, source, frames in CONDITIONS:
    fr, ep, elig, pers = analyze_frames(label, frames, **PARAMS)
    per_frame_rows += fr; episode_rows += ep
    summary_rows.append(summarize(label, source, fr, ep, elig, pers))

sensitivity_rows = []
for thr in SENS_THRESHOLDS:
    p = {**PARAMS, "contact_px": float(thr)}
    for label, source, frames in CONDITIONS:
        fr, ep, elig, pers = analyze_frames(label, frames, **p)
        sensitivity_rows.append({"contact_px": thr, **summarize(label, source, fr, ep, elig, pers)})

summary_df = pd.DataFrame(summary_rows)
per_frame_df = pd.DataFrame(per_frame_rows)
episodes_df = pd.DataFrame(episode_rows)
sensitivity_df = pd.DataFrame(sensitivity_rows)

print("=== Summary (primary endpoint: persistent_target_fraction_percent) ===")
display(summary_df)
print("=== Per-frame (head) ==="); display(per_frame_df.head(12))
print("=== Episodes (head) ==="); display(episodes_df.head(12))
if not sensitivity_df.empty:
    print("=== Threshold sensitivity ==="); display(sensitivity_df)
''')

md("## Step 4 — Visualize")

code('''#@title Charts
import matplotlib.pyplot as plt

# 1) Engaged-target % over time, per condition
plt.figure(figsize=(8, 4))
for label, g in per_frame_df.groupby("label"):
    plt.plot(g["time_min"], g["engaged_target_percent"], marker="o", ms=3, label=label)
plt.xlabel("Time (min)"); plt.ylabel("Engaged targets (%)")
plt.title("Engaged-target % over time"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 2) Persistent-target fraction by condition
plt.figure(figsize=(6, 4))
plt.bar(summary_df["label"], summary_df["persistent_target_fraction_percent"])
plt.ylabel("Persistent-target fraction (%)")
plt.title("Primary endpoint by condition"); plt.xticks(rotation=15); plt.tight_layout(); plt.show()

# 3) Threshold sensitivity (if enabled)
if not sensitivity_df.empty:
    plt.figure(figsize=(7, 4))
    for label, g in sensitivity_df.groupby("label"):
        g = g.sort_values("contact_px")
        plt.plot(g["contact_px"], g["persistent_target_fraction_percent"], marker="s", label=label)
    plt.xlabel("Contact proximity (px)"); plt.ylabel("Persistent-target fraction (%)")
    plt.title("Threshold sensitivity"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
''')

md("## Step 5 — Download results")

code('''#@title Export CSV / JSON / ZIP
import json, zipfile

config = {**PARAMS,
          "red_channel": "R - 0.5G; threshold 35; area 35-650; peak >=80; border exclusion 3 px",
          "green_channel": "G - 0.5R; threshold 20; hole filling; area >=80",
          "videos": {lbl: src for lbl, src, _ in CONDITIONS},
          "sensitivity_thresholds": SENS_THRESHOLDS,
          "interpretation": "Persistent NK-proximal target occupation; not direct ultrastructural membrane contact."}

summary_df.to_csv("target_persistence_summary.csv", index=False)
per_frame_df.to_csv("target_persistence_per_frame.csv", index=False)
episodes_df.to_csv("target_persistence_episodes.csv", index=False)
with open("analysis_config.json", "w") as fh:
    json.dump(config, fh, indent=2)
export_files = ["target_persistence_summary.csv", "target_persistence_per_frame.csv",
                "target_persistence_episodes.csv", "analysis_config.json"]
if not sensitivity_df.empty:
    sensitivity_df.to_csv("threshold_sensitivity.csv", index=False)
    export_files.append("threshold_sensitivity.csv")

with zipfile.ZipFile("nk_persistence_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in export_files:
        zf.write(f)

from google.colab import files
files.download("nk_persistence_results.zip")
print("Exported:", export_files)
''')

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

for out in ["/app/NK_Target_Persistence_Analysis_webapp.ipynb",
            "/app/frontend/public/NK_Target_Persistence_Analysis_webapp.ipynb"]:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", out)
