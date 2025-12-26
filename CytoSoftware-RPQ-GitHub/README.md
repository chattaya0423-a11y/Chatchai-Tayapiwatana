# CytoSoftware-RPQ (Red Particle Quantification)

**CytoSoftware-RPQ (RPQ)** detects and counts red fluorescent particles (spots/blobs) in MP4 time‑lapse microscopy videos **without cell segmentation**, and exports:
- `red_particles_by_time.csv` (counts vs time)
- `red_blob_features.csv` (per‑blob features)
- `annotated_red_particles.mp4` (detections overlaid)
- `run_config.json` (run parameters for reproducibility)

## Repository layout

- `src/rpq/` : notebook‑independent core + a small CLI
- `notebooks/` : the interactive Jupyter/Colab notebook + its Colab-exported `.py`
- `docs/` : user manual (DOCX)

## Quick start (Notebook / Colab)

Open `notebooks/CytoSoftware-RPQ.ipynb` in **Google Colab** (recommended) or local Jupyter.

Workflow: **Apply & Preview → Run Analysis**.

## Quick start (Local CLI)

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) Run
```bash
python -m rpq.cli --video /path/to/video.mp4 --out rpq_out
```

Common tuning flags:
```bash
python -m rpq.cli --video video.mp4 --out rpq_out \
  --bg-method rolling_ball --rb-radius 25 --bg-sigma 8 \
  --red-dom 1.6 --red-min 0.08 --robust-k 3.0 \
  --sigma-min 1.0 --sigma-max 4.5 --num-sigmas 8 \
  --area-min 6 --area-max 260
```

## Notes

- The notebook contains Colab widgets and `google.colab` helpers; the **CLI and `src/rpq/` code do not**.
- If you want a fully local notebook experience, remove/guard `google.colab` imports in the notebook (optional).

## Citation

See `CITATION.cff` (edit as needed).
