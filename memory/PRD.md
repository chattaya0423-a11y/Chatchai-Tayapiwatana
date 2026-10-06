# NK-Track Analytics — PRD

## Original problem statement
"Generate web app from this google colab file. Make the user friendly GUI and clear output."
Source: `NK_Target_Persistence_Analysis_Colab.ipynb` — "Persistent NK-proximal target occupation analysis".

## What the app does
A scientific dashboard that reproduces the Colab pipeline exactly: users upload live-cell
fluorescence microscopy movies (red channel = NK cells, green channel = target cells). The
engine segments cells, tracks target regions by nearest-centroid linking, detects proximity
contact (default ≤10 px), classifies persistent engagement episodes (default ≥4 consecutive
frames at 5 min/frame) and reports per-frame data, contact episodes, a summary table and the
analysis configuration.

Primary metric: `persistent-target fraction = persistent target tracks / eligible target tracks × 100`.

## Architecture
- **Backend** FastAPI (`/app/backend`): `analysis.py` (exact notebook port — segment_frame, assign_tracks,
  analyze_frames, summarize), `demo.py` (synthetic 2-condition generator run through the real pipeline),
  `server.py` (endpoints). Video decoding via `imageio` + `imageio-ffmpeg` (bundled binary, persists via pip).
  Results stored in MongoDB (`jobs` collection).
- **Frontend** React + Tailwind + recharts + framer-motion + lucide + sonner. Dark scientific theme.
  Components: UploadPanel, ResultsView, KpiCards, Charts, DataTables, HistoryBar, EmptyState, Header.

### API
- `GET /api/` health
- `POST /api/analyze` — multipart: `files` (1..n videos) + `payload` JSON (labels, params, sensitivity flag)
- `POST /api/demo` — synthetic demo experiment
- `GET /api/jobs` / `GET /api/jobs/{id}` / `DELETE /api/jobs/{id}`
- `GET /api/jobs/{id}/download?kind=summary|per_frame|episodes|config|zip`

## User personas
- Cell biologists / immunologists reproducing the published 10-px / 4-frame persistence analysis.
- Researchers comparing conditions (e.g. +antibody vs. control) and validating robustness via sensitivity.

## Core requirements (static)
- Faithful reproduction of notebook computation and output columns.
- Multi-video upload with editable condition labels.
- Adjustable parameters: contact px, stable frames, frame interval, max displacement.
- Clear outputs: KPI cards, charts, browsable tables, CSV/JSON/ZIP downloads.
- Optional 2/5/10 px threshold sensitivity.

## Implemented (2026-06)
- Full backend pipeline + demo + downloads (ZIP bundles all CSVs + config + sensitivity).
- Full dashboard: upload/params/run, KPIs, engaged-%-over-time + headline bars + sensitivity charts,
  5 tabbed tables (summary, per-frame, episodes, sensitivity, config), run history, `?job=` deep-link.
- Verified: backend 13/13 pytest pass; frontend E2E pass. Table-filter now applies across all tabs.

## Backlog / future (P1/P2)
- P1: Interactive frame viewer with cell-overlay (segmentation masks / contact lines).
- P2: Per-parameter re-run without re-upload (cache decoded frames server-side).
- P2: Numeric param validation (reject ≤0) and `kind` allow-list hardening on download endpoint.
- P2: Multi-run comparison view across separate jobs.

## Next tasks
- Await user feedback on real uploaded movies; tune segmentation thresholds if needed.
