from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import io
import json
import zipfile
import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import pandas as pd

import analysis
import demo as demo_mod

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI(title="NK-Track Analytics")
api_router = APIRouter(prefix="/api")

logger = logging.getLogger("nk_track")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

SUMMARY_COLS = [
    "label", "video_path", "frames", "mean_target_regions", "mean_engaged_target_percent",
    "all_contact_episode_count", "persistent_target_tracks",
    "eligible_target_tracks_observed_4plus_frames", "persistent_target_fraction_percent",
    "persistent_episode_count", "mean_persistent_duration_min",
]
PER_FRAME_COLS = [
    "label", "frame", "time_min", "target_regions_detected", "target_regions_engaged",
    "engaged_target_percent", "nk_objects_detected",
]
EPISODE_COLS = [
    "label", "target_track_id", "start_frame", "end_frame", "n_frames",
    "duration_min", "sampled_occupancy_min", "persistent",
]


def _run_pipeline(name, decoded, params, do_sensitivity, sensitivity_thresholds):
    """decoded: list of (label, source_name, frames). Runs synchronously (in thread)."""
    summary, per_frame, episodes = [], [], []
    for label, source, frames in decoded:
        fr, ep, elig, pers = analysis.analyze_frames(
            label, frames,
            contact_px=params["contact_px"],
            stable_frames=params["stable_frames"],
            frame_interval_min=params["frame_interval_min"],
            max_target_displacement_px=params["max_target_displacement_px"],
        )
        per_frame.extend(fr)
        episodes.extend(ep)
        summary.append(analysis.summarize(label, source, fr, ep, elig, pers))

    sensitivity = []
    if do_sensitivity:
        for thr in sensitivity_thresholds:
            for label, source, frames in decoded:
                fr, ep, elig, pers = analysis.analyze_frames(
                    label, frames,
                    contact_px=thr,
                    stable_frames=params["stable_frames"],
                    frame_interval_min=params["frame_interval_min"],
                    max_target_displacement_px=params["max_target_displacement_px"],
                )
                row = analysis.summarize(label, source, fr, ep, elig, pers)
                sensitivity.append({"contact_px": thr, **row})

    videos = {label: source for label, source, _ in decoded}
    job = {
        "id": str(uuid.uuid4()),
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "videos": videos,
        "config": analysis.build_config(params, videos),
        "summary": summary,
        "per_frame": per_frame,
        "episodes": episodes,
        "sensitivity": sensitivity if do_sensitivity else None,
    }
    return job


def _decode_uploads(saved):
    decoded = []
    for label, source, path in saved:
        frames = analysis.read_video_rgb(path)
        decoded.append((label, source, frames))
    return decoded


@api_router.get("/")
async def root():
    return {"message": "NK-Track Analytics API"}


@api_router.post("/analyze")
async def analyze_endpoint(files: List[UploadFile] = File(...), payload: str = Form(...)):
    try:
        meta = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid payload JSON.")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Upload at least one video file.")

    labels = meta.get("labels") or []
    if len(labels) != len(files):
        labels = [f"S{i + 1}" for i in range(len(files))]

    params = {
        "contact_px": float(meta.get("contact_px", 10.0)),
        "stable_frames": int(meta.get("stable_frames", 4)),
        "frame_interval_min": float(meta.get("frame_interval_min", 5.0)),
        "max_target_displacement_px": float(meta.get("max_target_displacement_px", 100.0)),
    }
    do_sensitivity = bool(meta.get("sensitivity", False))
    sensitivity_thresholds = meta.get("sensitivity_thresholds") or [2, 5, 10]
    name = meta.get("name") or "Untitled analysis"

    with tempfile.TemporaryDirectory(prefix="nk_up_") as tmp:
        saved = []
        for idx, up in enumerate(files):
            source = up.filename or f"video_{idx + 1}"
            dest = Path(tmp) / f"{idx}_{source}"
            dest.write_bytes(await up.read())
            saved.append((labels[idx], source, dest))
        try:
            decoded = await asyncio.to_thread(_decode_uploads, saved)
        except Exception as exc:  # noqa: BLE001
            logger.exception("decode failed")
            raise HTTPException(status_code=422, detail=f"Could not decode a video: {exc}")

        try:
            job = await asyncio.to_thread(
                _run_pipeline, name, decoded, params, do_sensitivity, sensitivity_thresholds
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("analysis failed")
            raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")

    await db.jobs.insert_one({**job})
    job.pop("_id", None)
    return job


@api_router.post("/demo")
async def run_demo():
    decoded = await asyncio.to_thread(demo_mod.demo_conditions)
    params = {
        "contact_px": 10.0, "stable_frames": 4,
        "frame_interval_min": 5.0, "max_target_displacement_px": 100.0,
    }
    job = await asyncio.to_thread(
        _run_pipeline, "Demo experiment (synthetic)", decoded, params, True, [2, 5, 10]
    )
    await db.jobs.insert_one({**job})
    job.pop("_id", None)
    return job


@api_router.get("/jobs")
async def list_jobs():
    docs = await db.jobs.find(
        {}, {"_id": 0, "id": 1, "name": 1, "created_at": 1, "videos": 1, "summary": 1, "params": 1}
    ).sort("created_at", -1).to_list(100)
    return docs


@api_router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    doc = await db.jobs.find_one({"id": job_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found.")
    return doc


@api_router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    res = await db.jobs.delete_one({"id": job_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"deleted": True}


def _df(rows, cols):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    extra = [c for c in df.columns if c not in cols]
    ordered = [c for c in cols if c in df.columns] + extra
    return df[ordered]


def _csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    data = df.to_csv(index=False).encode()
    return StreamingResponse(
        io.BytesIO(data), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@api_router.get("/jobs/{job_id}/download")
async def download(job_id: str, kind: str = "zip"):
    doc = await db.jobs.find_one({"id": job_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found.")

    summary = _df(doc.get("summary", []), SUMMARY_COLS)
    per_frame = _df(doc.get("per_frame", []), PER_FRAME_COLS)
    episodes = _df(doc.get("episodes", []), EPISODE_COLS)
    config_bytes = json.dumps(doc.get("config", {}), indent=2).encode()

    if kind == "summary":
        return _csv_response(summary, "target_persistence_summary.csv")
    if kind == "per_frame":
        return _csv_response(per_frame, "target_persistence_per_frame.csv")
    if kind == "episodes":
        return _csv_response(episodes, "target_persistence_episodes.csv")
    if kind == "config":
        return StreamingResponse(
            io.BytesIO(config_bytes), media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=analysis_config.json"},
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("target_persistence_summary.csv", summary.to_csv(index=False))
        zf.writestr("target_persistence_per_frame.csv", per_frame.to_csv(index=False))
        zf.writestr("target_persistence_episodes.csv", episodes.to_csv(index=False))
        zf.writestr("analysis_config.json", config_bytes)
        if doc.get("sensitivity"):
            sens = _df(doc["sensitivity"], ["contact_px"] + SUMMARY_COLS)
            zf.writestr("threshold_sensitivity.csv", sens.to_csv(index=False))
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=nk_persistence_results.zip"},
    )


app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
