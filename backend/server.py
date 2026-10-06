from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import io
import json
import shutil
import zipfile
import asyncio
import logging
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

WORK_DIR = Path("/tmp/nk_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)

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

    return summary, per_frame, episodes, (sensitivity if do_sensitivity else None)


def _decode_dir(job_dir: Path, saved):
    """saved: list of (label, source, filename). Returns decoded frames + total detections."""
    decoded = []
    for label, source, fname in saved:
        frames = analysis.read_video_rgb(job_dir / fname)
        decoded.append((label, source, frames))
    return decoded


async def process_job(job_id, saved, params, do_sensitivity, thresholds):
    job_dir = WORK_DIR / job_id
    try:
        await db.jobs.update_one({"id": job_id}, {"$set": {"stage": "Decoding video frames"}})
        decoded = await asyncio.to_thread(_decode_dir, job_dir, saved)

        total_frames = sum(len(fr) for _, _, fr in decoded)
        await db.jobs.update_one(
            {"id": job_id},
            {"$set": {"stage": f"Segmenting & tracking · {total_frames} frames"}},
        )
        doc = await db.jobs.find_one({"id": job_id}, {"name": 1})
        summary, per_frame, episodes, sensitivity = await asyncio.to_thread(
            _run_pipeline, doc["name"], decoded, params, do_sensitivity, thresholds
        )

        total_detected = sum(r["target_regions_detected"] for r in per_frame)
        total_nk = sum(r["nk_objects_detected"] for r in per_frame)
        warning = None
        if total_detected == 0 or total_nk == 0:
            warning = (
                "No red NK objects and/or green target regions were detected with the manuscript "
                "thresholds. Confirm the movie has NK cells in the RED channel and target cells in "
                "the GREEN channel."
            )

        await db.jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": "done",
                "stage": "Complete",
                "summary": summary,
                "per_frame": per_frame,
                "episodes": episodes,
                "sensitivity": sensitivity,
                "warning": warning,
            }},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("job %s failed", job_id)
        await db.jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "error", "stage": "Failed", "error": str(exc)[:500]}},
        )
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


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
        "contact_px": max(0.5, float(meta.get("contact_px", 10.0))),
        "stable_frames": max(1, int(meta.get("stable_frames", 4))),
        "frame_interval_min": max(0.1, float(meta.get("frame_interval_min", 5.0))),
        "max_target_displacement_px": max(1.0, float(meta.get("max_target_displacement_px", 100.0))),
    }
    do_sensitivity = bool(meta.get("sensitivity", False))
    thresholds = meta.get("sensitivity_thresholds") or [2, 5, 10]
    name = meta.get("name") or "Untitled analysis"

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    videos = {}
    for idx, up in enumerate(files):
        source = up.filename or f"video_{idx + 1}"
        fname = f"{idx}_{Path(source).name}"
        (job_dir / fname).write_bytes(await up.read())
        saved.append((labels[idx], source, fname))
        videos[labels[idx]] = source

    await db.jobs.insert_one({
        "id": job_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "videos": videos,
        "config": analysis.build_config(params, videos),
        "status": "processing",
        "stage": "Queued",
        "summary": None, "per_frame": None, "episodes": None, "sensitivity": None,
    })

    asyncio.create_task(process_job(job_id, saved, params, do_sensitivity, thresholds))
    return {"id": job_id, "status": "processing"}


@api_router.post("/demo")
async def run_demo():
    decoded = await asyncio.to_thread(demo_mod.demo_conditions)
    params = {
        "contact_px": 10.0, "stable_frames": 4,
        "frame_interval_min": 5.0, "max_target_displacement_px": 100.0,
    }
    summary, per_frame, episodes, sensitivity = await asyncio.to_thread(
        _run_pipeline, "Demo experiment (synthetic)", decoded, params, True, [2, 5, 10]
    )
    videos = {label: source for label, source, _ in decoded}
    job = {
        "id": str(uuid.uuid4()),
        "name": "Demo experiment (synthetic)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "videos": videos,
        "config": analysis.build_config(params, videos),
        "status": "done",
        "stage": "Complete",
        "summary": summary,
        "per_frame": per_frame,
        "episodes": episodes,
        "sensitivity": sensitivity,
        "warning": None,
    }
    await db.jobs.insert_one({**job})
    job.pop("_id", None)
    return job


@api_router.get("/jobs")
async def list_jobs():
    docs = await db.jobs.find(
        {}, {"_id": 0, "id": 1, "name": 1, "created_at": 1, "videos": 1, "status": 1, "params": 1}
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
    df = pd.DataFrame(rows or [])
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
    if doc.get("status") != "done":
        raise HTTPException(status_code=409, detail="Analysis not finished yet.")

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
