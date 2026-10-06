"""Backend API tests for NK-Track Analytics."""
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL")
if not BASE_URL:
    # Fallback: read from frontend/.env
    env_path = Path("/app/frontend/.env")
    for line in env_path.read_text().splitlines():
        if line.startswith("REACT_APP_BACKEND_URL="):
            BASE_URL = line.split("=", 1)[1].strip()
            break
BASE_URL = BASE_URL.rstrip("/")
API = f"{BASE_URL}/api"
SAMPLE = Path("/app/test_assets/sample_nk_movie.mp4")


@pytest.fixture(scope="session")
def created_ids():
    ids = []
    yield ids
    # Cleanup
    for jid in ids:
        try:
            requests.delete(f"{API}/jobs/{jid}", timeout=15)
        except Exception:
            pass


# ---------- health ----------
def test_health():
    r = requests.get(f"{API}/", timeout=15)
    assert r.status_code == 200
    assert "message" in r.json()


# ---------- demo ----------
def test_demo(created_ids):
    r = requests.post(f"{API}/demo", timeout=120)
    assert r.status_code == 200, r.text
    job = r.json()
    for k in ("id", "name", "params", "videos", "config", "summary", "per_frame", "episodes", "sensitivity"):
        assert k in job, f"missing {k}"
    assert len(job["summary"]) == 2
    assert len(job["per_frame"]) == 40
    assert isinstance(job["episodes"], list)
    assert len(job["sensitivity"]) == 6  # 3 thresholds x 2 conditions
    for row in job["summary"]:
        for key in ("persistent_target_fraction_percent",
                    "eligible_target_tracks_observed_4plus_frames",
                    "mean_engaged_target_percent"):
            assert key in row
            assert isinstance(row[key], (int, float))
    created_ids.append(job["id"])
    pytest.demo_job_id = job["id"]


# ---------- analyze single ----------
def test_analyze_single_upload(created_ids):
    assert SAMPLE.exists() and SAMPLE.stat().st_size > 0
    payload = {
        "name": "QA run",
        "labels": ["T1"],
        "contact_px": 10,
        "stable_frames": 4,
        "frame_interval_min": 5,
        "max_target_displacement_px": 100,
        "sensitivity": True,
        "sensitivity_thresholds": [2, 5, 10],
    }
    with SAMPLE.open("rb") as f:
        files = [("files", (SAMPLE.name, f.read(), "video/mp4"))]
    data = {"payload": json.dumps(payload)}
    r = requests.post(f"{API}/analyze", files=files, data=data, timeout=300)
    assert r.status_code == 200, r.text
    job = r.json()
    assert len(job["summary"]) == 1
    assert job["summary"][0]["label"] == "T1"
    assert len(job["per_frame"]) > 0
    assert isinstance(job["episodes"], list)
    assert job["sensitivity"] and len(job["sensitivity"]) == 3
    row = job["summary"][0]
    for key in ("persistent_target_fraction_percent",
                "eligible_target_tracks_observed_4plus_frames",
                "mean_engaged_target_percent",
                "persistent_target_tracks"):
        assert isinstance(row[key], (int, float)), f"{key} not numeric"
    created_ids.append(job["id"])
    pytest.single_job_id = job["id"]


# ---------- analyze two files ----------
def test_analyze_two_files(created_ids):
    with SAMPLE.open("rb") as f:
        content = f.read()
    files = [
        ("files", ("movie1.mp4", content, "video/mp4")),
        ("files", ("movie2.mp4", content, "video/mp4")),
    ]
    payload = {"name": "Pair", "labels": ["S1", "S2"], "sensitivity": False}
    r = requests.post(f"{API}/analyze",
                      files=files, data={"payload": json.dumps(payload)}, timeout=300)
    assert r.status_code == 200, r.text
    job = r.json()
    assert len(job["summary"]) == 2
    assert {s["label"] for s in job["summary"]} == {"S1", "S2"}
    created_ids.append(job["id"])


# ---------- list jobs ----------
def test_list_jobs(created_ids):
    r = requests.get(f"{API}/jobs", timeout=15)
    assert r.status_code == 200
    jobs = r.json()
    assert isinstance(jobs, list)
    ids = {j["id"] for j in jobs}
    for jid in created_ids:
        assert jid in ids
    # sorted descending by created_at
    times = [j["created_at"] for j in jobs]
    assert times == sorted(times, reverse=True)


# ---------- get job ----------
def test_get_job_detail():
    jid = getattr(pytest, "demo_job_id", None)
    assert jid
    r = requests.get(f"{API}/jobs/{jid}", timeout=15)
    assert r.status_code == 200
    doc = r.json()
    assert doc["id"] == jid
    assert "per_frame" in doc


def test_get_job_404():
    r = requests.get(f"{API}/jobs/nonexistent-id", timeout=15)
    assert r.status_code == 404


# ---------- downloads ----------
@pytest.mark.parametrize("kind,ctype", [
    ("summary", "text/csv"),
    ("per_frame", "text/csv"),
    ("episodes", "text/csv"),
    ("config", "application/json"),
])
def test_download_kinds(kind, ctype):
    jid = getattr(pytest, "demo_job_id", None)
    assert jid
    r = requests.get(f"{API}/jobs/{jid}/download", params={"kind": kind}, timeout=30)
    assert r.status_code == 200
    assert ctype in r.headers.get("content-type", "")
    assert len(r.content) > 0


def test_download_zip():
    jid = getattr(pytest, "demo_job_id", None)
    assert jid
    r = requests.get(f"{API}/jobs/{jid}/download", params={"kind": "zip"}, timeout=30)
    assert r.status_code == 200
    assert "application/zip" in r.headers.get("content-type", "")
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    expected = {
        "target_persistence_summary.csv",
        "target_persistence_per_frame.csv",
        "target_persistence_episodes.csv",
        "analysis_config.json",
        "threshold_sensitivity.csv",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


# ---------- delete ----------
def test_delete_job():
    # Create a disposable demo job then delete it
    r = requests.post(f"{API}/demo", timeout=120)
    jid = r.json()["id"]
    d = requests.delete(f"{API}/jobs/{jid}", timeout=15)
    assert d.status_code == 200
    assert d.json() == {"deleted": True}
    # confirm 404 afterwards
    g = requests.get(f"{API}/jobs/{jid}", timeout=15)
    assert g.status_code == 404
