"""Backend API tests for NK-Track Analytics (async analyze + polling)."""
import io
import json
import os
import time
import zipfile
from pathlib import Path

import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL")
if not BASE_URL:
    env_path = Path("/app/frontend/.env")
    for line in env_path.read_text().splitlines():
        if line.startswith("REACT_APP_BACKEND_URL="):
            BASE_URL = line.split("=", 1)[1].strip()
            break
BASE_URL = BASE_URL.rstrip("/")
API = f"{BASE_URL}/api"
SAMPLE = Path("/app/test_assets/sample_nk_movie.mp4")
BIG = Path("/app/test_assets/big.mp4")
HEAVY = Path("/app/test_assets/heavy.mp4")


def _poll(job_id, timeout=180, interval=2.0):
    """Poll /api/jobs/{id} until status in {done, error} or timeout."""
    start = time.time()
    last = None
    while time.time() - start < timeout:
        r = requests.get(f"{API}/jobs/{job_id}", timeout=30)
        assert r.status_code == 200, r.text
        last = r.json()
        if last.get("status") in ("done", "error"):
            return last
        time.sleep(interval)
    raise AssertionError(f"Polling timed out after {timeout}s. Last status: {last and last.get('status')}, stage={last and last.get('stage')}")


@pytest.fixture(scope="session")
def created_ids():
    ids = []
    yield ids
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


# ---------- demo (synchronous) ----------
def test_demo(created_ids):
    r = requests.post(f"{API}/demo", timeout=180)
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "done"
    for k in ("id", "name", "params", "videos", "config", "summary", "per_frame", "episodes", "sensitivity"):
        assert k in job, f"missing {k}"
    assert len(job["summary"]) == 2
    assert len(job["per_frame"]) == 40
    assert isinstance(job["episodes"], list)
    assert len(job["sensitivity"]) == 6
    for row in job["summary"]:
        for key in ("persistent_target_fraction_percent",
                    "eligible_target_tracks_observed_4plus_frames",
                    "mean_engaged_target_percent"):
            assert key in row
            assert isinstance(row[key], (int, float))
    created_ids.append(job["id"])
    pytest.demo_job_id = job["id"]


# ---------- async analyze: fast submit ----------
def test_analyze_submit_is_fast_and_returns_processing(created_ids):
    assert SAMPLE.exists() and SAMPLE.stat().st_size > 0
    payload = {
        "name": "QA fast submit",
        "labels": ["T1"],
        "contact_px": 10,
        "stable_frames": 4,
        "frame_interval_min": 5,
        "max_target_displacement_px": 100,
        "sensitivity": False,
    }
    with SAMPLE.open("rb") as f:
        files = [("files", (SAMPLE.name, f.read(), "video/mp4"))]
    t0 = time.time()
    r = requests.post(f"{API}/analyze", files=files,
                      data={"payload": json.dumps(payload)}, timeout=30)
    dt = time.time() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert "id" in body
    assert body.get("status") == "processing"
    assert dt < 10, f"Submit was slow ({dt:.1f}s); should be well under 5s but allow network slack"
    created_ids.append(body["id"])
    # poll to completion so the job is DONE for subsequent checks
    final = _poll(body["id"], timeout=180)
    assert final["status"] == "done", f"job errored: {final.get('error')}"
    assert len(final["summary"]) == 1
    assert final["summary"][0]["label"] == "T1"
    assert len(final["per_frame"]) > 0
    pytest.small_job_id = body["id"]


# ---------- async analyze heavy with sensitivity ----------
def test_analyze_heavy_async_with_sensitivity(created_ids):
    if not HEAVY.exists() or HEAVY.stat().st_size == 0:
        pytest.skip("heavy.mp4 not available")
    payload = {
        "name": "QA async",
        "labels": ["H1"],
        "contact_px": 10,
        "stable_frames": 4,
        "frame_interval_min": 5,
        "max_target_displacement_px": 100,
        "sensitivity": True,
        "sensitivity_thresholds": [2, 5, 10],
    }
    with HEAVY.open("rb") as f:
        files = [("files", (HEAVY.name, f.read(), "video/mp4"))]
    t0 = time.time()
    r = requests.post(f"{API}/analyze", files=files,
                      data={"payload": json.dumps(payload)}, timeout=120)
    dt = time.time() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "processing"
    assert "id" in body
    # upload+submit can take a bit for a large file but response itself fast
    print(f"Heavy submit took {dt:.1f}s (includes upload)")
    created_ids.append(body["id"])

    # observe stage progression
    saw_stage = False
    r1 = requests.get(f"{API}/jobs/{body['id']}", timeout=15)
    if r1.status_code == 200 and r1.json().get("stage"):
        saw_stage = True

    final = _poll(body["id"], timeout=240, interval=3.0)
    assert final["status"] == "done", f"error: {final.get('error')}"
    assert final.get("stage") == "Complete"
    assert len(final["summary"]) == 1
    assert len(final["per_frame"]) > 0
    assert isinstance(final["episodes"], list)
    assert final["sensitivity"] is not None
    assert len(final["sensitivity"]) == 3
    assert saw_stage, "expected a 'stage' string present during processing"


# ---------- two file async ----------
def test_analyze_two_files_async(created_ids):
    if not BIG.exists() or BIG.stat().st_size == 0:
        pytest.skip("big.mp4 not available")
    with SAMPLE.open("rb") as fs, BIG.open("rb") as fb:
        files = [
            ("files", (SAMPLE.name, fs.read(), "video/mp4")),
            ("files", (BIG.name, fb.read(), "video/mp4")),
        ]
    payload = {"name": "Pair", "labels": ["S1", "S2"], "sensitivity": False}
    r = requests.post(f"{API}/analyze", files=files,
                      data={"payload": json.dumps(payload)}, timeout=120)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "processing"
    created_ids.append(body["id"])
    final = _poll(body["id"], timeout=240, interval=3.0)
    assert final["status"] == "done", f"error: {final.get('error')}"
    assert len(final["summary"]) == 2
    assert {s["label"] for s in final["summary"]} == {"S1", "S2"}


# ---------- list jobs with status ----------
def test_list_jobs_has_status(created_ids):
    r = requests.get(f"{API}/jobs", timeout=15)
    assert r.status_code == 200
    jobs = r.json()
    assert isinstance(jobs, list)
    assert len(jobs) > 0
    for j in jobs[:5]:
        assert "status" in j
    ids = {j["id"] for j in jobs}
    for jid in created_ids:
        assert jid in ids


def test_get_job_404():
    r = requests.get(f"{API}/jobs/nonexistent-id", timeout=15)
    assert r.status_code == 404


# ---------- download guard: 409 for processing ----------
def test_download_409_when_processing(created_ids):
    # Submit a heavy-ish job and immediately try to download before done
    source = HEAVY if HEAVY.exists() else SAMPLE
    with source.open("rb") as f:
        files = [("files", (source.name, f.read(), "video/mp4"))]
    payload = {"name": "409 test", "labels": ["X"], "sensitivity": True,
               "sensitivity_thresholds": [2, 5, 10]}
    r = requests.post(f"{API}/analyze", files=files,
                      data={"payload": json.dumps(payload)}, timeout=120)
    assert r.status_code == 200
    jid = r.json()["id"]
    created_ids.append(jid)
    # try download immediately
    time.sleep(0.5)
    d = requests.get(f"{API}/jobs/{jid}/download", params={"kind": "zip"}, timeout=20)
    # If job finished extremely fast, allow 200; else expect 409
    assert d.status_code in (409, 200), f"unexpected {d.status_code}: {d.text}"
    if d.status_code == 200:
        pytest.skip("Job completed too fast to observe 409 guard")
    # poll until done for cleanup
    _poll(jid, timeout=240, interval=3.0)


# ---------- downloads for done job ----------
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


def test_download_zip_has_5_files():
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
    r = requests.post(f"{API}/demo", timeout=180)
    jid = r.json()["id"]
    d = requests.delete(f"{API}/jobs/{jid}", timeout=15)
    assert d.status_code == 200
    assert d.json() == {"deleted": True}
    g = requests.get(f"{API}/jobs/{jid}", timeout=15)
    assert g.status_code == 404
