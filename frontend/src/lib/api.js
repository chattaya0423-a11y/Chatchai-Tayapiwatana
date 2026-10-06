import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

export const CONDITION_COLORS = [
  "#10B981", "#06B6D4", "#F59E0B", "#A855F7", "#F43F5E", "#38BDF8", "#FB923C", "#4ADE80",
];

export async function postAnalyze(files, meta, onProgress) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("payload", JSON.stringify(meta));
  const { data } = await axios.post(`${API}/analyze`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Poll a processing job until it finishes. Calls onStage(stage) on updates.
export async function pollJob(id, onStage, intervalMs = 1500, maxMs = 1800000) {
  const start = Date.now();
  let last = "";
  while (Date.now() - start < maxMs) {
    const job = await getJob(id);
    if (onStage && job.stage && job.stage !== last) {
      last = job.stage;
      onStage(job.stage);
    }
    if (job.status === "done") return job;
    if (job.status === "error") {
      const err = new Error(job.error || "Analysis failed");
      err.jobError = true;
      throw err;
    }
    await sleep(intervalMs);
  }
  throw new Error("Analysis timed out.");
}

export async function postDemo() {
  const { data } = await axios.post(`${API}/demo`);
  return data;
}

export async function listJobs() {
  const { data } = await axios.get(`${API}/jobs`);
  return data;
}

export async function getJob(id) {
  const { data } = await axios.get(`${API}/jobs/${id}`);
  return data;
}

export async function removeJob(id) {
  const { data } = await axios.delete(`${API}/jobs/${id}`);
  return data;
}

export function downloadUrl(id, kind) {
  return `${API}/jobs/${id}/download?kind=${kind}`;
}
