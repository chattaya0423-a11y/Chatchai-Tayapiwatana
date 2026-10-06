import { useEffect, useState, useCallback } from "react";
import "@/App.css";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import Header from "@/components/Header";
import UploadPanel from "@/components/UploadPanel";
import ResultsView from "@/components/ResultsView";
import EmptyState from "@/components/EmptyState";
import HistoryBar from "@/components/HistoryBar";
import { postAnalyze, postDemo, listJobs, getJob, removeJob, pollJob } from "@/lib/api";

function App() {
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [history, setHistory] = useState([]);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await listJobs());
    } catch {
      /* non-blocking */
    }
  }, []);

  useEffect(() => {
    refreshHistory();
    const id = new URLSearchParams(window.location.search).get("job");
    if (id) openJob(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshHistory]);

  const runAnalyze = async (files, meta) => {
    setBusy(true);
    setProgress(0);
    setStage("Uploading movies");
    try {
      const { id } = await postAnalyze(files, meta, (p) => {
        setProgress(p);
        if (p >= 100) setStage("Decoding video frames");
      });
      refreshHistory();
      const result = await pollJob(id, (s) => setStage(s));
      setJob(result);
      if (result.warning) {
        toast.warning("Analysis finished with a warning", { description: result.warning });
      } else {
        toast.success("Analysis complete", { description: `${result.summary.length} condition(s) processed.` });
      }
      refreshHistory();
    } catch (e) {
      const detail = e?.jobError ? e.message : (e?.response?.data?.detail || e.message);
      toast.error("Analysis failed", { description: String(detail) });
    } finally {
      setBusy(false);
      setStage("");
    }
  };

  const runDemo = async () => {
    setBusy(true);
    setStage("Generating synthetic experiment");
    try {
      const result = await postDemo();
      setJob(result);
      toast.success("Demo experiment loaded", { description: "Synthetic NK + target movies analysed." });
      refreshHistory();
    } catch (e) {
      toast.error("Could not load demo", { description: e?.response?.data?.detail || e.message });
    } finally {
      setBusy(false);
      setStage("");
    }
  };

  const openJob = async (id) => {
    setBusy(true);
    setStage("Loading run");
    try {
      let result = await getJob(id);
      if (result.status === "processing") {
        result = await pollJob(id, (s) => setStage(s));
      } else if (result.status === "error") {
        toast.error("That run failed", { description: result.error || "" });
        return;
      }
      setJob(result);
    } catch (e) {
      toast.error("Could not load that run", { description: e?.message });
    } finally {
      setBusy(false);
      setStage("");
    }
  };

  const deleteJob = async (id) => {
    try {
      await removeJob(id);
      if (job?.id === id) setJob(null);
      refreshHistory();
      toast.success("Run deleted");
    } catch {
      toast.error("Delete failed");
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header onDemo={runDemo} busy={busy} job={job} />
      <main className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">
          <div className="space-y-6">
            <UploadPanel onRun={runAnalyze} onDemo={runDemo} busy={busy} progress={progress} stage={stage} />
            <HistoryBar history={history} activeId={job?.id} onOpen={openJob} onDelete={deleteJob} />
          </div>
          <div>
            {job ? <ResultsView job={job} /> : <EmptyState busy={busy} stage={stage} />}
          </div>
        </div>
      </main>
      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;
