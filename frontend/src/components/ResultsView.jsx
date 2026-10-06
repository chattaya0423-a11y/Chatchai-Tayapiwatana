import { motion } from "framer-motion";
import { Download, FileText, FileJson, Archive, Info, AlertTriangle } from "lucide-react";
import KpiCards from "@/components/KpiCards";
import Charts from "@/components/Charts";
import DataTables from "@/components/DataTables";
import { downloadUrl } from "@/lib/api";
import { prettyDate } from "@/lib/format";

const DOWNLOADS = [
  { kind: "summary", label: "Summary CSV", icon: FileText },
  { kind: "per_frame", label: "Per-frame CSV", icon: FileText },
  { kind: "episodes", label: "Episodes CSV", icon: FileText },
  { kind: "config", label: "Config JSON", icon: FileJson },
  { kind: "zip", label: "All (ZIP)", icon: Archive, primary: true },
];

export default function ResultsView({ job }) {
  const p = job.params;
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 sm:space-y-6">
      <div className="rounded-xl border border-border bg-card p-4 sm:p-5">
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
          <div className="min-w-0">
            <h2 className="font-display text-xl sm:text-2xl font-bold tracking-tight truncate">{job.name}</h2>
            <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 font-mono text-[11px] text-muted-foreground">
              <span>{prettyDate(job.created_at)}</span>
              <span className="text-border">·</span>
              <span>contact ≤ {p.contact_px}px</span>
              <span className="text-border">·</span>
              <span>≥ {p.stable_frames} frames</span>
              <span className="text-border">·</span>
              <span>{p.frame_interval_min} min/frame</span>
              <span className="text-border">·</span>
              <span>max disp {p.max_target_displacement_px}px</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {DOWNLOADS.map((d) => (
              <a
                key={d.kind}
                data-testid={`download-${d.kind}`}
                href={downloadUrl(job.id, d.kind)}
                className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                  d.primary
                    ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25"
                    : "border-border bg-background/50 text-muted-foreground hover:text-foreground hover:bg-accent"
                }`}
              >
                <d.icon className="w-3.5 h-3.5" />
                {d.label}
              </a>
            ))}
          </div>
        </div>
      </div>

      <KpiCards summary={job.summary} />

      {job.warning && (
        <div data-testid="results-warning" className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-amber-200/90 leading-relaxed">{job.warning}</div>
        </div>
      )}

      <Charts job={job} />
      <DataTables job={job} />

      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4 sm:p-5">
        <div className="flex gap-3">
          <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
          <div className="text-xs text-muted-foreground leading-relaxed">
            <span className="text-cyan-300 font-semibold">Primary metric:</span>{" "}
            <span className="font-mono">persistent-target fraction = persistent target tracks / eligible target tracks × 100</span>.
            Four consecutive frames correspond to a 15-minute observed span at 5-minute intervals. The endpoint is{" "}
            <span className="text-foreground font-medium">persistent NK-proximal target occupation</span>, not definitive
            ultrastructural membrane contact. Independent movies, experiments and donors are required for inferential
            statistics; frames and target tracks are not independent biological replicates.
          </div>
        </div>
      </div>
    </motion.div>
  );
}
