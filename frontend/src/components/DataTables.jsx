import { useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { fmt, fmtInt } from "@/lib/format";
import { CONDITION_COLORS } from "@/lib/api";

const num = (v, d = 2) => (v === null || v === undefined ? "—" : fmt(v, d));

function Dot({ label, labels }) {
  const i = labels.indexOf(label);
  const color = CONDITION_COLORS[(i < 0 ? 0 : i) % CONDITION_COLORS.length];
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-xs">
      <span className="w-2 h-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

function ScrollTable({ head, children, testid }) {
  return (
    <div data-testid={testid} className="rounded-lg border border-border overflow-hidden">
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-left">
          <thead className="sticky top-0 bg-secondary z-10">
            <tr>{head.map((h) => (
              <th key={h} className="px-3 py-2.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground whitespace-nowrap">{h}</th>
            ))}</tr>
          </thead>
          <tbody className="divide-y divide-border">{children}</tbody>
        </table>
      </div>
    </div>
  );
}

const Td = ({ children, mono = true, className = "" }) => (
  <td className={`px-3 py-2 ${mono ? "font-mono" : ""} text-xs whitespace-nowrap ${className}`}>{children}</td>
);

export default function DataTables({ job }) {
  const labels = job.summary.map((s) => s.label);
  const [q, setQ] = useState("");

  const frames = job.per_frame.filter((r) => r.label.toLowerCase().includes(q.toLowerCase()));
  const episodes = job.episodes.filter((r) => r.label.toLowerCase().includes(q.toLowerCase()));

  return (
    <div className="rounded-xl border border-border bg-card p-4 sm:p-5">
      <Tabs defaultValue="summary">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-4">
          <TabsList data-testid="tables-tablist" className="bg-secondary flex-wrap h-auto">
            <TabsTrigger value="summary" data-testid="tab-summary">Summary</TabsTrigger>
            <TabsTrigger value="per_frame" data-testid="tab-per-frame">Per-frame</TabsTrigger>
            <TabsTrigger value="episodes" data-testid="tab-episodes">Episodes</TabsTrigger>
            {job.sensitivity?.length ? <TabsTrigger value="sensitivity" data-testid="tab-sensitivity">Sensitivity</TabsTrigger> : null}
            <TabsTrigger value="config" data-testid="tab-config">Config</TabsTrigger>
          </TabsList>
          <Input
            data-testid="table-filter"
            placeholder="Filter by condition…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="h-8 w-full sm:w-44 text-xs bg-background sm:ml-auto"
          />
        </div>

        <TabsContent value="summary">
          <ScrollTable testid="summary-table" head={["Condition", "Frames", "Mean targets", "Mean engaged %", "Episodes", "Persistent tracks", "Eligible", "Persistent %", "Persistent eps", "Mean dur (min)"]}>
            {job.summary.map((s) => (
              <tr key={s.label} className="hover:bg-accent/40">
                <Td><Dot label={s.label} labels={labels} /></Td>
                <Td>{fmtInt(s.frames)}</Td>
                <Td>{num(s.mean_target_regions)}</Td>
                <Td className="text-emerald-300">{num(s.mean_engaged_target_percent)}</Td>
                <Td>{fmtInt(s.all_contact_episode_count)}</Td>
                <Td>{fmtInt(s.persistent_target_tracks)}</Td>
                <Td>{fmtInt(s.eligible_target_tracks_observed_4plus_frames)}</Td>
                <Td className="text-cyan-300 font-semibold">{num(s.persistent_target_fraction_percent)}</Td>
                <Td>{fmtInt(s.persistent_episode_count)}</Td>
                <Td>{num(s.mean_persistent_duration_min)}</Td>
              </tr>
            ))}
          </ScrollTable>
        </TabsContent>

        <TabsContent value="per_frame">
          <ScrollTable testid="per-frame-table" head={["Condition", "Frame", "Time (min)", "Targets", "Engaged", "Engaged %", "NK objects"]}>
            {frames.map((r, i) => (
              <tr key={i} className="hover:bg-accent/40">
                <Td><Dot label={r.label} labels={labels} /></Td>
                <Td>{r.frame}</Td>
                <Td>{num(r.time_min)}</Td>
                <Td>{fmtInt(r.target_regions_detected)}</Td>
                <Td>{fmtInt(r.target_regions_engaged)}</Td>
                <Td className="text-emerald-300">{num(r.engaged_target_percent)}</Td>
                <Td>{fmtInt(r.nk_objects_detected)}</Td>
              </tr>
            ))}
          </ScrollTable>
        </TabsContent>

        <TabsContent value="episodes">
          <ScrollTable testid="episodes-table" head={["Condition", "Track ID", "Start", "End", "Frames", "Duration (min)", "Occupancy (min)", "Persistent"]}>
            {episodes.map((r, i) => (
              <tr key={i} className="hover:bg-accent/40">
                <Td><Dot label={r.label} labels={labels} /></Td>
                <Td>{r.target_track_id}</Td>
                <Td>{r.start_frame}</Td>
                <Td>{r.end_frame}</Td>
                <Td>{r.n_frames}</Td>
                <Td>{num(r.duration_min)}</Td>
                <Td>{num(r.sampled_occupancy_min)}</Td>
                <Td>
                  {r.persistent
                    ? <Badge className="bg-cyan-500/15 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/15">persistent</Badge>
                    : <span className="text-muted-foreground">transient</span>}
                </Td>
              </tr>
            ))}
          </ScrollTable>
        </TabsContent>

        {job.sensitivity?.length ? (
          <TabsContent value="sensitivity">
            <ScrollTable testid="sensitivity-table" head={["Contact px", "Condition", "Mean engaged %", "Persistent tracks", "Eligible", "Persistent %", "Persistent eps", "Mean dur (min)"]}>
              {job.sensitivity.map((r, i) => (
                <tr key={i} className="hover:bg-accent/40">
                  <Td className="text-amber-300 font-semibold">{r.contact_px}</Td>
                  <Td><Dot label={r.label} labels={labels} /></Td>
                  <Td className="text-emerald-300">{num(r.mean_engaged_target_percent)}</Td>
                  <Td>{fmtInt(r.persistent_target_tracks)}</Td>
                  <Td>{fmtInt(r.eligible_target_tracks_observed_4plus_frames)}</Td>
                  <Td className="text-cyan-300 font-semibold">{num(r.persistent_target_fraction_percent)}</Td>
                  <Td>{fmtInt(r.persistent_episode_count)}</Td>
                  <Td>{num(r.mean_persistent_duration_min)}</Td>
                </tr>
              ))}
            </ScrollTable>
          </TabsContent>
        ) : null}

        <TabsContent value="config">
          <pre data-testid="config-json" className="rounded-lg border border-border bg-background p-4 text-xs font-mono text-emerald-200/90 overflow-auto max-h-[420px]">
            {JSON.stringify(job.config, null, 2)}
          </pre>
        </TabsContent>
      </Tabs>
    </div>
  );
}
