import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";
import { CONDITION_COLORS } from "@/lib/api";
import { fmt } from "@/lib/format";

const AXIS = { fontSize: 11, fontFamily: "JetBrains Mono, monospace", fill: "#64748B" };
const TOOLTIP = {
  contentStyle: {
    background: "#0F172A", border: "1px solid #334155", borderRadius: 8, fontSize: 12,
  },
  labelStyle: { color: "#94A3B8", fontFamily: "JetBrains Mono, monospace" },
};

function Panel({ title, subtitle, children, testid }) {
  return (
    <div data-testid={testid} className="rounded-xl border border-border bg-card p-4 sm:p-5">
      <div className="mb-3">
        <h3 className="font-display text-sm font-semibold tracking-tight">{title}</h3>
        {subtitle && <p className="text-[11px] text-muted-foreground mt-0.5">{subtitle}</p>}
      </div>
      <div className="h-[260px]">{children}</div>
    </div>
  );
}

export default function Charts({ job }) {
  const labels = job.summary.map((s) => s.label);
  const colorFor = (i) => CONDITION_COLORS[i % CONDITION_COLORS.length];

  // Engaged % over time, pivoted by time.
  const timeMap = new Map();
  job.per_frame.forEach((r) => {
    const key = r.time_min;
    if (!timeMap.has(key)) timeMap.set(key, { time_min: key });
    timeMap.get(key)[r.label] = r.engaged_target_percent;
  });
  const timeSeries = Array.from(timeMap.values()).sort((a, b) => a.time_min - b.time_min);

  // Headline metrics per condition.
  const metricBars = job.summary.map((s) => ({
    label: s.label,
    "Persistent fraction %": s.persistent_target_fraction_percent ?? 0,
    "Mean engaged %": s.mean_engaged_target_percent ?? 0,
  }));

  // Sensitivity line (persistent fraction vs contact px).
  let sensSeries = null;
  if (job.sensitivity?.length) {
    const sMap = new Map();
    job.sensitivity.forEach((r) => {
      if (!sMap.has(r.contact_px)) sMap.set(r.contact_px, { contact_px: r.contact_px });
      sMap.get(r.contact_px)[r.label] = r.persistent_target_fraction_percent ?? 0;
    });
    sensSeries = Array.from(sMap.values()).sort((a, b) => a.contact_px - b.contact_px);
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <Panel testid="chart-engaged-time" title="Engaged-target % over time" subtitle="Share of target regions within contact range per frame">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeSeries} margin={{ top: 6, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
            <XAxis dataKey="time_min" tick={AXIS} stroke="#334155" unit="m" />
            <YAxis tick={AXIS} stroke="#334155" unit="%" />
            <Tooltip {...TOOLTIP} formatter={(v) => (v == null ? "—" : `${fmt(v)}%`)} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {labels.map((l, i) => (
              <Line key={l} type="monotone" dataKey={l} stroke={colorFor(i)} strokeWidth={2.2} dot={{ r: 2 }} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel testid="chart-metric-bars" title="Headline metrics by condition" subtitle="Persistent-target fraction vs. mean engaged percentage">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={metricBars} margin={{ top: 6, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
            <XAxis dataKey="label" tick={AXIS} stroke="#334155" />
            <YAxis tick={AXIS} stroke="#334155" unit="%" />
            <Tooltip {...TOOLTIP} cursor={{ fill: "#1E293B55" }} formatter={(v) => `${fmt(v)}%`} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="Persistent fraction %" fill="#06B6D4" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Mean engaged %" fill="#10B981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      {sensSeries && (
        <Panel testid="chart-sensitivity" title="Threshold sensitivity" subtitle="Persistent-target fraction across proximity thresholds">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sensSeries} margin={{ top: 6, right: 12, left: -16, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="contact_px" tick={AXIS} stroke="#334155" unit="px" />
              <YAxis tick={AXIS} stroke="#334155" unit="%" />
              <Tooltip {...TOOLTIP} formatter={(v) => `${fmt(v)}%`} labelFormatter={(v) => `${v} px`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {labels.map((l, i) => (
                <Line key={l} type="monotone" dataKey={l} stroke={colorFor(i)} strokeWidth={2.2} dot={{ r: 3 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      )}
    </div>
  );
}
