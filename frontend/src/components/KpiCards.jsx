import { motion } from "framer-motion";
import { Percent, Target, Link2, Timer } from "lucide-react";
import { fmt, fmtInt, meanBy, sumBy } from "@/lib/format";

export default function KpiCards({ summary }) {
  const cards = [
    {
      id: "persistent-fraction",
      label: "Persistent-target fraction",
      value: fmt(meanBy(summary, "persistent_target_fraction_percent")),
      unit: "%",
      hint: "mean across conditions",
      icon: Percent,
      color: "text-cyan-300",
      ring: "from-cyan-500/20 to-cyan-500/5 border-cyan-500/30",
    },
    {
      id: "eligible-tracks",
      label: "Eligible target tracks",
      value: fmtInt(sumBy(summary, "eligible_target_tracks_observed_4plus_frames")),
      unit: "",
      hint: "observed ≥ stable frames",
      icon: Target,
      color: "text-emerald-300",
      ring: "from-emerald-500/20 to-emerald-500/5 border-emerald-500/30",
    },
    {
      id: "persistent-episodes",
      label: "Persistent episodes",
      value: fmtInt(sumBy(summary, "persistent_episode_count")),
      unit: "",
      hint: "contact ≥ stable frames",
      icon: Link2,
      color: "text-amber-300",
      ring: "from-amber-500/20 to-amber-500/5 border-amber-500/30",
    },
    {
      id: "mean-duration",
      label: "Mean engagement duration",
      value: fmt(meanBy(summary, "mean_persistent_duration_min")),
      unit: "min",
      hint: "across persistent episodes",
      icon: Timer,
      color: "text-sky-300",
      ring: "from-sky-500/20 to-sky-500/5 border-sky-500/30",
    },
  ];

  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 sm:gap-4">
      {cards.map((c, i) => (
        <motion.div
          key={c.id}
          data-testid={`kpi-${c.id}`}
          initial={{ y: 14, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: i * 0.07 }}
          className="relative rounded-xl border border-border bg-card p-4 sm:p-5 overflow-hidden"
        >
          <div className={`absolute -top-6 -right-6 w-20 h-20 rounded-full bg-gradient-to-br ${c.ring} border blur-md opacity-70`} />
          <c.icon className={`w-5 h-5 ${c.color} mb-3`} />
          <div className="font-mono text-2xl sm:text-3xl font-extrabold tracking-tight leading-none">
            {c.value}
            {c.unit && <span className="text-sm text-muted-foreground ml-1">{c.unit}</span>}
          </div>
          <div className="text-xs font-medium mt-2">{c.label}</div>
          <div className="text-[10px] text-muted-foreground font-mono mt-0.5">{c.hint}</div>
        </motion.div>
      ))}
    </div>
  );
}
