import { motion } from "framer-motion";
import { Microscope, Dna, Radar, Loader2 } from "lucide-react";

const STEPS = [
  { icon: Microscope, label: "Channel segmentation", hint: "Red NK · Green target masks" },
  { icon: Radar, label: "Nearest-centroid tracking", hint: "Links target regions across frames" },
  { icon: Dna, label: "Proximity & persistence", hint: "≤ contact px for ≥ stable frames" },
];

export default function EmptyState({ busy, stage }) {
  return (
    <div className="relative rounded-xl border border-border bg-card overflow-hidden min-h-[560px] flex flex-col">
      <div className="absolute inset-0 grain opacity-60" />
      <div className="relative flex-1 flex flex-col items-center justify-center text-center px-6 py-14">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="relative mb-8"
        >
          <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-emerald-500/15 to-cyan-500/10 border border-emerald-500/25 flex items-center justify-center">
            {busy ? (
              <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
            ) : (
              <Microscope className="w-10 h-10 text-emerald-400" strokeWidth={1.8} />
            )}
          </div>
        </motion.div>

        <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight max-w-md">
          {busy ? stage || "Running analysis…" : "Quantify persistent target occupation"}
        </h2>
        <p className="mt-3 text-sm text-muted-foreground max-w-md leading-relaxed">
          Upload red/green live-cell movies (NK cells vs. target cells) or load the demo experiment.
          The engine reproduces the published 10-pixel, 4-frame persistence pipeline and returns clear
          tables, charts and downloadable results.
        </p>

        <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-2xl">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ y: 12, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.15 + i * 0.1 }}
              className="rounded-lg border border-border bg-background/60 p-4 text-left"
            >
              <s.icon className="w-5 h-5 text-cyan-400 mb-2" />
              <div className="text-sm font-semibold">{s.label}</div>
              <div className="text-xs text-muted-foreground mt-0.5">{s.hint}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
