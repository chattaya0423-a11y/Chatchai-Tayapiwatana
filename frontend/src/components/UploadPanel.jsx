import { useRef, useState } from "react";
import { UploadCloud, X, Film, Play, Loader2, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";

const DEFAULTS = {
  contact_px: 10,
  stable_frames: 4,
  frame_interval_min: 5,
  max_target_displacement_px: 100,
};

function ParamRow({ id, label, unit, value, min, max, step, onChange }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <Label htmlFor={id} className="text-xs text-muted-foreground">{label}</Label>
        <span className="font-mono text-sm font-semibold text-cyan-300">
          {value}
          <span className="text-muted-foreground text-[10px] ml-1">{unit}</span>
        </span>
      </div>
      <Slider
        id={id}
        data-testid={`param-${id}`}
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0])}
      />
    </div>
  );
}

export default function UploadPanel({ onRun, onDemo, busy, progress, stage }) {
  const [files, setFiles] = useState([]);
  const [labels, setLabels] = useState([]);
  const [params, setParams] = useState(DEFAULTS);
  const [thresholds, setThresholds] = useState([]);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const addFiles = (list) => {
    const incoming = Array.from(list).filter((f) => f.size > 0);
    if (!incoming.length) return;
    const start = files.length;
    setFiles((p) => [...p, ...incoming]);
    setLabels((p) => [...p, ...incoming.map((_, i) => `S${start + i + 1}`)]);
  };

  const removeFile = (idx) => {
    setFiles((p) => p.filter((_, i) => i !== idx));
    setLabels((p) => p.filter((_, i) => i !== idx));
  };

  const setLabel = (idx, val) => setLabels((p) => p.map((l, i) => (i === idx ? val : l)));
  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  const toggleThreshold = (v) =>
    setThresholds((p) => (p.includes(v) ? p.filter((x) => x !== v) : [...p, v]));

  const run = () => {
    if (!files.length) {
      toast.error("Add at least one video first");
      return;
    }
    const sens = [...thresholds].sort((a, b) => a - b);
    onRun(files, {
      name: `${files.length} condition${files.length > 1 ? "s" : ""} · ${params.contact_px}px`,
      labels,
      ...params,
      sensitivity: sens.length > 0,
      sensitivity_thresholds: sens,
    });
  };

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <h2 className="font-display text-lg font-bold tracking-tight mb-1">New analysis</h2>
      <p className="text-xs text-muted-foreground mb-4">Red = NK cells · Green = target cells</p>

      <div
        data-testid="dropzone"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files); }}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
          drag ? "border-cyan-400 bg-cyan-500/10" : "border-border hover:border-emerald-500/50 hover:bg-accent/40"
        }`}
      >
        <input
          ref={inputRef}
          data-testid="file-input"
          type="file"
          accept="video/*,.avi,.mov,.mp4,.mkv"
          multiple
          hidden
          onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }}
        />
        <UploadCloud className="w-7 h-7 mx-auto text-emerald-400 mb-2" />
        <div className="text-sm font-medium">Drop movies or click to browse</div>
        <div className="text-[11px] text-muted-foreground mt-1">.avi .mov .mp4 · multiple conditions supported</div>
      </div>

      {files.length > 0 && (
        <div className="mt-3 space-y-2">
          {files.map((f, i) => (
            <div key={i} className="flex items-center gap-2 rounded-lg border border-border bg-background/50 px-2.5 py-2">
              <Film className="w-4 h-4 text-emerald-400 shrink-0" />
              <Input
                data-testid={`label-input-${i}`}
                value={labels[i]}
                onChange={(e) => setLabel(i, e.target.value)}
                className="h-7 w-20 text-xs font-mono bg-background"
              />
              <span className="text-[11px] text-muted-foreground truncate flex-1" title={f.name}>{f.name}</span>
              <button
                data-testid={`remove-file-${i}`}
                onClick={() => removeFile(i)}
                className="text-muted-foreground hover:text-red-400 shrink-0"
                aria-label="Remove"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 mt-5 mb-3">
        <SlidersHorizontal className="w-4 h-4 text-muted-foreground" />
        <h3 className="font-display text-sm font-semibold">Parameters</h3>
      </div>
      <div className="space-y-4">
        <ParamRow id="contact_px" label="Contact proximity" unit="px" value={params.contact_px} min={1} max={30} step={1} onChange={(v) => setParam("contact_px", v)} />
        <ParamRow id="stable_frames" label="Persistent engagement" unit="frames" value={params.stable_frames} min={2} max={12} step={1} onChange={(v) => setParam("stable_frames", v)} />
        <ParamRow id="frame_interval_min" label="Frame interval" unit="min" value={params.frame_interval_min} min={0.5} max={30} step={0.5} onChange={(v) => setParam("frame_interval_min", v)} />
        <ParamRow id="max_target_displacement_px" label="Max displacement" unit="px" value={params.max_target_displacement_px} min={20} max={200} step={5} onChange={(v) => setParam("max_target_displacement_px", v)} />
      </div>

      <div className="mt-5 rounded-lg border border-border bg-background/50 px-3 py-2.5">
        <div className="flex items-baseline justify-between">
          <div className="text-xs font-medium">Threshold sensitivity</div>
          <span className="text-[10px] text-muted-foreground font-mono">
            {thresholds.length ? `${thresholds.length} selected` : "optional"}
          </span>
        </div>
        <div className="text-[10px] text-muted-foreground font-mono mb-2.5">
          compare chosen proximity thresholds
        </div>
        <div className="grid grid-cols-3 gap-2">
          {[2, 5, 10].map((px) => {
            const active = thresholds.includes(px);
            return (
              <label
                key={px}
                data-testid={`sensitivity-option-${px}`}
                className={`flex items-center justify-center gap-2 rounded-md border px-2 py-2 cursor-pointer transition-colors ${
                  active
                    ? "border-amber-500/50 bg-amber-500/15 text-amber-200"
                    : "border-border bg-background/60 text-muted-foreground hover:bg-accent"
                }`}
              >
                <Checkbox
                  data-testid={`sensitivity-checkbox-${px}`}
                  checked={active}
                  onCheckedChange={() => toggleThreshold(px)}
                  className="border-muted-foreground data-[state=checked]:bg-amber-500 data-[state=checked]:border-amber-500"
                />
                <span className="font-mono text-xs font-semibold">{px} px</span>
              </label>
            );
          })}
        </div>
      </div>

      {busy && (
        <div className="mt-4">
          <Progress value={progress || (stage ? 60 : 0)} className="h-1.5" />
          <div className="text-[11px] text-cyan-300 font-mono mt-1.5 flex items-center gap-1.5">
            <Loader2 className="w-3 h-3 animate-spin" /> {stage}
          </div>
        </div>
      )}

      <Button
        data-testid="run-analysis-button"
        onClick={run}
        disabled={busy}
        className="w-full mt-4 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold"
      >
        {busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
        Run analysis
      </Button>
      <button
        data-testid="panel-demo-button"
        onClick={onDemo}
        disabled={busy}
        className="w-full mt-2 text-xs text-muted-foreground hover:text-cyan-300 transition disabled:opacity-50"
      >
        or load the demo experiment →
      </button>
    </div>
  );
}
