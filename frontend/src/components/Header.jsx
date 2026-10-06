import { Activity, Github, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Header({ onDemo, busy }) {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-md">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30">
            <Activity className="w-5 h-5 text-emerald-400" strokeWidth={2.4} />
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-red-500 ring-2 ring-background" />
          </div>
          <div className="min-w-0">
            <h1 className="font-display text-base sm:text-lg font-bold tracking-tight leading-none truncate">
              NK-Track Analytics
            </h1>
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-400/90 mt-1">
              Persistent NK-proximal occupation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden md:inline font-mono text-[10px] px-2 py-1 rounded border border-border text-muted-foreground">
            v2.4 · sci-core
          </span>
          <Button
            data-testid="header-demo-button"
            onClick={onDemo}
            disabled={busy}
            variant="outline"
            className="border-cyan-500/40 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20 hover:text-cyan-200"
          >
            <Sparkles className="w-4 h-4 mr-1.5" />
            Load demo
          </Button>
        </div>
      </div>
    </header>
  );
}
