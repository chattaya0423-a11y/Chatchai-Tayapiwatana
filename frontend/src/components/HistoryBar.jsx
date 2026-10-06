import { History, Trash2, FlaskConical } from "lucide-react";
import { prettyDate } from "@/lib/format";

export default function HistoryBar({ history, activeId, onOpen, onDelete }) {
  if (!history?.length) return null;
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <History className="w-4 h-4 text-muted-foreground" />
        <h3 className="font-display text-sm font-semibold tracking-tight">Recent runs</h3>
        <span className="font-mono text-[10px] text-muted-foreground ml-auto">{history.length}</span>
      </div>
      <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
        {history.map((h) => {
          const active = h.id === activeId;
          const conds = Object.keys(h.videos || {}).length;
          return (
            <div
              key={h.id}
              data-testid={`history-item-${h.id}`}
              className={`group flex items-center gap-2 rounded-lg border px-3 py-2 cursor-pointer transition-colors ${
                active
                  ? "border-cyan-500/50 bg-cyan-500/10"
                  : "border-border bg-background/40 hover:bg-accent"
              }`}
              onClick={() => onOpen(h.id)}
            >
              <FlaskConical className={`w-4 h-4 shrink-0 ${active ? "text-cyan-400" : "text-emerald-400/80"}`} />
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium truncate">{h.name}</div>
                <div className="font-mono text-[10px] text-muted-foreground">
                  {conds} cond · {prettyDate(h.created_at)}
                </div>
              </div>
              <button
                data-testid={`history-delete-${h.id}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(h.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-red-400 transition"
                aria-label="Delete run"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
