import type { ActivityLogEntry } from "../api";

interface Props {
  entries: ActivityLogEntry[];
}

const LEVEL_COLORS: Record<string, string> = {
  info: "text-gray-300",
  warn: "text-amber-400",
  error: "text-red-400",
};

export function HarnessActivityLog({ entries }: Props) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 h-64 overflow-y-auto font-mono text-xs p-3">
      {entries.length === 0 && (
        <p className="text-gray-600 italic">No harness activity recorded yet.</p>
      )}
      {entries.map((entry, i) => {
        const ts = entry.ts
          ? new Date(entry.ts).toLocaleTimeString()
          : "";
        const levelColor = LEVEL_COLORS[entry.level] ?? "text-gray-400";
        return (
          <div key={`${entry.ts}-${i}`} className="leading-relaxed">
            <span className="text-gray-600">[{ts}]</span>{" "}
            <span className="text-blue-400">{entry.tool}</span>{" "}
            <span className={levelColor}>{entry.msg}</span>
          </div>
        );
      })}
    </div>
  );
}
