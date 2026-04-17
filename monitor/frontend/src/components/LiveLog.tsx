import type { AgentLogData } from "../api";

interface Props {
  logs: AgentLogData[];
}

interface ParsedLine {
  ts: string;
  text: string;
  color: string;
  badge?: string;
  badgeColor?: string;
}

const TYPE_BADGE: Record<string, { label: string; color: string }> = {
  step_start: { label: "step", color: "text-cyan-400" },
  step_finish: { label: "done", color: "text-cyan-300" },
  tool_use: { label: "tool", color: "text-violet-400" },
  tool_result: { label: "result", color: "text-violet-300" },
  text: { label: "text", color: "text-gray-300" },
  message: { label: "msg", color: "text-gray-400" },
};

function parseLine(log: AgentLogData): ParsedLine {
  const ts = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : "";
  const raw = log.message.trim();

  // Try to parse opencode structured JSON log
  try {
    const obj = JSON.parse(raw);
    if (obj && typeof obj === "object") {
      const type = obj.type as string | undefined;
      const badge = type ? (TYPE_BADGE[type] ?? null) : null;

      // Extract the most readable field
      let text: string = "";
      if (type === "tool_use") {
        const tool = obj.part?.tool ?? obj.tool ?? obj.name ?? "?";
        const input = obj.part?.input ?? obj.input ?? {};
        const inputStr = typeof input === "object"
          ? Object.entries(input).slice(0, 2).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ")
          : String(input);
        text = `${tool}(${inputStr.length > 80 ? inputStr.slice(0, 77) + "…" : inputStr})`;
      } else if (type === "tool_result") {
        const content = obj.part?.content ?? obj.content ?? "";
        const s = typeof content === "string" ? content : JSON.stringify(content);
        text = s.length > 120 ? s.slice(0, 117) + "…" : s;
      } else if (type === "text") {
        const s = obj.part?.text ?? obj.text ?? "";
        text = s.length > 160 ? s.slice(0, 157) + "…" : s;
      } else if (type === "step_finish") {
        text = `reason=${obj.part?.reason ?? "?"} snap=${String(obj.part?.snapshot ?? "").slice(0, 8)}`;
      } else if (type === "step_start") {
        text = "";
      } else if (obj.message && typeof obj.message === "string") {
        text = obj.message.length > 160 ? obj.message.slice(0, 157) + "…" : obj.message;
      } else {
        text = raw.length > 160 ? raw.slice(0, 157) + "…" : raw;
      }

      return {
        ts,
        text: text || `[${type}]`,
        color: badge?.color ?? "text-gray-400",
        badge: badge?.label,
        badgeColor: badge?.color,
      };
    }
  } catch {
    // not JSON
  }

  // Plain text / structured service log
  // Detect opencode service logs: "INFO ... service=X ..."
  const serviceMatch = raw.match(/service=(\S+)/);
  const errorLevel = raw.startsWith("ERROR") || raw.startsWith("error");
  return {
    ts,
    text: raw.length > 200 ? raw.slice(0, 197) + "…" : raw,
    color: log.level === "error" ? (errorLevel ? "text-red-400" : "text-gray-500") : "text-gray-400",
    badge: serviceMatch ? serviceMatch[1] : undefined,
    badgeColor: "text-gray-600",
  };
}

export function LiveLog({ logs }: Props) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 h-64 overflow-y-auto font-mono text-xs p-3 space-y-0.5">
      {logs.length === 0 && (
        <p className="text-gray-600 italic">Waiting for agent output...</p>
      )}
      {logs.map((log) => {
        const { ts, text, color, badge, badgeColor } = parseLine(log);
        return (
          <div key={log.id} className="flex gap-1.5 leading-relaxed min-w-0">
            <span className="text-gray-700 shrink-0">[{ts}]</span>
            {badge && (
              <span className={`shrink-0 ${badgeColor ?? "text-gray-600"}`}>[{badge}]</span>
            )}
            <span className={`${color} break-all`}>{text}</span>
          </div>
        );
      })}
    </div>
  );
}
