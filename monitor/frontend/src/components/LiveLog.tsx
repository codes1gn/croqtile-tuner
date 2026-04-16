import type { AgentLogData } from "../api";

interface Props {
  logs: AgentLogData[];
}

export function LiveLog({ logs }: Props) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 h-64 overflow-y-auto font-mono text-xs p-3">
      {logs.length === 0 && (
        <p className="text-gray-600 italic">Waiting for agent output...</p>
      )}
      {logs.map((log) => {
        const ts = log.timestamp
          ? new Date(log.timestamp).toLocaleTimeString()
          : "";
        const color = log.level === "error" ? "text-red-400" : "text-gray-400";
        return (
          <div key={log.id} className={`${color} leading-relaxed`}>
            <span className="text-gray-600">[{ts}]</span> {log.message}
          </div>
        );
      })}
    </div>
  );
}
