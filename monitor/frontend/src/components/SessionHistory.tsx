import type { SessionHistoryData } from "../api";

interface Props {
  history: SessionHistoryData | null;
}

const ROLE_STYLES: Record<string, string> = {
  user: "border-cyan-700/60 bg-cyan-950/20",
  assistant: "border-slate-700 bg-slate-900/60",
};

const KIND_LABELS: Record<string, string> = {
  text: "Message",
  reasoning: "Reasoning",
  tool: "Tool",
};

export function SessionHistory({ history }: Props) {

  if (!history?.session_id) {
    return (
      <div className="rounded-lg border border-gray-700 bg-gray-900 p-4 text-sm text-gray-500">
        Agent session has not been captured for this task yet.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-gray-700 bg-gray-900/70 p-3 text-xs text-gray-400">
        <div className="font-semibold text-gray-200">{history.session_title ?? history.session_id}</div>
        <div className="mt-1 font-mono break-all text-gray-500">{history.session_id}</div>
        {history.session_directory && (
          <div className="mt-1 font-mono break-all text-gray-500">{history.session_directory}</div>
        )}
      </div>

      <div className="max-h-[28rem] space-y-3 overflow-y-auto rounded-lg border border-gray-700 bg-gray-950 p-3">
        {history.entries.length === 0 && (
          <p className="text-sm italic text-gray-600">No persisted session entries yet.</p>
        )}
        {history.entries.map((entry) => {
          const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : "";
          const roleStyle = ROLE_STYLES[entry.role] ?? ROLE_STYLES.assistant;
          return (
            <div key={entry.id} className={`rounded-lg border p-3 ${roleStyle}`}>
              <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wide text-gray-500">
                <span className="font-semibold text-gray-300">{entry.role}</span>
                <span>{KIND_LABELS[entry.kind] ?? entry.kind}</span>
                {entry.tool && <span>tool={entry.tool}</span>}
                {entry.status && <span>status={entry.status}</span>}
                {ts && <span className="ml-auto text-gray-600 normal-case tracking-normal">{ts}</span>}
              </div>
              <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-gray-100">
                {entry.text}
              </pre>
            </div>
          );
        })}
      </div>
    </div>
  );
}