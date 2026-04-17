import { useState, useEffect, useCallback } from "react";

interface AgentInfo {
  pid: number;
  session_id: string | null;
  working_dir: string | null;
  kernel: string | null;
  command: string;
}

interface AgentGroups {
  cursor_cli: AgentInfo[];
  opencode: AgentInfo[];
}

const AGENT_TYPE_LABELS: Record<keyof AgentGroups, { label: string; color: string }> = {
  cursor_cli: { label: "Cursor CLI", color: "bg-blue-500/20 text-blue-300" },
  opencode: { label: "OpenCode", color: "bg-green-500/20 text-green-300" },
};

export function AgentMonitorPanel() {
  const [agents, setAgents] = useState<AgentGroups | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const fetchAgents = useCallback(async () => {
    try {
      const res = await fetch("/api/agents");
      if (!res.ok) throw new Error("Failed to fetch agents");
      const data = await res.json();
      setAgents(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
    const timer = setInterval(fetchAgents, 10000); // Refresh every 10s
    return () => clearInterval(timer);
  }, [fetchAgents]);

  const totalAgents = agents
    ? Object.values(agents).reduce((sum, arr) => sum + arr.length, 0)
    : 0;

  if (!loading && !error && totalAgents === 0) {
    return null;
  }

  return (
    <section className="rounded-2xl border border-gray-800 bg-gradient-to-br from-gray-900 via-gray-900 to-slate-950 p-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-emerald-500">Active Agents</div>
          <div className="mt-1 text-2xl font-bold text-gray-100">{totalAgents}</div>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="px-3 py-1.5 text-sm rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 transition"
        >
          {expanded ? "Hide" : "Show Details"}
        </button>
      </div>

      {loading && (
        <p className="mt-3 text-sm text-gray-500">Loading agents...</p>
      )}

      {error && (
        <p className="mt-3 text-sm text-red-400">{error}</p>
      )}

      {expanded && agents && (
        <div className="mt-4 space-y-4">
          {(Object.keys(AGENT_TYPE_LABELS) as Array<keyof AgentGroups>).map((agentType) => {
            const agentList = agents[agentType];
            const { label, color } = AGENT_TYPE_LABELS[agentType];

            return (
              <div key={agentType} className="rounded-xl border border-gray-800 bg-black/20 p-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${color}`}>
                    {label}
                  </span>
                  <span className="text-sm text-gray-400">
                    {agentList.length} running
                  </span>
                </div>

                {agentList.length === 0 ? (
                  <p className="text-xs text-gray-600">No active agents</p>
                ) : (
                  <div className="space-y-2">
                    {agentList.map((agent, idx) => (
                      <div
                        key={`${agentType}-${agent.pid}-${idx}`}
                        className="rounded-lg bg-gray-900/50 p-2 text-xs"
                      >
                        <div className="flex items-center gap-3 text-gray-300">
                          <span className="font-mono text-gray-500">PID: {agent.pid}</span>
                          {agent.session_id && (
                            <span className="text-cyan-400">
                              Session: {agent.session_id.slice(0, 12)}...
                            </span>
                          )}
                        </div>
                        {agent.kernel && (
                          <div className="mt-1 text-emerald-400">
                            Kernel: <span className="font-mono">{agent.kernel}</span>
                          </div>
                        )}
                        {agent.working_dir && (
                          <div className="mt-1 text-gray-500 truncate" title={agent.working_dir}>
                            Dir: {agent.working_dir}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Quick summary when collapsed */}
      {!expanded && agents && totalAgents > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {(Object.keys(AGENT_TYPE_LABELS) as Array<keyof AgentGroups>).map((agentType) => {
            const count = agents[agentType].length;
            if (count === 0) return null;
            const { label, color } = AGENT_TYPE_LABELS[agentType];
            return (
              <span
                key={agentType}
                className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${color}`}
              >
                {label}: {count}
              </span>
            );
          })}
        </div>
      )}
    </section>
  );
}
