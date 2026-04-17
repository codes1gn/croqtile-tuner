import { useNavigate } from "react-router-dom";
import type { TaskData } from "../api";
import { parseDtype, dtypeLabel } from "../dtype";
import { StatusBadge } from "./StatusBadge";

interface Props {
  tasks: TaskData[];
  activeTaskId: number | null;
}

function modelShortLabel(model: string): string {
  const slash = model.indexOf("/");
  return slash > 0 ? model.slice(slash + 1) : model;
}

const DSL_LABELS: Record<string, string> = {
  croqtile: "Croqtile",
  cuda: "CUDA",
  triton: "Triton",
  cute: "CuTe",
  cutile: "CuTile",
  helion: "Helion",
  tilelang: "TileLang",
};

const PLATFORM_LABELS: Record<string, string> = {
  opencode: "OpenCode",
  cursor_cli: "Cursor CLI",
};

function DtypeTag({ dtype }: { dtype: string }) {
  const p = parseDtype(dtype);
  if (p.symmetric) {
    return <div className="text-gray-500 text-[10px] mt-0.5">{dtypeLabel(p.in)}</div>;
  }
  return (
    <div className="flex items-center gap-0.5 mt-0.5">
      <span className="text-[9px] text-cyan-400 font-semibold">{dtypeLabel(p.in)}</span>
      <span className="text-[8px] text-gray-600">→</span>
      <span className="text-[9px] text-amber-400 font-semibold">{dtypeLabel(p.out)}</span>
    </div>
  );
}

export function TaskList({ tasks, activeTaskId }: Props) {
  const navigate = useNavigate();

  if (tasks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 px-6">
        <div className="w-16 h-16 rounded-2xl bg-gray-800 border border-gray-700 flex items-center justify-center mb-5">
          <svg className="w-8 h-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-gray-300 mb-1">No tuning tasks</h3>
        <p className="text-sm text-gray-500 mb-6 text-center max-w-sm">
          Queue a GPU kernel for AI-driven performance tuning. The agent will iteratively optimize your kernel&apos;s TFLOPS.
        </p>
        <div className="text-xs text-gray-600 border border-gray-800 rounded-lg px-4 py-3 bg-gray-900/50 font-mono">
          Click <span className="text-blue-400">+ Add Task</span> above to get started
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-800 bg-gray-900/50">
      <table className="w-full text-sm table-fixed">
        <thead>
          <tr className="text-left text-gray-400 border-b border-gray-700 text-xs uppercase tracking-wider">
            <th className="py-2.5 px-3 w-24">Status</th>
            <th className="py-2.5 px-3 w-24">Op</th>
            <th className="py-2.5 px-3 w-36">Shape</th>
            <th className="py-2.5 px-3 w-36">Model</th>
            <th className="py-2.5 px-3 w-20">DSL</th>
            <th className="py-2.5 px-3 w-28">Device</th>
            <th className="py-2.5 px-3 w-14">Iter</th>
            <th className="py-2.5 px-3 w-40">vs Baseline</th>
            <th className="py-2.5 px-3 w-24">Best</th>
            <th className="py-2.5 px-3 w-36">Created</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((task) => {
            const perfPct = task.baseline_tflops && task.best_tflops
              ? Math.min(Math.round((task.best_tflops / task.baseline_tflops) * 100), 999)
              : null;
            const barWidth = perfPct != null ? Math.min(perfPct, 100) : 0;
            const barColor = perfPct == null ? "bg-gray-600"
              : perfPct >= 90 ? "bg-green-500"
              : perfPct >= 50 ? "bg-yellow-500"
              : "bg-red-500";
            return (
              <tr
                key={task.id}
                onClick={() => navigate(`/tasks/${task.id}`)}
                className={`border-b border-gray-800 hover:bg-gray-800/50 cursor-pointer transition ${task.id === activeTaskId ? "bg-cyan-950/20" : ""}`}
              >
                <td className="py-2 px-3">
                  <StatusBadge status={task.status} />
                </td>
                <td className="py-2 px-3 font-mono text-gray-200 uppercase">{task.op_type ?? "—"}</td>
                <td className="py-2 px-3 font-mono text-gray-300 text-xs whitespace-nowrap">
                  {task.m}×{task.n}×{task.k}
                  <DtypeTag dtype={task.dtype} />
                </td>
                <td className="py-2 px-3 text-gray-300">
                  {task.model ? (
                    <>
                      <div className="truncate max-w-[10rem]" title={task.model}>{modelShortLabel(task.model)}</div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className={`inline-flex items-center rounded-full px-1.5 py-0 text-[9px] font-semibold uppercase tracking-wider ${task.mode === "cursor_cli" ? "bg-violet-500/20 text-violet-300" : "bg-cyan-500/20 text-cyan-300"}`}>
                          {PLATFORM_LABELS[task.mode] ?? task.mode}
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className="text-gray-600">N/A</div>
                  )}
                </td>
                <td className="py-2 px-3 text-gray-300">
                  {DSL_LABELS[task.dsl ?? ""] ?? task.dsl ?? DSL_LABELS[task.mode] ?? task.mode}
                </td>
                <td className="py-2 px-3 text-gray-400 text-xs">
                  {task.device ?? "—"}
                </td>
                <td className="py-2 px-3 font-mono text-gray-400 text-sm">
                  {task.current_iteration}
                </td>
                <td className="py-2 px-3">
                  {task.baseline_tflops != null ? (
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-gray-700 rounded-full h-2">
                        <div
                          className={`${barColor} h-2 rounded-full transition-all`}
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <span className="text-gray-400 text-xs whitespace-nowrap">
                        {perfPct != null ? `${perfPct}%` : "—"}
                      </span>
                    </div>
                  ) : (
                    <span className="text-gray-600 text-xs">no baseline</span>
                  )}
                </td>
                <td className="py-2 px-3 font-mono text-gray-300">
                  {task.best_tflops != null ? `${task.best_tflops.toFixed(1)}` : "--"}
                </td>
                <td className="py-2 px-3 text-gray-500 text-xs whitespace-nowrap">
                  {task.created_at ? new Date(task.created_at).toLocaleString() : "--"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
