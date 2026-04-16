import { useNavigate } from "react-router-dom";
import type { TaskData } from "../api";
import { StatusBadge } from "./StatusBadge";

interface Props {
  tasks: TaskData[];
  activeTaskId: number | null;
}

const MODEL_LABELS: Record<string, string> = {
  "opencode/qwen3.6-plus-free": "Qwen3.6 Plus Free",
  "opencode/minimax-m2.5-free": "Minimax M2.5 Free",
  "opencode/big-pickle": "Big Pickle Free",
};

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
  cursor_ide: "Cursor IDE",
  cursor_cli: "Cursor CLI",
  copilot_ide: "Copilot IDE",
};

export function TaskList({ tasks, activeTaskId }: Props) {
  const navigate = useNavigate();

  if (tasks.length === 0) {
    return (
      <div className="text-center py-16 text-gray-500">
        <p className="text-lg">No tuning tasks yet.</p>
        <p className="text-sm mt-1">Click "+ Add Task" to queue a kernel for tuning.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-400 border-b border-gray-700">
            <th className="py-2 px-3">Status</th>
            <th className="py-2 px-3">Op</th>
            <th className="py-2 px-3">Shape</th>
            <th className="py-2 px-3">Model</th>
            <th className="py-2 px-3">DSL</th>
            <th className="py-2 px-3">Device</th>
            <th className="py-2 px-3">Iter</th>
            <th className="py-2 px-3">Progress vs Baseline</th>
            <th className="py-2 px-3">Best TFLOPS</th>
            <th className="py-2 px-3">Created</th>
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
                  <div className="text-gray-600 text-[10px] uppercase">{task.dtype}</div>
                </td>
                <td className="py-2 px-3 text-gray-300">
                  {task.model ? (
                    <>
                      <div>{MODEL_LABELS[task.model] ?? task.model}</div>
                      <div className="font-mono text-[11px] text-gray-500">{task.model}</div>
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
