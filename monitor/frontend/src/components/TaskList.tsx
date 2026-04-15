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
            <th className="py-3 px-4">Status</th>
            <th className="py-3 px-4">Shape Key</th>
            <th className="py-3 px-4">Model</th>
            <th className="py-3 px-4">Dtype</th>
            <th className="py-3 px-4">Mode</th>
            <th className="py-3 px-4">Progress</th>
            <th className="py-3 px-4">Best TFLOPS</th>
            <th className="py-3 px-4">Created</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((task) => {
            const pct = task.max_iterations > 0
              ? Math.round((task.current_iteration / task.max_iterations) * 100)
              : 0;
            return (
              <tr
                key={task.id}
                onClick={() => navigate(`/tasks/${task.id}`)}
                className={`border-b border-gray-800 hover:bg-gray-800/50 cursor-pointer transition ${task.id === activeTaskId ? "bg-cyan-950/20" : ""}`}
              >
                <td className="py-3 px-4">
                  <StatusBadge status={task.status} />
                </td>
                <td className="py-3 px-4 font-mono text-gray-200">{task.shape_key}</td>
                <td className="py-3 px-4 text-gray-300">
                  <div>{MODEL_LABELS[task.model ?? ""] ?? "--"}</div>
                  <div className="font-mono text-[11px] text-gray-500">{task.model ?? "system default"}</div>
                </td>
                <td className="py-3 px-4 text-gray-300 uppercase">{task.dtype}</td>
                <td className="py-3 px-4 text-gray-300">
                  {task.mode === "from_current_best" ? "from-best" : "scratch"}
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div className="w-24 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-gray-400 text-xs whitespace-nowrap">
                      {task.current_iteration}/{task.max_iterations}
                    </span>
                  </div>
                </td>
                <td className="py-3 px-4 font-mono text-gray-300">
                  {task.best_tflops != null ? `${task.best_tflops.toFixed(1)}` : "--"}
                </td>
                <td className="py-3 px-4 text-gray-500 text-xs">
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
