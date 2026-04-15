import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { IterationLogData } from "../api";

interface Props {
  logs: IterationLogData[];
  baseline: number | null;
}

export function TflopsChart({ logs, baseline }: Props) {
  const data = [...logs]
    .sort((a, b) => a.iteration - b.iteration)
    .map((l) => ({
      iteration: l.iteration,
      tflops: l.tflops,
      decision: l.decision,
    }));

  if (data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
        No iteration data yet.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="iteration"
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "Iteration", position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 11 }}
        />
        <YAxis
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "TFLOPS", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#9ca3af" }}
          itemStyle={{ color: "#60a5fa" }}
        />
        {baseline != null && (
          <ReferenceLine y={baseline} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "baseline", fill: "#f59e0b", fontSize: 10 }} />
        )}
        <Line
          type="monotone"
          dataKey="tflops"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={{ r: 3, fill: "#3b82f6" }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
