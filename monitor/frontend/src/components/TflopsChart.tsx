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
  const sorted = [...logs].sort((a, b) => a.iteration - b.iteration);

  // Separate baseline (iter 0) from tuning iterations
  const iter0 = sorted.find((l) => l.iteration === 0);
  const tuningData = sorted
    .filter((l) => l.iteration > 0)
    .map((l) => ({
      iteration: l.iteration,
      tflops: l.tflops,
      decision: l.decision,
    }));

  // Rooftop: prefer task.baseline_tflops, fall back to iter 0's tflops
  const rooftop = baseline ?? iter0?.tflops ?? null;

  if (tuningData.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
        No iteration data yet.
      </div>
    );
  }

  const maxTflops = Math.max(
    ...tuningData.map((d) => d.tflops ?? 0),
    rooftop ?? 0,
  );
  const yMax = Math.ceil(maxTflops * 1.15);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={tuningData} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="iteration"
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "Iteration", position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 11 }}
        />
        <YAxis
          stroke="#6b7280"
          domain={[0, yMax]}
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "TFLOPS", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#9ca3af" }}
          formatter={(value: number, _name: string, props: { payload: { decision?: string } }) => {
            const dec = props.payload.decision;
            const color = dec === "KEEP" ? "#34d399" : dec === "DISCARD" ? "#f87171" : "#60a5fa";
            return [<span style={{ color }}>{value.toFixed(1)}{dec ? ` (${dec})` : ""}</span>, "TFLOPS"];
          }}
        />
        {rooftop != null && (
          <ReferenceLine
            y={rooftop}
            stroke="#f59e0b"
            strokeDasharray="8 4"
            strokeWidth={2}
            label={{
              value: `cuBLAS rooftop ${rooftop.toFixed(1)}`,
              fill: "#f59e0b",
              fontSize: 11,
              position: "right",
            }}
          />
        )}
        <Line
          type="monotone"
          dataKey="tflops"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={({ cx, cy, payload }: { cx: number; cy: number; payload: { decision?: string } }) => (
            <circle
              key={`dot-${cx}-${cy}`}
              cx={cx}
              cy={cy}
              r={4}
              fill={payload.decision === "KEEP" ? "#34d399" : payload.decision === "DISCARD" ? "#f87171" : "#3b82f6"}
              stroke="none"
            />
          )}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
