import {
  ComposedChart,
  Line,
  Scatter,
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
  // Exclude BASELINE-decision rows (they are the cuBLAS reference, shown only as rooftop)
  const iters = [...logs]
    .filter((l) => l.decision?.toUpperCase() !== "BASELINE")
    .sort((a, b) => a.iteration - b.iteration);

  // Build two datasets:
  // 1. stepData: running-best ascending staircase — only advances when a new best is reached
  // 2. discardMarkers: all DISCARD/failed iterations as scatter markers (not on the line)
  let runningBest = 0;
  const stepData: { iteration: number; best: number }[] = [];
  const discardMarkers: { iteration: number; tflops: number }[] = [];

  for (const l of iters) {
    const tf = l.tflops ?? 0;
    if (tf > 0 && tf > runningBest) {
      runningBest = tf;
      stepData.push({ iteration: l.iteration, best: runningBest });
    } else if (tf > 0) {
      discardMarkers.push({ iteration: l.iteration, tflops: tf });
    }
    // zero/null (compile errors) are silently ignored
  }

  const rooftop = baseline ?? null;

  if (stepData.length === 0 && discardMarkers.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
        No iteration data yet.
      </div>
    );
  }

  const maxTflops = Math.max(
    ...stepData.map((d) => d.best),
    ...discardMarkers.map((d) => d.tflops),
    rooftop ?? 0,
  );
  const yMax = Math.ceil(maxTflops * 1.15) || 1;

  const allXValues = [
    ...stepData.map((d) => d.iteration),
    ...discardMarkers.map((d) => d.iteration),
  ];
  const xMin = allXValues.length ? Math.min(...allXValues) : 1;
  const xMax = allXValues.length ? Math.max(...allXValues) : 1;

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart margin={{ top: 10, right: 120, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          type="number"
          dataKey="iteration"
          domain={[xMin, xMax]}
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "Iteration", position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 11 }}
          allowDataOverflow
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
          formatter={(value: number, name: string) => {
            if (name === "best") return [`${value.toFixed(3)} TFLOPS`, "Best so far"];
            if (name === "tflops") return [`${value.toFixed(3)} TFLOPS`, "Trial (discarded)"];
            return [value, name];
          }}
        />
        {rooftop != null && (
          <ReferenceLine
            y={rooftop}
            stroke="#f59e0b"
            strokeDasharray="8 4"
            strokeWidth={2}
            label={{
              value: `Baseline ${rooftop.toFixed(1)} TFLOPS`,
              fill: "#f59e0b",
              fontSize: 11,
              position: "right",
            }}
          />
        )}

        {/* Ascending staircase: only steps up when a new best is achieved */}
        <Line
          data={stepData}
          type="stepAfter"
          dataKey="best"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={({ cx, cy }: { cx: number; cy: number }) => (
            <circle key={`keep-${cx}-${cy}`} cx={cx} cy={cy} r={5} fill="#34d399" stroke="#1f2937" strokeWidth={1.5} />
          )}
          activeDot={{ r: 7, fill: "#34d399" }}
          isAnimationActive={false}
        />

        {/* Discard markers: shown as red crosses at their actual TFLOPS, not on the line */}
        <Scatter
          data={discardMarkers}
          dataKey="tflops"
          shape={({ cx, cy }: { cx: number; cy: number }) => (
            <g key={`discard-${cx}-${cy}`}>
              <line x1={cx - 4} y1={cy - 4} x2={cx + 4} y2={cy + 4} stroke="#f87171" strokeWidth={1.5} />
              <line x1={cx + 4} y1={cy - 4} x2={cx - 4} y2={cy + 4} stroke="#f87171" strokeWidth={1.5} />
            </g>
          )}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
