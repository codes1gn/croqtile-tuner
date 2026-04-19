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

export function isBaselineEntry(l: IterationLogData): boolean {
  const d = (l.decision ?? "").toUpperCase();
  if (d === "BASELINE") return true;
  const k = (l.kernel_path ?? "").toLowerCase();
  // Framework reference kernels
  if (k.startsWith("framework/")) return true;
  // Only iter000 kernels with baseline markers are external baselines
  // Custom kernels like iter001_baseline_1p1c are NOT baselines
  if (k.startsWith("iter000") && (k.includes("cublas") || k.includes("torch") || k.includes("baseline"))) return true;
  const b = (l.bottleneck ?? "").toLowerCase();
  if (b === "baseline" || b === "baseline_profile") return true;
  return false;
}

export function TflopsChart({ logs, baseline }: Props) {
  const tuningIters = [...logs]
    .filter((l) => !isBaselineEntry(l))
    .sort((a, b) => a.iteration - b.iteration);

  let runningBest = 0;
  const keepPoints: { iteration: number; tflops: number; best: number; kernel: string | null }[] = [];
  const discardPoints: { iteration: number; tflops: number; kernel: string | null }[] = [];
  const failPoints: { iteration: number; tflops: number; decision: string; kernel: string | null }[] = [];

  for (const l of tuningIters) {
    const tf = l.tflops ?? 0;
    const dec = (l.decision ?? "").toUpperCase();
    const isFail = dec === "COMPILE_FAIL" || dec === "SEGFAULT" || dec === "HANG" || tf === 0;
    if (isFail) {
      failPoints.push({ iteration: l.iteration, tflops: 0, decision: dec, kernel: l.kernel_path });
    } else if (dec === "KEEP" && tf > runningBest) {
      runningBest = tf;
      keepPoints.push({ iteration: l.iteration, tflops: tf, best: runningBest, kernel: l.kernel_path });
    } else {
      discardPoints.push({ iteration: l.iteration, tflops: tf, kernel: l.kernel_path });
    }
  }

  const bestStaircase = keepPoints.map((p) => ({ iteration: p.iteration, best: p.best }));

  const rooftop = baseline ?? null;

  if (keepPoints.length === 0 && discardPoints.length === 0 && failPoints.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
        No iteration data yet.
      </div>
    );
  }

  const allTflops = [
    ...keepPoints.map((d) => d.tflops),
    ...discardPoints.map((d) => d.tflops),
    rooftop ?? 0,
  ].filter((t) => t > 0);
  const maxTflops = Math.max(...allTflops, 1);
  const minPositive = allTflops.length > 0 ? Math.min(...allTflops) : 0;
  const yMin = Math.max(0, Math.floor(minPositive * 0.8));
  const yMax = Math.ceil(maxTflops * 1.1) || 1;

  const allIters = [
    ...keepPoints.map((d) => d.iteration),
    ...discardPoints.map((d) => d.iteration),
    ...failPoints.map((d) => d.iteration),
  ];
  const xMax = allIters.length ? Math.max(...allIters) : 1;
  const xDomainMax = xMax + Math.max(1, Math.ceil(xMax * 0.05));
  const xTickCount = Math.min(xDomainMax, 20);

  return (
    <ResponsiveContainer width="100%" height={380}>
      <ComposedChart margin={{ top: 10, right: 130, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          type="number"
          dataKey="iteration"
          domain={[0, xDomainMax]}
          stroke="#6b7280"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "Iteration", position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 11 }}
          allowDecimals={false}
          allowDataOverflow
          tickCount={xTickCount}
        />
        <YAxis
          stroke="#6b7280"
          domain={[yMin, yMax]}
          tick={{ fill: "#9ca3af", fontSize: 11 }}
          label={{ value: "TFLOPS", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#9ca3af" }}
          labelFormatter={(label) => `Iteration ${label}`}
          formatter={(value: number, name: string) => {
            if (name === "best") return [`${value.toFixed(3)} TFLOPS`, "Best so far"];
            if (name === "tflops") return [`${value.toFixed(3)} TFLOPS`, "Measured"];
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
              value: `cuBLAS baseline ${rooftop.toFixed(1)}`,
              fill: "#f59e0b",
              fontSize: 11,
              position: "right",
            }}
          />
        )}

        {bestStaircase.length > 0 && (
          <Line
            data={bestStaircase}
            type="stepAfter"
            dataKey="best"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={false}
            isAnimationActive={false}
          />
        )}

        {/* KEEP iterations: green circles */}
        <Scatter
          data={keepPoints}
          dataKey="tflops"
          name="tflops"
          shape={((props: unknown) => {
            const { cx, cy } = props as { cx: number; cy: number };
            return <circle cx={cx} cy={cy} r={5} fill="#34d399" stroke="#1f2937" strokeWidth={1.5} />;
          }) as (props: unknown) => React.JSX.Element}
          isAnimationActive={false}
        />

        {/* DISCARD iterations: dim red circles */}
        <Scatter
          data={discardPoints}
          dataKey="tflops"
          name="tflops"
          shape={((props: unknown) => {
            const { cx, cy } = props as { cx: number; cy: number };
            return <circle cx={cx} cy={cy} r={4} fill="#f87171" stroke="#1f2937" strokeWidth={1.5} />;
          }) as (props: unknown) => React.JSX.Element}
          isAnimationActive={false}
        />

        {/* COMPILE_FAIL / SEGFAULT / HANG: red X marks at bottom */}
        <Scatter
          data={failPoints.map((p) => ({ ...p, tflops: yMin }))}
          dataKey="tflops"
          name="tflops"
          shape={((props: unknown) => {
            const { cx, cy } = props as { cx: number; cy: number };
            const s = 5;
            return (
              <g>
                <line x1={cx - s} y1={cy - s} x2={cx + s} y2={cy + s} stroke="#ef4444" strokeWidth={2.5} strokeLinecap="round" />
                <line x1={cx + s} y1={cy - s} x2={cx - s} y2={cy + s} stroke="#ef4444" strokeWidth={2.5} strokeLinecap="round" />
              </g>
            );
          }) as (props: unknown) => React.JSX.Element}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
