import { useEffect, useState } from "react";
import { api } from "../api";

interface Props {
  availableModels: string[];
  availableVariants: string[];
  defaultModel: string;
  defaultVariant: string;
  onCreated: () => void;
  onCancel: () => void;
  onRefreshModels?: () => Promise<string[]>;
}

function providerGroup(models: string[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const m of models) {
    const slash = m.indexOf("/");
    const provider = slash > 0 ? m.slice(0, slash) : "other";
    (groups[provider] ??= []).push(m);
  }
  return groups;
}

function shortName(model: string): string {
  const slash = model.indexOf("/");
  return slash > 0 ? model.slice(slash + 1) : model;
}

// Common operator types for GPU kernel tuning
const OPERATOR_TYPES = [
  { value: "gemm_sp", label: "Sparse GEMM (2:4)" },
  { value: "gemm", label: "Dense GEMM" },
  { value: "matmul", label: "MatMul" },
  { value: "matmul_bf16", label: "MatMul BF16" },
  { value: "blockscale_gemm", label: "Block-Scale GEMM (FP8)" },
  { value: "bmm", label: "Batched MatMul" },
  { value: "gemv", label: "GEMV" },
  { value: "fused_moe", label: "Fused MoE" },
  { value: "custom", label: "Custom..." },
];

// Input data types
const INPUT_DTYPES = [
  { value: "f16", label: "FP16" },
  { value: "e4m3", label: "FP8 E4M3" },
  { value: "e5m2", label: "FP8 E5M2" },
  { value: "bf16", label: "BF16" },
  { value: "f32", label: "FP32" },
];

// Output/Accumulator data types
const OUTPUT_DTYPES = [
  { value: "f16", label: "FP16" },
  { value: "f32", label: "FP32" },
  { value: "bf16", label: "BF16" },
];

export function AddTaskForm({ availableModels, availableVariants, defaultModel, defaultVariant, onCreated, onCancel, onRefreshModels }: Props) {
  const [opType, setOpType] = useState("gemm_sp");
  const [customOp, setCustomOp] = useState("");
  const [inputDtype, setInputDtype] = useState("e4m3");
  const [outputDtype, setOutputDtype] = useState("f16");
  const [m, setM] = useState("16384");
  const [n, setN] = useState("16384");
  const [k, setK] = useState("16384");
  const [dsl, setDsl] = useState("croqtile");
  const [platform, setPlatform] = useState("opencode");
  const [model, setModel] = useState(
    defaultModel && availableModels.includes(defaultModel) ? defaultModel : availableModels[0] ?? ""
  );
  const [variant, setVariant] = useState(defaultVariant || (availableVariants[0] ?? ""));
  const [requestBudget, setRequestBudget] = useState("1");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [models, setModels] = useState(availableModels);

  const platformModels = models.filter((m) => {
    if (platform === "opencode") {
      return m.startsWith("opencode/") || m.startsWith("github-copilot/");
    }
    if (platform === "cursor_cli") {
      return m.startsWith("cursor/");
    }
    return true;
  });

  const handleRefreshModels = async () => {
    if (!onRefreshModels) return;
    setRefreshing(true);
    try {
      const fresh = await onRefreshModels();
      setModels(fresh);
    } catch {
      setError("Failed to refresh models");
    } finally {
      setRefreshing(false);
    }
  };

  const platformGroups = providerGroup(platformModels);
  const providerOrder = ["github-copilot", "opencode", "cursor"];
  const sortedProviders = [
    ...providerOrder.filter((p) => platformGroups[p]),
    ...Object.keys(platformGroups).filter((p) => !providerOrder.includes(p)),
  ];

  useEffect(() => {
    if (!platformModels.includes(model) && platformModels.length > 0) {
      setModel(platformModels[0]);
    }
  }, [platform, platformModels, model]);

  useEffect(() => {
    setModels(availableModels);
  }, [availableModels]);

  const modelHasBuiltinVariant = model.startsWith("cursor/");
  const effectiveVariants = modelHasBuiltinVariant ? [""] : availableVariants;

  useEffect(() => {
    if (modelHasBuiltinVariant) {
      setVariant("");
    } else if (!effectiveVariants.includes(variant)) {
      setVariant(effectiveVariants[0] ?? "");
    }
  }, [model, modelHasBuiltinVariant, effectiveVariants, variant]);

  const effectiveOp = opType === "custom" ? customOp : opType;
  const dtype = inputDtype === outputDtype ? inputDtype : `${inputDtype}${outputDtype}`;
  const shapeKey = `${effectiveOp}_${dtype}_${m}x${n}x${k}`;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    const mVal = parseInt(m);
    const nVal = parseInt(n);
    const kVal = parseInt(k);

    if (opType === "custom" && !customOp.trim()) {
      return setError("Custom operator name is required");
    }
    if (!model.trim()) {
      return setError("Model is required");
    }
    if (isNaN(mVal) || mVal < 128) return setError("M must be >= 128");
    if (isNaN(nVal) || nVal < 256) return setError("N must be >= 256");
    if (isNaN(kVal) || kVal < 128) return setError("K must be >= 128");

    setSubmitting(true);
    try {
      const budgetVal = parseInt(requestBudget) || 1;
      await api.createTask({ 
        op_type: effectiveOp, 
        dtype, 
        m: mVal, 
        n: nVal, 
        k: kVal, 
        dsl,
        mode: platform, 
        model, 
        variant,
        request_budget: budgetVal,
      });
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create task");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <form
        onSubmit={handleSubmit}
        className="bg-gray-800 rounded-xl p-6 w-full max-w-md shadow-2xl border border-gray-700"
      >
        <h2 className="text-lg font-bold mb-4 text-gray-100">Add Kernel Tuning Task</h2>

        <div className="space-y-3">
          {/* Operator Type */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Operator Type</label>
            <select
              value={opType}
              onChange={(e) => setOpType(e.target.value)}
              className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
            >
              {OPERATOR_TYPES.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </div>
          
          {/* Custom operator name input */}
          {opType === "custom" && (
            <div>
              <label className="block text-sm text-gray-400 mb-1">Custom Operator Name</label>
              <input
                type="text"
                value={customOp}
                onChange={(e) => setCustomOp(e.target.value)}
                placeholder="e.g., my_kernel"
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
                required
              />
            </div>
          )}

          {/* Data Types: Input and Output */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Input Type</label>
              <select
                value={inputDtype}
                onChange={(e) => setInputDtype(e.target.value)}
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                {INPUT_DTYPES.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Output Type</label>
              <select
                value={outputDtype}
                onChange={(e) => setOutputDtype(e.target.value)}
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                {OUTPUT_DTYPES.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
          </div>
          <p className="text-xs text-gray-500">
            Combined dtype: <span className="font-mono text-cyan-400">{dtype}</span>
          </p>

          {/* Shape: M, N, K */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "M", value: m, set: setM, min: 128 },
              { label: "N", value: n, set: setN, min: 256 },
              { label: "K", value: k, set: setK, min: 128 },
            ].map(({ label, value, set, min }) => (
              <div key={label}>
                <label className="block text-sm text-gray-400 mb-1">
                  {label} <span className="text-gray-500">({">"}= {min})</span>
                </label>
                <input
                  type="number"
                  value={value}
                  onChange={(e) => set(e.target.value)}
                  min={min}
                  className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
                  required
                />
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-400 mb-1">DSL</label>
              <select
                value={dsl}
                onChange={(e) => setDsl(e.target.value)}
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                <option value="croqtile">Croqtile</option>
                <option value="cuda">CUDA</option>
                <option value="triton">Triton</option>
                <option value="cute">CuTe</option>
                <option value="cutile">CuTile</option>
                <option value="helion">Helion</option>
                <option value="tilelang">TileLang</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Agent Platform</label>
              <select
                value={platform}
                onChange={(e) => setPlatform(e.target.value)}
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                <option value="opencode">OpenCode (auto-dispatched via opencode CLI)</option>
                <option value="cursor_cli">Cursor CLI (auto-dispatched via cursor-agent CLI)</option>
              </select>
              {platform === "cursor_cli" && (
                <p className="mt-1 text-[10px] text-cyan-400">
                  Uses <code>cursor-agent --print</code> with a Cursor account model. Requires <code>cursor-agent</code> to be installed and authenticated.
                </p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="col-span-2">
              <label className="flex items-center text-sm text-gray-400 mb-1">
                Model
                {onRefreshModels && (
                  <button
                    type="button"
                    onClick={handleRefreshModels}
                    disabled={refreshing}
                    className="ml-2 text-xs text-blue-400 hover:text-blue-300 disabled:opacity-50"
                    title="Refresh model list from CLI"
                  >
                    {refreshing ? "refreshing..." : "refresh"}
                  </button>
                )}
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={platformModels.length === 0}
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                {platformModels.length === 0 && <option value="">No models for this platform</option>}
                {sortedProviders.map((provider) => (
                  <optgroup key={provider} label={provider}>
                    {platformGroups[provider].map((item) => (
                      <option key={item} value={item}>{shortName(item)}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Variant</label>
              <select
                value={variant}
                onChange={(e) => setVariant(e.target.value)}
                disabled={modelHasBuiltinVariant}
                className="w-full bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
              >
                {effectiveVariants.map((v) => (
                  <option key={v} value={v}>{v || "(none)"}</option>
                ))}
              </select>
            </div>
          </div>
          <p className="text-xs text-gray-500 font-mono">{model}{variant ? ` --variant ${variant}` : ""}</p>

          {/* Request Budget */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Request Budget
              <span className="text-gray-500 ml-1">(auto-wake consumes 1 per dispatch)</span>
            </label>
            <input
              type="number"
              value={requestBudget}
              onChange={(e) => setRequestBudget(e.target.value)}
              min={1}
              className="w-32 bg-gray-700 rounded px-3 py-2 text-gray-100 border border-gray-600 focus:border-blue-500 focus:outline-none"
            />
          </div>
          
          {/* Shape Key Preview */}
          <div className="mt-2 p-2 rounded bg-gray-900 border border-gray-700">
            <span className="text-xs text-gray-500">Shape Key: </span>
            <span className="text-sm font-mono text-cyan-400">{shapeKey}</span>
          </div>
        </div>

        {error && (
          <p className="mt-3 text-sm text-red-400 bg-red-900/30 rounded px-3 py-1.5">{error}</p>
        )}

        <div className="flex justify-end gap-3 mt-5">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white font-medium transition disabled:opacity-50"
          >
            {submitting ? "Adding..." : "Add Task"}
          </button>
        </div>
      </form>
    </div>
  );
}
