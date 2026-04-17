/**
 * Utilities for parsing and displaying compound dtype strings.
 *
 * GEMM dtype strings encode the accumulator type by concatenating two dtype
 * tokens, e.g. "fp16fp32" means input=fp16, accumulator/output=fp32.
 * Single-token dtypes (e.g. "fp16") mean the same type for input and output.
 */

const DTYPE_TOKENS = [
  "bf16", "fp16", "fp32", "fp64",
  "e4m3", "e5m2",
  "int64", "int32", "int16", "int8",
  "f64", "f32", "f16",
];

const CANONICAL: Record<string, string> = {
  f16: "fp16", f32: "fp32", f64: "fp64",
  fp16: "fp16", fp32: "fp32", fp64: "fp64",
  bf16: "bf16",
  e4m3: "e4m3", e5m2: "e5m2",
  int8: "int8", int16: "int16", int32: "int32", int64: "int64",
};

/** Normalize a dtype token to its canonical form (f16→fp16, f32→fp32). */
export function canonicalDtype(tok: string): string {
  return CANONICAL[tok.toLowerCase()] ?? tok.toLowerCase();
}

export interface ParsedDtype {
  in: string;
  out: string;
  /** True when input and output types are the same. */
  symmetric: boolean;
}

/**
 * Parse a compound dtype string into input and output type tokens.
 *
 * Examples:
 *   "fp16fp32"  → { in: "fp16", out: "fp32", symmetric: false }
 *   "e4m3f32"   → { in: "e4m3", out: "f32",  symmetric: false }
 *   "fp16"      → { in: "fp16", out: "fp16", symmetric: true  }
 */
export function parseDtype(dtype: string): ParsedDtype {
  const lower = dtype.toLowerCase();
  for (const tok of DTYPE_TOKENS) {
    if (lower.startsWith(tok) && lower.length > tok.length) {
      const rest = lower.slice(tok.length);
      if (DTYPE_TOKENS.includes(rest)) {
        return { in: tok, out: rest, symmetric: false };
      }
    }
  }
  // Single token or unrecognised — treat as symmetric
  return { in: lower, out: lower, symmetric: true };
}

/** Short human-readable label for a dtype token (always canonical form). */
export function dtypeLabel(tok: string): string {
  const canonical = canonicalDtype(tok);
  const MAP: Record<string, string> = {
    fp16: "FP16", fp32: "FP32", fp64: "FP64",
    bf16: "BF16",
    e4m3: "E4M3", e5m2: "E5M2",
    int8: "INT8", int16: "INT16", int32: "INT32", int64: "INT64",
  };
  return MAP[canonical] ?? canonical.toUpperCase();
}
