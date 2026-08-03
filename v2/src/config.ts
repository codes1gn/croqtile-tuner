import { readFileSync } from "fs";
import { z } from "zod";
import { parse } from "yaml";
import { errMsg } from "./util.ts";

// Shared defaults — referenced by both the schema and main.ts's CLI fallbacks
// so changing a default means one edit.
export const DEFAULT_ROUNDS = 3;
export const DEFAULT_STORE = false;
export const DEFAULT_ROUND_TIMEOUT_S = 600;

// Config schema — the fields emerged from practice (PLAN: schema emerges,
// never pre-designed). One file controls everything (Iter 6.1).
export const ConfigSchema = z.object({
  model: z.object({
    provider: z.string().default("anthropic"),
    model: z.string().optional(),
    api_key: z.string().optional(),
  }).default({ provider: "anthropic" }),
  task: z.object({
    name: z.string().optional(),
    cwd: z.string().optional(),
    kernel: z.string(),
    build: z.string(),
    profile: z.string(),
    dsl: z.string().optional(),
    gpu: z.string().optional(),
    shape_key: z.string().optional(),
  }),
  orchestrator: z.object({
    rounds: z.number().int().positive().default(DEFAULT_ROUNDS),
    store: z.boolean().default(DEFAULT_STORE),
    round_timeout_s: z.number().positive().default(DEFAULT_ROUND_TIMEOUT_S),
  }).default({ rounds: DEFAULT_ROUNDS, store: DEFAULT_STORE, round_timeout_s: DEFAULT_ROUND_TIMEOUT_S }),
});

export type Config = z.infer<typeof ConfigSchema>;

export type ConfigResult = { ok: true; value: Config } | { ok: false; error: string };

export function loadConfig(path: string): ConfigResult {
  let text: string;
  try {
    text = readFileSync(path, "utf-8");
  } catch {
    return { ok: false, error: `cannot read config file: ${path}` };
  }

  let parsed: unknown;
  try {
    parsed = parse(text);
  } catch (err) {
    return { ok: false, error: `YAML parse error: ${errMsg(err)}` };
  }

  const result = ConfigSchema.safeParse(parsed ?? {});
  if (!result.success) {
    return { ok: false, error: `invalid config: ${z.prettifyError(result.error)}` };
  }
  return { ok: true, value: result.data };
}
