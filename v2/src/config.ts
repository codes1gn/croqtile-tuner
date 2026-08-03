import { readFileSync } from "fs";
import { z } from "zod";
import { parse } from "yaml";

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
    rounds: z.number().int().positive().default(3),
    store: z.boolean().default(false),
    round_timeout_s: z.number().positive().default(600),
  }).default({ rounds: 3, store: false, round_timeout_s: 600 }),
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
    return { ok: false, error: `YAML parse error: ${err instanceof Error ? err.message : err}` };
  }

  const result = ConfigSchema.safeParse(parsed ?? {});
  if (!result.success) {
    return { ok: false, error: `invalid config: ${z.prettifyError(result.error)}` };
  }
  return { ok: true, value: result.data };
}
