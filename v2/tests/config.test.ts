import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "fs";
import { loadConfig } from "../src/config.ts";

const CWD = "/tmp/croqtile-tuner-test/config";

function writeFixture(name: string, content: string): string {
  mkdirSync(CWD, { recursive: true });
  const p = `${CWD}/${name}`;
  writeFileSync(p, content);
  return p;
}

const VALID = `model:
  provider: ollama
  model: qwen2.5-coder:1.5b
task:
  kernel: kernel.co
  build: "bash build.sh"
  profile: "./kernel 2048 2048 2048"
  dsl: croqtile
orchestrator:
  rounds: 5
  store: true
`;

test("loadConfig: parses a valid config", () => {
  const path = writeFixture("valid.yaml", VALID);
  const result = loadConfig(path);
  assert.ok(result.ok);
  if (!result.ok) return;
  assert.equal(result.value.model.provider, "ollama");
  assert.equal(result.value.model.model, "qwen2.5-coder:1.5b");
  assert.equal(result.value.task.kernel, "kernel.co");
  assert.equal(result.value.task.dsl, "croqtile");
  assert.equal(result.value.orchestrator.rounds, 5);
  assert.equal(result.value.orchestrator.store, true);
});

test("loadConfig: applies defaults for omitted sections", () => {
  const path = writeFixture("minimal.yaml", "task:\n  kernel: k.cu\n  build: \"true\"\n  profile: \"echo 'TFLOPS: 1'\"\n");
  const result = loadConfig(path);
  assert.ok(result.ok);
  if (!result.ok) return;
  assert.equal(result.value.model.provider, "anthropic");
  assert.equal(result.value.orchestrator.rounds, 3);
  assert.equal(result.value.orchestrator.store, false);
  assert.equal(result.value.orchestrator.round_timeout_s, 600);
});

test("loadConfig: YAML syntax error is reported", () => {
  const path = writeFixture("broken.yaml", "task:\n  kernel: [unclosed");
  const result = loadConfig(path);
  assert.ok(!result.ok);
  if (result.ok) return;
  assert.match(result.error, /YAML parse error/);
});

test("loadConfig: schema violation (missing task.kernel) is reported", () => {
  const path = writeFixture("bad-schema.yaml", "task:\n  build: \"true\"\n  profile: \"echo 'TFLOPS: 1'\"\n");
  const result = loadConfig(path);
  assert.ok(!result.ok);
  if (result.ok) return;
  assert.match(result.error, /invalid config/);
});

test("loadConfig: missing file is reported", () => {
  const result = loadConfig("/nonexistent/croqtile-tuner.yaml");
  assert.ok(!result.ok);
  if (result.ok) return;
  assert.match(result.error, /cannot read config file/);
});
