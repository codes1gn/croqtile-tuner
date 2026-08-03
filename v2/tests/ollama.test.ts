import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import { loadEnv } from "../src/env.ts";
import { createSession } from "../src/session.ts";
import { tune } from "../src/tuner.ts";
import { cleanDir } from "./helpers.ts";

loadEnv();

const OLLAMA_URL = process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434/v1";

async function ollamaAvailable(): Promise<boolean> {
  try {
    const r = await fetch(`${OLLAMA_URL.replace("/v1", "")}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) return false;
    const data = await r.json() as { models: { name: string }[] };
    return data.models?.some(m => m.name.startsWith("qwen3"));
  } catch {
    return false;
  }
}

function hasNvcc(): boolean {
  try {
    execSync("nvcc --version", { stdio: "ignore", env: { ...process.env, PATH: `/usr/local/cuda/bin:${process.env.PATH}` } });
    return true;
  } catch { return false; }
}

const skip = !(await ollamaAvailable());
const skipCuda = skip || !hasNvcc();
if (skip) console.log("Skipping ollama tests: server not reachable or qwen3 model not pulled");

test("ollama session: creates and finds model", { skip }, async () => {
  const session = await createSession({
    cwd: "/tmp",
    provider: "ollama",
    modelId: "qwen3:0.6b",
  });
  session.dispose();
});

test("ollama session: responds to prompt with tool calls", { skip }, async () => {
  const cwd = "/tmp/ollama-test-toolcall";
  cleanDir(cwd);

  const session = await createSession({ cwd, provider: "ollama", modelId: "qwen3:0.6b" });

  await session.prompt("Write a file at hello.txt with content: hello from ollama test");

  const messages = session.messages;
  const hasToolCall = messages.some(m =>
    m.role === "assistant" && m.content.some((c: { type: string }) => c.type === "toolCall")
  );
  const fileWritten = existsSync(`${cwd}/hello.txt`);

  session.dispose();

  assert.ok(hasToolCall || fileWritten, "Expected either a tool call or the file to be written");
  if (fileWritten) {
    const content = readFileSync(`${cwd}/hello.txt`, "utf-8");
    assert.ok(content.includes("hello"), `File content: ${content}`);
  }
});

test("ollama tuner: one round with real kernel", { skip: skipCuda }, async () => {
  const cwd = "/tmp/ollama-test-tuner";
  cleanDir(cwd);

  writeFileSync(`${cwd}/add.cu`, `#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecadd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    size_t sz = N * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sz); cudaMalloc(&d_b, sz); cudaMalloc(&d_c, sz);

    vecadd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 100; i++) vecadd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("Time: %.4f ms\\n", ms / 100.0f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
`);

  const results = await tune({
    task: {
      name: "add",
      cwd,
      kernelPath: "add.cu",
      buildCmd: "PATH=/usr/local/cuda/bin:$PATH nvcc -O3 -arch=sm_90 -o add add.cu",
      profileCmd: "./add",
    },
    rounds: 1,
    provider: "ollama",
    modelId: "qwen3:0.6b",
  });

  assert.equal(results.length, 1);
  assert.ok(results[0].success, `Round failed: ${results[0].errorMessage}`);
});
