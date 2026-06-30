import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, existsSync, rmSync } from "fs";
import { loadEnv } from "../src/env.ts";
import { createSession } from "../src/session.ts";

loadEnv();

const CWD = "/tmp/croqtile-tuner-test/live-workspace";

test("live: write tool creates file via real API", { skip: !process.env.GOOGLE_API_KEY && !process.env.OPENROUTER_API_KEY }, async () => {
  mkdirSync(CWD, { recursive: true });
  rmSync(`${CWD}/test.cu`, { force: true });

  const { session } = await createSession({ cwd: CWD });

  try {
    await session.prompt("Write a file called test.cu containing: __global__ void k() {}");
    assert.ok(existsSync(`${CWD}/test.cu`), "test.cu should exist");
  } finally {
    session.dispose();
  }
});
