import { test } from "node:test";
import assert from "node:assert/strict";
import { parseTflops, runMeasure } from "../src/measure.ts";

test("parseTflops: DSL contract format 'TFLOPS: 36.12'", () => {
  assert.equal(parseTflops("TFLOPS: 36.12   time_ms: 1.2"), 36.12);
});

test("parseTflops: trailing format '277.65 TFLOPS (67.4% cuBLAS)'", () => {
  assert.equal(parseTflops("277.65 TFLOPS (67.4% cuBLAS)"), 277.65);
});

test("parseTflops: case-insensitive", () => {
  assert.equal(parseTflops("tflops: 1.5"), 1.5);
});

test("parseTflops: first occurrence wins", () => {
  assert.equal(parseTflops("TFLOPS: 2.0 then TFLOPS: 3.0"), 2.0);
});

test("parseTflops: no match returns undefined", () => {
  assert.equal(parseTflops("time: 1.0ms, no tflops here"), undefined);
});

test("runMeasure: parses TFLOPS from command output", async () => {
  const r = await runMeasure("echo 'TFLOPS: 1.5'", "/tmp");
  assert.ok(r.ok);
  assert.equal(r.tflops, 1.5);
});

test("runMeasure: non-zero exit fails with code", async () => {
  const r = await runMeasure("echo 'TFLOPS: 1.5'; exit 3", "/tmp");
  assert.ok(!r.ok);
  assert.match(r.error ?? "", /exit code 3/);
});

test("runMeasure: no TFLOPS in output fails", async () => {
  const r = await runMeasure("echo 'hello'", "/tmp");
  assert.ok(!r.ok);
  assert.match(r.error ?? "", /no TFLOPS/);
});
