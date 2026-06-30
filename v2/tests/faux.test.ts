import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "fs";
import { createFauxSession, cleanDir, fauxAssistantMessage, fauxToolCall } from "./helpers.ts";

const CWD = "/tmp/croqtile-tuner-test/faux";

test("write tool creates file", async () => {
  cleanDir(CWD);

  const session = await createFauxSession({
    cwd: CWD,
    tools: ["write"],
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "hello.cu", content: "__global__ void k() {}\n" })]),
      fauxAssistantMessage("Done."),
    ],
  });

  try {
    await session.prompt("Write hello.cu");
    assert.ok(existsSync(`${CWD}/hello.cu`));
  } finally {
    session.dispose();
  }
});

test("text-only response ends gracefully", async () => {
  cleanDir(CWD);

  const session = await createFauxSession({
    cwd: CWD,
    tools: [],
    responses: [fauxAssistantMessage("Hi.")],
  });

  try {
    await session.prompt("Say hi");
    const last = session.messages[session.messages.length - 1];
    assert.equal(last.role, "assistant");
    assert.ok(last.content.length > 0);
  } finally {
    session.dispose();
  }
});
