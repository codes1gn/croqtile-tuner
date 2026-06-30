import { mkdirSync, existsSync, readFileSync } from "fs";
import { createSession } from "./session.ts";

const cwd = process.env.CROQTILE_WORKSPACE ?? "/tmp/croqtile-tuner/workspace";
mkdirSync(cwd, { recursive: true });

const { session, model } = await createSession({ cwd });

try {
  console.log(`Session ready. Model: ${model.provider}/${model.id}`);
  console.log("Prompting: 'Write hello.cu'...\n");

  session.subscribe((event) => {
    if (event.type === "message_update") {
      const e = event.assistantMessageEvent;
      if (e.type === "text_delta") {
        process.stdout.write(e.delta);
      }
    } else if (event.type === "tool_execution_start") {
      process.stdout.write(`\n[tool: ${event.toolName}]\n`);
    }
  });

  await session.prompt("Write a file called hello.cu with a minimal CUDA hello world kernel that prints from GPU.");
  console.log("\n\n--- Agent finished ---");

  const helloPath = cwd + "/hello.cu";
  if (existsSync(helloPath)) {
    console.log(`\nhello.cu exists (${readFileSync(helloPath).length} bytes)`);
  } else {
    console.error("hello.cu was NOT created!");
    process.exit(1);
  }
} finally {
  session.dispose();
}
