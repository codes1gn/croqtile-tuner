# Development Style — Project Constraints

## Core Principles

1. **Readability first** — code should be simple, minimal, and core. Remove anything that doesn't earn its place.
2. **Leave TODO for memorization** — use `// TODO:` comments to mark incomplete work, future improvements, or deferred decisions.
3. **Surgery changes** — when editing code, make precise surgical changes. Do not bloat or rewrite surrounding code unless necessary.
4. **Think twice** — before implementing, ask: can this be done in a more compact or more intuitive way? If yes, do that instead.

## TypeScript Style

### Types
- Use `type` over `interface` — prefer algebraic data types, union types, mapped types.
- Only use `interface` when declaration merging is explicitly needed (rare).

### Functions
- Top-level exported functions: use `function` declarations (hoisting, clear intent).
- Callbacks and internal closures: use arrow functions.
- Always type return values explicitly for exported functions.

### Error Handling
- Use **Result type** pattern (explicit, Rust-style):
  ```typescript
  type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };
  ```
- Never throw exceptions for expected failures (compile errors, validation failures).
- Only throw for truly unexpected/unrecoverable errors (programmer bugs).

### File Structure
- Start flat (`src/*.ts`). Only create subdirectories when files exceed ~10.
- One primary export per file when possible.
- Group by feature, not by type (no `types/`, `utils/`, `helpers/` folders).

### Module System
- ESM only (`import`/`export`). No CommonJS.
- File extensions in imports when required by runtime.

### Runtime
- **Bun** as primary runtime and bundler.
- `bun build --compile` for producing standalone binary.
- `bun test` for testing.

### Strictness
- `"strict": true` in tsconfig.
- `"noUncheckedIndexedAccess": true` — array/object indexing returns `T | undefined`.
- No `any`. Use `unknown` + narrowing when type is uncertain.

## Code Quality

- No dead code. Delete it, don't comment it out.
- No premature abstraction. Duplicate 2-3 times before extracting.
- Prefer composition over inheritance.
- Prefer pure functions. Isolate side effects to boundaries (IO, agent calls).
- Variable/function names should be self-documenting. Comments only for non-obvious "why".

## Dependencies

- Minimal dependency principle. Every dependency must justify its existence.
- Core allowed: `@earendil-works/pi-coding-agent`, `zod`, standard Node/Bun APIs.
- Prefer stdlib over npm packages for simple tasks (fs, path, child_process).

## Migration Constraints

This project is migrating from a skill-driven system (SKILL.md + Cursor/Claude Code) to a standalone TS binary. Rules:

- **Never break the existing system** — old skills and tools must keep working during transition.
- **Reuse existing shell scripts** — `.claude/skills/croq-tune/tools/*.sh` are called via bash. Don't rewrite them in TS unless there's a clear reason.
- **Knowledge transfer, not rewrite** — SKILL.md content becomes Pi system prompts. The domain knowledge is the same; only the carrier changes.
- **Coexist with Python monitor** — until TS monitor is ready, both can run. Design interfaces (SSE events, DB schema) to be language-agnostic.
- **Preserve tuning data format** — existing `tuning/` directory structure and log formats must remain readable by the new system.
- **Gradual rollout** — one DSL at a time. Prove on croqtile first, then expand.

## Git

- Commit messages: imperative mood, concise, explain "why" not "what".
- Small, focused commits. One logical change per commit.
