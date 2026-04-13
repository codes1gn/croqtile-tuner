# Croq-Tune Todo Payloads

Use these templates for framework-specific continuation updates.

## Shared IDs

- continuation id: `continue-croq-tune`
- step id: `round-step`

## Cursor IDE (`TodoWrite`)

Insert/refresh continuation:

```json
{
  "merge": true,
  "todos": [
    {
      "id": "continue-croq-tune",
      "content": "Continue /croq-tune <dsl> <dtype> [shape_key]",
      "status": "in_progress"
    }
  ]
}
```

Post-STORE refresh:

```json
{
  "merge": true,
  "todos": [
    { "id": "round-step", "status": "completed" },
    {
      "id": "continue-croq-tune",
      "content": "Continue /croq-tune <dsl> <dtype> [shape_key]",
      "status": "in_progress"
    }
  ]
}
```

## OpenCode

- If a todo tool exists (for example `todo_write`), use the same ids and field semantics as Cursor.
- If no todo tool exists, persist equivalent state in `.agent/todo.json`.

## Copilot VSCode IDE

Use file-backed state `.agent/todo.json`:

```json
{
  "items": [
    {
      "id": "continue-croq-tune",
      "content": "Continue /croq-tune <dsl> <dtype> [shape_key]",
      "status": "in_progress"
    },
    {
      "id": "round-step",
      "content": "<current round step>",
      "status": "completed"
    }
  ]
}
```
