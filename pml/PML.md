# PML (Project Me Language)

PML is a tiny, practical DSL whose purpose is to **describe large, long-lived software systems** at a higher level than Python, while still compiling into **plain Python**.

## Design goals
- Human-readable enough to review, but structured enough for the agent to transform.
- Compiles into Python packages with tests.
- Encourages explicit modules, interfaces, and contracts.
- Keeps files minimal (prefer a few well-named modules over file explosions).

## Core concepts
- **Unit**: a module or component.
- **Contract**: interface + invariants.
- **Pipeline**: ordered execution phases.
- **Step**: atomic iteration task the agent executes.

## Step JSON format
`pml/steps.json` is the agent’s work queue.
Each item is:
```json
{
  "id": "runtime-0001",
  "phase": "runtime",
  "title": "Scaffold runtime",
  "body": "What to build/change and acceptance criteria",
  "status": "todo"
}
```

## Runtime principle
The agent should:
1. Read current workspace.
2. Execute exactly one step.
3. Save progress immediately.
4. Auto-generate new steps when queue is empty.

