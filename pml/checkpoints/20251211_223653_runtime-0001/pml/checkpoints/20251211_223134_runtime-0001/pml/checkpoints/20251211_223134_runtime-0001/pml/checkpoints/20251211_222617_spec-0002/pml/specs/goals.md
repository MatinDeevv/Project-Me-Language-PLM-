# PML Goals, Non-Goals, and Constraints

## Goals

1. **LLM-friendly syntax** – Easy to generate, refactor, and validate by AI agents
2. **Deterministic compilation** – PML → Python with predictable output
3. **Modular for large systems** – Support packages, modules, interfaces
4. **Safe by default** – Explicit permissions required for IO / exec operations
5. **Project-aware structure** – Encode modules, layers, and relationships
6. **Agent-friendly** – Compact syntax that is cheap in tokens
7. **Self-evolving** – Allow the agent to refine this spec over time
8. **Compile to Python** – Never replace Python; describe structure using it

## Non-Goals

1. **Replace Python** – PML compiles to Python, never replaces it
2. **Be a general-purpose language** – Focused on system design and structuring
3. **Provide full programming features** – Keep syntax minimal and focused
4. **Be a runtime environment** – Runtime is minimal helper library for emitted code
5. **Support all Python features** – Only compile core constructs to Python

## Constraints

1. **Minimal syntax** – Keep it simple, readable, and compact
2. **Predictable output** – Compilation must be deterministic
3. **Human-readable** – Must be readable by humans even as systems grow
4. **Agent-readable** – Syntax should be cheap in tokens for AI agents
5. **Backwards compatibility** – Changes should not break existing code
6. **No runtime dependencies** – Runtime is minimal and self-contained
7. **Focus on long-lived systems** – Designed for large, evolving projects
