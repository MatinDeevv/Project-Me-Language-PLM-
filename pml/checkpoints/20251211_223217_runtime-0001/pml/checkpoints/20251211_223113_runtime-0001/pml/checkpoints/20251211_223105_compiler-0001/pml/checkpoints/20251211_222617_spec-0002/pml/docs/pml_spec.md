# PML (Project Modeling Language) – v0

PML is a high-level, text-based modeling language used by an AI agent to design,
evolve, and maintain large systems over long time horizons.

## Core principles:

- PML is **compiled to Python** – it never replaces Python; it describes structure.
- PML is **agent-friendly** – regular, compact syntax that is cheap in tokens.
- PML is **project-aware** – it encodes modules, layers, and relationships.
- PML is **self-evolving** – the agent is allowed to refine this spec.

## Syntax overview:

PML uses a simple, indentation-based syntax with these core constructs:

- `module` – defines a logical module
- `api` – describes an API surface
- `workflow` – describes long-running processes
- `constraint` – defines invariants and rules

## Initial goals:

1. Describe file/folder layouts and responsibilities.
2. Describe module APIs, invariants, and constraints.
3. Describe long-running workflows and background jobs.
4. Stay simple enough that humans can still read it.

This file is the living spec. The agent should:

- Keep this human-readable.
- Record major versions and breaking changes.
- Add examples in a separate file: `pml_examples.md`.
