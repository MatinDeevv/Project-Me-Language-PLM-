# PML — Project Me Language (Programmable Meta Language)

"
            "PML is a domain-specific language that compiles to Python.

"
            "**Design goals**
"
            "- LLM-friendly: easy to generate, refactor, and validate
"
            "- Deterministic compilation: PML -> Python with predictable output
"
            "- Modular for big systems: packages, modules, interfaces
"
            "- Safe by default: explicit permissions for IO / exec

"
            "**Project folders**
"
            "- `pml/specs/` language specs
"
            "- `pml/compiler/` parser / AST / emitter
"
            "- `pml/runtime/` runtime helpers used by emitted Python
"
            "- `pml/examples/` sample PML programs
