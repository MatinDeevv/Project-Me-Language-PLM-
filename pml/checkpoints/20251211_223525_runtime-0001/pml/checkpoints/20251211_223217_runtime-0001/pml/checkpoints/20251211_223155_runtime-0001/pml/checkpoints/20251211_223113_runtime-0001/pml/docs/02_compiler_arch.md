# Compiler Architecture

"
            "PML compiler pipeline (target: Python):

"
            "1) Lexer
"
            "2) Parser -> AST
"
            "3) Semantic checks (types, imports, permissions)
"
            "4) Lowering / normalization
"
            "5) Python emitter

"
            "We keep output stable: same PML => same Python.
