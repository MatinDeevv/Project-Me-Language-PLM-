#!/usr/bin/env python3
"""Project Me — PML-only autonomous agent.

This script is intentionally a single file, but internally modular.
It maintains a PML workspace under ./pml/ and can run for days:
- Steps queue from pml/steps.json
- Memory log in pml/memory.jsonl
- Lightweight RAG over memory + workspace index
- LM Studio chat/completions client (persistent session, optional streaming)
- Tool layer for safe file ops + safe shell
- Multi-step agent loop with self-critique + repair
- Autosave snapshots + Git commit/push checkpoints
- Thermal-only cooldown monitor (no “high RAM usage” sleeps)

Usage:
  python main.py warmup
  python main.py pml-init
  python main.py pml-step
  python main.py pml-loop --loop-till-stopped

Compat flags:
  python main.py --pml-create --loop-till-stopped
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests

# --------------------------------------------------------------------------------------
# Paths / core config
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PML_DIR = BASE_DIR / "pml"
PML_DOCS_DIR = PML_DIR / "docs"
PML_SPEC_DIR = PML_DIR / "specs"
PML_COMPILER_DIR = PML_DIR / "compiler"
PML_RUNTIME_DIR = PML_DIR / "runtime"
PML_EXAMPLES_DIR = PML_DIR / "examples"
PML_TMP_DIR = PML_DIR / "tmp"

PML_STEPS_PATH = PML_DIR / "steps.json"
PML_MEMORY_PATH = PML_DIR / "memory.jsonl"
PML_INDEX_PATH = PML_DIR / "index.json"
PML_RUN_LOG = PML_DIR / "run.log"
PML_CRASH_LOG = PML_DIR / "crash.log"
PML_CHECKPOINT_DIR = PML_DIR / "checkpoints"
PML_CONFIG_PATH = PML_DIR / "config.json"

DEFAULT_LM_URL = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
DEFAULT_LM_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-coder-30b")
DEFAULT_MAX_TOKENS = int(os.environ.get("PML_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.environ.get("PML_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.environ.get("PML_TOP_P", "0.9"))

DEFAULT_AGENT_CALLS_PER_STEP = int(os.environ.get("PML_AGENT_CALLS", "4"))
DEFAULT_SLEEP_BETWEEN_STEPS = int(os.environ.get("PML_SLEEP_BETWEEN", "3"))

# Thermal-only safety. No “RAM high => sleep” behavior.
DEFAULT_MAX_GPU_TEMP_C = int(os.environ.get("PML_MAX_GPU_TEMP_C", "87"))
DEFAULT_MAX_CPU_TEMP_C = int(os.environ.get("PML_MAX_CPU_TEMP_C", "95"))
DEFAULT_OVERHEAT_COOLDOWN_SEC = int(os.environ.get("PML_OVERHEAT_COOLDOWN_SEC", str(10 * 60)))

# HTTPS support:
# - If your LM Studio is behind a reverse proxy with HTTPS, set LMSTUDIO_URL=https://...
# - Set LMSTUDIO_SSL_VERIFY=0 to disable cert validation if you're using self-signed certs.
DEFAULT_SSL_VERIFY = os.environ.get("LMSTUDIO_SSL_VERIFY", "1") not in ("0", "false", "False")

# Shell safety list (tight by default). Add via env PML_SHELL_ALLOW="git,python,pytest" etc.
_BASE_ALLOWED_SHELL = {
    "python",
    "py",
    "pytest",
    "pip",
    "uv",
    "dir",
    "ls",
    "type",
    "cat",
    "echo",
    "git",
}
_env_allow = os.environ.get("PML_SHELL_ALLOW", "").strip()
if _env_allow:
    for part in _env_allow.split(","):
        p = part.strip().lower()
        if p:
            _BASE_ALLOWED_SHELL.add(p)
ALLOWED_SHELL_PREFIXES = tuple(sorted(_BASE_ALLOWED_SHELL))

# Workspace policy
# Keep all language work under ./pml. Tools are restricted to BASE_DIR by safe_path().
# The agent prompt itself instructs “ONLY edit ./pml”.


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: Path, also_stdout: bool = True) -> None:
        self.log_path = log_path
        self.also_stdout = also_stdout
        self._lock = threading.Lock()

    def _write(self, line: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        tid = threading.get_ident()
        line = f"[{ts}][T{tid}] {msg}"
        with self._lock:
            if self.also_stdout:
                print(line, flush=True)
            try:
                self._write(line)
            except Exception:
                pass

    def exception(self, msg: str) -> None:
        self.log("[EXC] " + msg)
        tb = traceback.format_exc()
        for ln in tb.splitlines():
            self.log("[EXC] " + ln)


LOG = Logger(PML_RUN_LOG)


def crashlog(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        PML_CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PML_CRASH_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Atomic write helpers / durability
# --------------------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


# --------------------------------------------------------------------------------------
# Safe path + tool layer
# --------------------------------------------------------------------------------------


def safe_path(rel: str) -> Path:
    p = (BASE_DIR / rel).resolve()
    if not str(p).startswith(str(BASE_DIR)):
        raise ValueError(f"Path escapes project root: {p}")
    return p


def ensure_pml_layout() -> None:
    for d in [
        PML_DIR,
        PML_DOCS_DIR,
        PML_SPEC_DIR,
        PML_COMPILER_DIR,
        PML_RUNTIME_DIR,
        PML_EXAMPLES_DIR,
        PML_TMP_DIR,
        PML_CHECKPOINT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def tool_list_dir(path: str = ".") -> Dict[str, Any]:
    p = safe_path(path)
    items = []
    for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        try:
            size = child.stat().st_size if child.is_file() else 0
        except Exception:
            size = 0
        items.append({"name": child.name, "is_dir": child.is_dir(), "size": size})
    return {"cwd": str(p), "items": items}


def tool_tree(path: str = "pml", max_nodes: int = 600) -> Dict[str, Any]:
    root = safe_path(path)
    nodes = 0
    lines: List[str] = []

    def walk(p: Path, prefix: str = "") -> None:
        nonlocal nodes
        if nodes >= max_nodes:
            return
        try:
            children = sorted(p.iterdir(), key=lambda x: x.name.lower())
        except Exception:
            return
        for i, child in enumerate(children):
            if nodes >= max_nodes:
                return
            connector = "└── " if i == len(children) - 1 else "├── "
            lines.append(f"{prefix}{connector}{child.name}")
            nodes += 1
            if child.is_dir():
                walk(child, prefix + ("    " if i == len(children) - 1 else "│   "))

    lines.append(root.name + "/")
    walk(root)
    return {"path": str(root), "tree": "\n".join(lines), "nodes": nodes, "max_nodes": max_nodes}


def tool_read_file(path: str, max_chars: int = 12000) -> Dict[str, Any]:
    p = safe_path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))
    data = p.read_text(encoding="utf-8", errors="ignore")
    truncated = len(data) > max_chars
    return {
        "path": str(p),
        "content": data[:max_chars],
        "truncated": truncated,
        "size": p.stat().st_size,
    }


def tool_write_file(path: str, content: str, append: bool = False) -> Dict[str, Any]:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if append and p.exists():
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return {"path": str(p), "append": True, "bytes": len(content)}
    atomic_write_text(p, content)
    return {"path": str(p), "append": False, "bytes": len(content)}


def tool_mkdir(path: str, exist_ok: bool = True) -> Dict[str, Any]:
    p = safe_path(path)
    p.mkdir(parents=True, exist_ok=exist_ok)
    return {"path": str(p), "exists": True}


def tool_move_path(src: str, dst: str) -> Dict[str, Any]:
    s = safe_path(src)
    d = safe_path(dst)
    d.parent.mkdir(parents=True, exist_ok=True)
    s.replace(d)
    return {"src": str(s), "dst": str(d)}


def tool_delete_path(path: str, recursive: bool = False) -> Dict[str, Any]:
    p = safe_path(path)
    if p == BASE_DIR:
        raise ValueError("Refusing to delete project root")
    if p.is_dir():
        if not recursive:
            p.rmdir()
        else:
            shutil.rmtree(p)
    elif p.exists():
        p.unlink()
    return {"path": str(p), "deleted": True}


def tool_grep(path: str, pattern: str, max_hits: int = 50) -> Dict[str, Any]:
    root = safe_path(path)
    rx = re.compile(pattern)
    hits: List[Dict[str, Any]] = []
    for f in root.rglob("*"):
        if f.is_dir():
            continue
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bin", ".exe"}:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in rx.finditer(text):
            if len(hits) >= max_hits:
                return {"path": str(root), "pattern": pattern, "hits": hits, "truncated": True}
            line_no = text[: m.start()].count("\n") + 1
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            snippet = text[start:end].replace("\n", " ")
            hits.append({"file": str(f.relative_to(BASE_DIR)), "line": line_no, "snippet": snippet})
    return {"path": str(root), "pattern": pattern, "hits": hits, "truncated": False}


def tool_hash_file(path: str) -> Dict[str, Any]:
    p = safe_path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "sha256": h.hexdigest(), "size": p.stat().st_size}


def tool_run_shell(cmd: str) -> Dict[str, Any]:
    cmd_strip = cmd.strip()
    if not cmd_strip:
        raise ValueError("Empty cmd")
    first = cmd_strip.split()[0].lower()
    if not any(first.startswith(pfx) for pfx in ALLOWED_SHELL_PREFIXES):
        raise ValueError(f"Command not allowed by safety list: {first}")
    proc = subprocess.run(
        cmd_strip,
        shell=True,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=None,
    )
    return {
        "cmd": cmd_strip,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-12000:],
        "stderr": (proc.stderr or "")[-12000:],
    }


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "list_dir": tool_list_dir,
    "tree": tool_tree,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "mkdir": tool_mkdir,
    "move_path": tool_move_path,
    "delete_path": tool_delete_path,
    "grep": tool_grep,
    "hash_file": tool_hash_file,
    "run_shell": tool_run_shell,
}


# --------------------------------------------------------------------------------------
# Git checkpointing
# --------------------------------------------------------------------------------------


def _run_git(args: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=None,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def git_has_changes() -> bool:
    try:
        rc, out, _ = _run_git(["status", "--porcelain"])
        return rc == 0 and bool(out.strip())
    except Exception:
        return False


def git_autosave(message: str) -> Dict[str, Any]:
    if not (BASE_DIR / ".git").exists():
        return {"ok": False, "reason": "no .git repo"}
    if not git_has_changes():
        return {"ok": False, "reason": "no changes"}

    rc1, out1, err1 = _run_git(["add", "-A"])
    rc2, out2, err2 = _run_git(["commit", "-m", message])
    rc3, out3, err3 = _run_git(["push"])

    ok = (rc1 == 0) and (rc2 == 0 or "nothing to commit" in (out2 + err2).lower()) and (rc3 == 0)
    return {
        "ok": ok,
        "add": {"rc": rc1, "out": out1[-2000:], "err": err1[-2000:]},
        "commit": {"rc": rc2, "out": out2[-2000:], "err": err2[-2000:]},
        "push": {"rc": rc3, "out": out3[-2000:], "err": err3[-2000:]},
    }


# --------------------------------------------------------------------------------------
# Health monitor (thermal-only)
# --------------------------------------------------------------------------------------


class HealthMonitor:
    def __init__(
        self,
        *,
        max_gpu_temp_c: int = DEFAULT_MAX_GPU_TEMP_C,
        max_cpu_temp_c: int = DEFAULT_MAX_CPU_TEMP_C,
        cooldown_seconds: int = DEFAULT_OVERHEAT_COOLDOWN_SEC,
    ) -> None:
        self.max_gpu_temp_c = max_gpu_temp_c
        self.max_cpu_temp_c = max_cpu_temp_c
        self.cooldown_seconds = cooldown_seconds

    def _read_gpu_temp_nvidia_smi(self) -> Optional[int]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=None,
            )
            if proc.returncode != 0:
                return None
            txt = (proc.stdout or "").strip().splitlines()
            if not txt:
                return None
            return int(txt[0].strip())
        except Exception:
            return None

    def _read_cpu_temp_best_effort(self) -> Optional[int]:
        # Cross-platform CPU temp is messy in Python without extra deps.
        # We keep this best-effort. If unknown, we just skip CPU temp gating.
        # On Linux, psutil.sensors_temperatures might work. On Windows, usually not.
        try:
            import psutil  # type: ignore

            temps = getattr(psutil, "sensors_temperatures", None)
            if not temps:
                return None
            d = temps(fahrenheit=False)
            if not d:
                return None
            best: Optional[float] = None
            for _, entries in d.items():
                for e in entries:
                    if e.current is None:
                        continue
                    if best is None or e.current > best:
                        best = float(e.current)
            if best is None:
                return None
            return int(round(best))
        except Exception:
            return None

    def check_and_cool_if_overheated(self) -> None:
        gpu_t = self._read_gpu_temp_nvidia_smi()
        cpu_t = self._read_cpu_temp_best_effort()

        over = False
        reasons: List[str] = []
        if gpu_t is not None and gpu_t >= self.max_gpu_temp_c:
            over = True
            reasons.append(f"gpu_temp={gpu_t}C >= {self.max_gpu_temp_c}C")
        if cpu_t is not None and cpu_t >= self.max_cpu_temp_c:
            over = True
            reasons.append(f"cpu_temp={cpu_t}C >= {self.max_cpu_temp_c}C")

        if over:
            LOG.log(f"[HEALTH] Overheat detected ({', '.join(reasons)}). Cooling for {self.cooldown_seconds}s")
            time.sleep(self.cooldown_seconds)
        else:
            # Optional trace once in a while handled by caller.
            pass


# --------------------------------------------------------------------------------------
# PML docs + workspace index
# --------------------------------------------------------------------------------------


class PMLDocs:
    def __init__(self) -> None:
        ensure_pml_layout()

    def ensure_skeleton(self) -> None:
        ensure_pml_layout()

        def w(rel: str, content: str) -> None:
            p = safe_path(rel)
            if not p.exists():
                atomic_write_text(p, content)

        w(
            "pml/docs/00_pml_overview.md",
            """# PML — Project Me Language (Programmable Meta Language)\n\n"
            "PML is a domain-specific language that compiles to Python.\n\n"
            "**Design goals**\n"
            "- LLM-friendly: easy to generate, refactor, and validate\n"
            "- Deterministic compilation: PML -> Python with predictable output\n"
            "- Modular for big systems: packages, modules, interfaces\n"
            "- Safe by default: explicit permissions for IO / exec\n\n"
            "**Project folders**\n"
            "- `pml/specs/` language specs\n"
            "- `pml/compiler/` parser / AST / emitter\n"
            "- `pml/runtime/` runtime helpers used by emitted Python\n"
            "- `pml/examples/` sample PML programs\n""",
        )

        w(
            "pml/docs/01_syntax.md",
            """# PML Syntax (Draft)\n\n"
            "This file will evolve. Start simple, then formalize.\n\n"
            "## Core primitives\n"
            "- `module`\n"
            "- `type`\n"
            "- `fn`\n"
            "- `import`\n"
            "- `let`\n\n"
            "## Example (placeholder)\n"
            "```pml\n"
            "module hello\n"
            "fn main() -> str {\n"
            "  return \"hello\"\n"
            "}\n"
            "```\n""",
        )

        w(
            "pml/docs/02_compiler_arch.md",
            """# Compiler Architecture\n\n"
            "PML compiler pipeline (target: Python):\n\n"
            "1) Lexer\n"
            "2) Parser -> AST\n"
            "3) Semantic checks (types, imports, permissions)\n"
            "4) Lowering / normalization\n"
            "5) Python emitter\n\n"
            "We keep output stable: same PML => same Python.\n""",
        )

        w(
            "pml/docs/03_runtime.md",
            """# Runtime\n\n"
            "The runtime is a small Python library used by emitted code.\n"
            "Keep it minimal and well-tested.\n""",
        )

        w(
            "pml/specs/core.md",
            """# PML Core Spec (Draft)\n\n"
            "## Version\n"
            "- Spec version: 0.0.1\n\n"
            "## Modules\n"
            "A PML module is the compilation unit.\n\n"
            "## Types\n"
            "Start with: int, float, bool, str, list[T], dict[K,V]\n\n"
            "## Functions\n"
            "Pure-by-default unless explicitly marked with effects.\n\n"
            "## Effects / Permissions\n"
            "- fs.read\n"
            "- fs.write\n"
            "- net.http\n"
            "- proc.exec\n""",
        )

        w(
            "pml/compiler/README.md",
            """# pml/compiler\n\n"
            "This folder will contain the PML compiler.\n"
            "Target modules:\n"
            "- lexer.py\n"
            "- parser.py\n"
            "- ast.py\n"
            "- sema.py\n"
            "- emit_py.py\n"
            "- cli.py\n""",
        )

        w(
            "pml/runtime/README.md",
            """# pml/runtime\n\n"
            "Runtime helpers used by emitted Python.\n""",
        )

        w(
            "pml/examples/hello.pml",
            """module hello\n\nfn main() -> str {\n  return \"hello\"\n}\n""",
        )

    def update_workspace_index(self, *, max_files: int = 500) -> Dict[str, Any]:
        ensure_pml_layout()

        def is_text_file(p: Path) -> bool:
            if p.is_dir():
                return False
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bin", ".exe", ".dll"}:
                return False
            return True

        items: List[Dict[str, Any]] = []
        count = 0
        for f in sorted(PML_DIR.rglob("*"), key=lambda x: str(x).lower()):
            if count >= max_files:
                break
            if not is_text_file(f):
                continue
            rel = str(f.relative_to(BASE_DIR))
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            summary = text[:1200]
            items.append(
                {
                    "path": rel,
                    "sha256": sha,
                    "size": len(text),
                    "head": summary,
                }
            )
            count += 1

        obj = {
            "ts": time.time(),
            "files": items,
            "truncated": count >= max_files,
            "max_files": max_files,
        }
        atomic_write_json(PML_INDEX_PATH, obj)
        return {"ok": True, "files": len(items), "truncated": obj["truncated"]}


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


@dataclass
class PMLStep:
    id: str
    title: str
    goal: str
    phase: str
    mode: str
    status: str = "pending"  # pending|doing|done|failed
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    attempts: int = 0
    last_error: Optional[str] = None
    last_result: Optional[str] = None


class StepStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_pml_layout()

    def load(self) -> List[PMLStep]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        steps: List[PMLStep] = []
        for obj in raw if isinstance(raw, list) else []:
            try:
                steps.append(PMLStep(**obj))
            except Exception:
                continue
        return steps

    def save(self, steps: List[PMLStep]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [dataclasses.asdict(s) for s in steps]
        atomic_write_json(self.path, data)

    def ensure_seed(self) -> List[PMLStep]:
        steps = self.load()
        if steps:
            return steps
        seed = [
            PMLStep(
                id="spec-0001",
                title="Define PML goals and non-goals",
                goal="Write a crisp list of PML goals, non-goals, and constraints. Save under pml/specs/goals.md.",
                phase="spec",
                mode="docs",
            ),
            PMLStep(
                id="spec-0002",
                title="Draft minimal syntax",
                goal="Draft a minimal PML syntax (module/type/fn/import/let/effects) in pml/docs/01_syntax.md, keeping it short but explicit.",
                phase="spec",
                mode="docs",
            ),
            PMLStep(
                id="compiler-0001",
                title="Scaffold compiler modules",
                goal="Create placeholder compiler modules under pml/compiler (lexer.py, parser.py, ast.py, emit_py.py, cli.py).",
                phase="compiler",
                mode="code",
            ),
            PMLStep(
                id="runtime-0001",
                title="Scaffold runtime",
                goal="Create minimal runtime package under pml/runtime (init, fs helpers placeholders, safe exec placeholders).",
                phase="runtime",
                mode="code",
            ),
            PMLStep(
                id="examples-0001",
                title="Add a small example program",
                goal="Create a small example PML program in pml/examples and describe expected Python output.",
                phase="examples",
                mode="docs",
            ),
        ]
        self.save(seed)
        return seed

    def pick_next(self, steps: List[PMLStep]) -> Optional[PMLStep]:
        pending = [s for s in steps if s.status in ("pending", "failed")]
        if not pending:
            return None
        # prefer spec/docs early
        def key(s: PMLStep) -> Tuple[int, float]:
            pr = {"spec": 0, "docs": 1, "compiler": 2, "runtime": 3, "examples": 4}.get(s.phase, 5)
            return pr, s.created_at

        pending.sort(key=key)
        return pending[0]

    def upsert(self, steps: List[PMLStep], step: PMLStep) -> List[PMLStep]:
        out: List[PMLStep] = []
        replaced = False
        for s in steps:
            if s.id == step.id:
                out.append(step)
                replaced = True
            else:
                out.append(s)
        if not replaced:
            out.append(step)
        return out


# --------------------------------------------------------------------------------------
# Memory + RAG
# --------------------------------------------------------------------------------------


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_pml_layout()

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_recent(self, limit: int = 400) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-limit:]

    @staticmethod
    def _tokens(text: str) -> List[str]:
        # cheap tokenization for similarity — good enough for now
        t = re.sub(r"[^a-zA-Z0-9_\s]", " ", (text or "").lower())
        parts = [p for p in t.split() if len(p) >= 4]
        return parts[:800]

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        rows = self.load_recent(limit=600)
        if not rows:
            return []
        q = set(self._tokens(query))
        if not q:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            txt = (r.get("goal", "") + " " + r.get("answer", ""))[:8000]
            s = set(self._tokens(txt))
            if not s:
                continue
            inter = q.intersection(s)
            if not inter:
                continue
            score = len(inter) / max(1, len(q))
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]


def format_rag(hits: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    if not hits:
        return "No relevant memory."
    out: List[str] = []
    for h in hits:
        sid = h.get("step_id", "?")
        title = h.get("title", "")
        goal = (h.get("goal", "") or "")[:300]
        ans = (h.get("answer", "") or "")[:600]
        out.append(f"- {sid}: {title}\n  goal: {goal}\n  result: {ans}")
    txt = "\n\n".join(out)
    return txt[:max_chars]


def format_index_snippets(index_obj: Dict[str, Any], max_files: int = 12) -> str:
    files = index_obj.get("files") or []
    if not isinstance(files, list) or not files:
        return "(no index yet)"
    out: List[str] = []
    for f in files[:max_files]:
        p = f.get("path")
        head = (f.get("head") or "").strip().replace("\n", " ")
        out.append(f"- {p}: {head[:240]}")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# LM Studio client (persistent session + optional streaming)
# --------------------------------------------------------------------------------------


class LMClient:
    def __init__(self, url: str, model: str, *, ssl_verify: bool = DEFAULT_SSL_VERIFY) -> None:
        self.url = url
        self.model = model
        self.ssl_verify = ssl_verify
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stream: bool = False,
    ) -> str:
        payload = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        if not stream:
            LOG.log(f"[LM] POST -> {self.url} | max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
            t0 = time.time()
            resp = self.session.post(self.url, json=payload, timeout=None, verify=self.ssl_verify)
            dt = time.time() - t0
            LOG.log(f"[LM] Response status: {resp.status_code} in {dt:.2f}s")
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

        # streaming
        LOG.log(f"[LM] STREAM POST -> {self.url} | max_tokens={max_tokens}")
        t0 = time.time()
        resp = self.session.post(self.url, json=payload, timeout=None, verify=self.ssl_verify, stream=True)
        dt0 = time.time() - t0
        LOG.log(f"[LM] Stream status: {resp.status_code} (ttfb={dt0:.2f}s)")
        resp.raise_for_status()

        content_parts: List[str] = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    content_parts.append(str(delta))
                    # live output so terminal isn't blank
                    sys.stdout.write(str(delta))
                    sys.stdout.flush()
            except Exception:
                # Some servers emit non-JSON keep-alives
                continue

        LOG.log(f"\n[LM] Stream finished in {time.time() - t0:.2f}s")
        return "".join(content_parts)


# --------------------------------------------------------------------------------------
# JSON extraction / action normalization
# --------------------------------------------------------------------------------------


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Fast path: raw JSON
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # Remove code fences (common)
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"```\s*$", "", s)

    # Try to find a balanced JSON object
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def normalize_action_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "action" in obj:
        return obj
    schema = obj.get("action_schema")
    if isinstance(schema, dict) and "action" in schema:
        LOG.log("[AGENT] Normalizing action_schema wrapper from model.")
        return schema
    return obj


# --------------------------------------------------------------------------------------
# Brainstormer (creates steps when queue is empty)
# --------------------------------------------------------------------------------------


class Brainstormer:
    def __init__(self, lm: LMClient, memory: MemoryStore) -> None:
        self.lm = lm
        self.memory = memory

    def propose_steps(self, count: int = 10) -> List[PMLStep]:
        recent = self.memory.load_recent(limit=60)
        digest: List[str] = []
        for r in recent[-10:]:
            digest.append(
                f"- {r.get('step_id','?')} [{r.get('phase','?')}] {r.get('title','')}: {str(r.get('answer',''))[:120]}"
            )
        hist = "\n".join(digest) if digest else "(no memory yet)"

        system = (
            "You are a ruthless backlog generator for PML (Project Me Language). "
            "Generate concrete, implementable steps that move the language forward. "
            "Prefer small steps with real code/doc outputs, not vague research. "
            "All work MUST stay under ./pml.\n\n"
            "Return ONLY a JSON list of objects: "
            "[{id,title,goal,phase,mode}, ...]. "
            "id must be unique and follow prefix like spec-XXXX, compiler-XXXX, runtime-XXXX, docs-XXXX, examples-XXXX."
        )

        user = (
            f"Recent progress:\n{hist}\n\n"
            f"Generate {count} next steps. Mix spec/docs/compiler/runtime/examples. "
            "Make them non-overlapping. Keep them deterministic: each step must name the exact file(s) to edit/create."
        )

        raw = self.lm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1500,
            temperature=0.3,
            top_p=0.9,
            stream=False,
        )

        # Parse list
        data: Any = None
        try:
            data = json.loads(raw)
        except Exception:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])

        out: List[PMLStep] = []
        if not isinstance(data, list):
            return out

        for obj in data:
            if not isinstance(obj, dict):
                continue
            try:
                out.append(
                    PMLStep(
                        id=str(obj["id"]),
                        title=str(obj["title"]),
                        goal=str(obj["goal"]),
                        phase=str(obj.get("phase", "spec")),
                        mode=str(obj.get("mode", "design")),
                    )
                )
            except Exception:
                continue
        return out


# --------------------------------------------------------------------------------------
# Safe execution layer for tools
# --------------------------------------------------------------------------------------


class ToolRunner:
    def __init__(self) -> None:
        self.tools = TOOLS

    def run(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self.tools.get(tool_name)
        if fn is None:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            result = fn(**tool_args)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# --------------------------------------------------------------------------------------
# PML agent: planning -> execution -> critique -> repair
# --------------------------------------------------------------------------------------


class PMLAgent:
    def __init__(
        self,
        lm: LMClient,
        memory: MemoryStore,
        steps: StepStore,
        docs: PMLDocs,
        tool_runner: ToolRunner,
        health: HealthMonitor,
        *,
        stream: bool = False,
    ) -> None:
        self.lm = lm
        self.memory = memory
        self.steps_store = steps
        self.docs = docs
        self.tools = tool_runner
        self.health = health
        self.stream = stream

    def _system_prompt(self) -> str:
        tools_text = (
            "Available tools (use them; do NOT guess file contents):\n"
            "- list_dir {path?: str='.'}\n"
            "- tree {path?: str='pml', max_nodes?: int}\n"
            "- read_file {path: str, max_chars?: int}\n"
            "- write_file {path: str, content: str, append?: bool}\n"
            "- mkdir {path: str, exist_ok?: bool}\n"
            "- move_path {src: str, dst: str}\n"
            "- delete_path {path: str, recursive?: bool}\n"
            "- grep {path: str, pattern: str, max_hits?: int}\n"
            "- hash_file {path: str}\n"
            "- run_shell {cmd: str} (limited allowlist)\n\n"
            "CRITICAL RULES:\n"
            "- ONLY modify files under ./pml (language work).\n"
            "- Prefer small diffs that compile/parse.\n"
            "- Keep PML deterministic and compiler output stable.\n"
            "- Always finish the step by returning action=final with a short summary and next suggestions.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY ONE JSON object per reply (no markdown, no extra keys).\n"
            "Either:\n"
            "  {\"action\":\"tool\",\"tool_name\":\"read_file\",\"tool_args\":{...},\"reasoning\":\"...\"}\n"
            "or:\n"
            "  {\"action\":\"final\",\"answer\":\"...\",\"reasoning\":\"...\"}\n"
        )

        return (
            "You are Project Me: a senior language designer and compiler engineer. "
            "You are building PML (Project Me Language / Programmable Meta Language), "
            "a DSL that compiles to Python for large, long-lived systems. "
            "You must be pragmatic: start minimal, ship working compiler pieces early, and iterate.\n\n"
            + tools_text
        )

    def _load_index(self) -> Dict[str, Any]:
        if not PML_INDEX_PATH.exists():
            return {}
        try:
            return json.loads(PML_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _prepare_context(self, step: PMLStep) -> List[Dict[str, str]]:
        self.docs.ensure_skeleton()
        self.docs.update_workspace_index(max_files=500)

        rag_hits = self.memory.search(step.goal, k=6)
        rag_txt = format_rag(rag_hits, max_chars=2500)

        tree_obj = tool_tree("pml", max_nodes=700)
        tree_txt = tree_obj.get("tree", "")

        idx = self._load_index()
        idx_snips = format_index_snippets(idx, max_files=12)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "system", "content": "PML workspace tree:\n" + (tree_txt or "(empty)")},
            {"role": "system", "content": "Workspace index snippets:\n" + (idx_snips or "(none)")},
            {"role": "system", "content": "Relevant memory (RAG):\n" + rag_txt},
            {
                "role": "user",
                "content": (
                    f"Execute this PML step now.\n"
                    f"id: {step.id}\n"
                    f"phase: {step.phase}\n"
                    f"mode: {step.mode}\n"
                    f"title: {step.title}\n"
                    f"goal: {step.goal}\n\n"
                    "Start by inspecting needed files/folders with tools. "
                    "Then implement changes. End with action=final summarizing what was done and what's next."
                ),
            },
        ]
        return messages

    def _lm_call(self, messages: List[Dict[str, str]], *, max_tokens: int) -> str:
        return self.lm.chat(
            messages,
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            stream=self.stream,
        )

    def _apply_action_loop(self, step: PMLStep, messages: List[Dict[str, str]], *, max_calls: int) -> Tuple[str, List[Dict[str, Any]]]:
        tool_calls: List[Dict[str, Any]] = []
        last_raw = ""

        for call_i in range(1, max_calls + 1):
            self.health.check_and_cool_if_overheated()

            LOG.log(f"[AGENT] Step {step.id} LM call {call_i}/{max_calls}")
            raw = self._lm_call(messages, max_tokens=DEFAULT_MAX_TOKENS)
            last_raw = raw

            if self.stream:
                # streaming already printed tokens; add newline boundary
                sys.stdout.write("\n")
                sys.stdout.flush()

            obj = extract_first_json_object(raw)
            if not obj:
                LOG.log("[AGENT] Could not parse JSON. Treating raw response as final.")
                return raw.strip(), tool_calls

            obj = normalize_action_obj(obj)
            action = str(obj.get("action", "final")).lower()

            if action == "final":
                ans = str(obj.get("answer", "") or "").strip()
                if not ans:
                    ans = raw.strip()
                return ans, tool_calls

            if action == "tool":
                tool_name = str(obj.get("tool_name", "")).strip()
                tool_args = obj.get("tool_args") or {}
                if not isinstance(tool_args, dict):
                    tool_args = {}

                LOG.log(f"[AGENT] Tool requested: {tool_name} args={tool_args}")
                result = self.tools.run(tool_name, tool_args)
                tool_calls.append({"tool": tool_name, "args": tool_args, "result": result})

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"action": "tool", "tool_name": tool_name, "tool_args": tool_args}),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool result:\n```json\n"
                            + json.dumps(result, indent=2, ensure_ascii=False)
                            + "\n```\n"
                            "Continue the same step. If enough work is done, return action=final."
                        ),
                    }
                )
                continue

            LOG.log(f"[AGENT] Unknown action='{action}', returning raw.")
            return raw.strip(), tool_calls

        LOG.log("[AGENT] Max calls reached, returning last raw.")
        return last_raw.strip(), tool_calls

    def _critique_and_patch(self, step: PMLStep, final_answer: str) -> Optional[str]:
        # A lightweight self-critique pass that ONLY edits docs/specs if needed.
        # This is intentionally conservative to avoid infinite rework.
        system = (
            "You are a strict reviewer for PML changes. "
            "Given the step output and the current workspace, find obvious gaps/bugs and propose minimal fixes. "
            "If fixes are needed, use tools to apply them. If not, return final with 'No changes'. "
            "Output ONE JSON object only."
        )

        tree_txt = tool_tree("pml", max_nodes=450).get("tree", "")
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "system", "content": system},
            {"role": "system", "content": "PML workspace tree:\n" + tree_txt},
            {
                "role": "user",
                "content": (
                    f"Step id: {step.id}\n"
                    f"Title: {step.title}\n"
                    f"Goal: {step.goal}\n\n"
                    f"Step output:\n{final_answer}\n\n"
                    "If you must patch something, do it now via tools. Otherwise return final='No changes'."
                ),
            },
        ]

        ans, _ = self._apply_action_loop(step, messages, max_calls=2)
        if ans.strip().lower() in ("no changes", "no change", "nothing", "none"):
            return None
        return ans

    def run_one_step(self, *, agent_calls: int) -> bool:
        self.docs.ensure_skeleton()
        steps = self.steps_store.ensure_seed()
        step = self.steps_store.pick_next(steps)

        if step is None:
            LOG.log("[PML] No pending steps. Generating more steps...")
            gen = Brainstormer(self.lm, self.memory).propose_steps(count=12)
            if not gen:
                LOG.log("[PML] Brainstormer produced nothing. Idle.")
                return False
            steps.extend(gen)
            self.steps_store.save(steps)
            step = self.steps_store.pick_next(steps)
            if step is None:
                return False

        step.status = "doing"
        step.updated_at = time.time()
        step.attempts += 1
        steps = self.steps_store.upsert(steps, step)
        self.steps_store.save(steps)

        LOG.log(f"[PML] Working step={step.id} phase={step.phase} title={step.title}")

        tool_calls: List[Dict[str, Any]] = []
        final_answer = ""
        ok = True

        try:
            messages = self._prepare_context(step)
            final_answer, tool_calls = self._apply_action_loop(step, messages, max_calls=agent_calls)

            # small self-critique / patch pass
            patch_note = self._critique_and_patch(step, final_answer)
            if patch_note:
                final_answer += "\n\n---\n\nSelf-critique patches applied:\n" + patch_note

            step.status = "done"
            step.last_error = None
            step.last_result = final_answer
            step.updated_at = time.time()

        except KeyboardInterrupt:
            ok = False
            step.status = "failed"
            step.last_error = "Interrupted"
            step.updated_at = time.time()
            LOG.log("[PML] Interrupted.")

        except Exception as e:
            ok = False
            step.status = "failed"
            step.last_error = str(e)
            step.updated_at = time.time()
            LOG.exception(f"[PML] Step crashed: {e}")
            crashlog(f"step={step.id} error={e}\n{traceback.format_exc()}")

        steps = self.steps_store.load()
        steps = self.steps_store.upsert(steps, step)
        self.steps_store.save(steps)

        mem_rec = {
            "ts": time.time(),
            "step_id": step.id,
            "phase": step.phase,
            "mode": step.mode,
            "title": step.title,
            "goal": step.goal,
            "status": step.status,
            "attempts": step.attempts,
            "error": step.last_error,
            "answer": final_answer,
            "tools": tool_calls,
        }
        self.memory.append(mem_rec)

        # Checkpoint snapshot + git
        self._checkpoint_snapshot(step)
        self._git_checkpoint(step)

        return ok

    def _checkpoint_snapshot(self, step: PMLStep) -> None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = PML_CHECKPOINT_DIR / f"{ts}_{step.id}"
        try:
            snap_dir.mkdir(parents=True, exist_ok=True)
            # copy only pml tree snapshot (best-effort)
            dst = snap_dir / "pml"
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(PML_DIR, dst)
            LOG.log(f"[SNAP] Saved checkpoint: {snap_dir}")
        except Exception as e:
            LOG.log(f"[SNAP] Failed: {e}")

    def _git_checkpoint(self, step: PMLStep) -> None:
        msg = f"pml: {step.id} - {step.title}"
        res = git_autosave(msg)
        if res.get("ok"):
            LOG.log("[GIT] autosave commit+push ok")
        else:
            LOG.log(f"[GIT] autosave skipped/failed: {res.get('reason') or res.get('error')}")


# --------------------------------------------------------------------------------------
# CLI commands
# --------------------------------------------------------------------------------------


def cmd_warmup(lm: LMClient) -> None:
    ensure_pml_layout()
    PMLDocs().ensure_skeleton()
    LOG.log("Running warmup call...")
    messages = [
        {"role": "system", "content": "You are a probe. Reply with exactly one word."},
        {"role": "user", "content": "Ready"},
    ]
    out = lm.chat(messages, max_tokens=8, temperature=0.0, top_p=1.0, stream=False)
    LOG.log(f"Warmup output: {out.strip()!r}")


def cmd_pml_init() -> None:
    ensure_pml_layout()
    docs = PMLDocs()
    docs.ensure_skeleton()
    StepStore(PML_STEPS_PATH).ensure_seed()
    docs.update_workspace_index()
    LOG.log("[PML] Initialized pml workspace, docs, and seed steps.")


def cmd_status() -> None:
    ensure_pml_layout()
    steps = StepStore(PML_STEPS_PATH).ensure_seed()
    pending = [s for s in steps if s.status != "done"]
    done = [s for s in steps if s.status == "done"]
    failed = [s for s in steps if s.status == "failed"]

    LOG.log(f"[STATUS] total={len(steps)} done={len(done)} pending={len(pending)} failed={len(failed)}")
    if pending:
        nxt = StepStore(PML_STEPS_PATH).pick_next(steps)
        if nxt:
            LOG.log(f"[STATUS] next={nxt.id} phase={nxt.phase} title={nxt.title}")


def cmd_generate_steps(lm: LMClient, memory: MemoryStore, steps_store: StepStore, count: int) -> None:
    gen = Brainstormer(lm, memory).propose_steps(count=count)
    if not gen:
        LOG.log("[PML] Brainstormer produced nothing.")
        return
    steps = steps_store.ensure_seed()
    steps.extend(gen)
    steps_store.save(steps)
    LOG.log(f"[PML] Added {len(gen)} steps.")


def cmd_pml_step(agent: PMLAgent, agent_calls: int) -> None:
    agent.run_one_step(agent_calls=agent_calls)


def cmd_pml_loop(agent: PMLAgent, *, agent_calls: int, sleep_between: int) -> None:
    LOG.log(
        f"[PML] Starting loop: agent_calls={agent_calls}, sleep_between={sleep_between}s (Ctrl+C to stop)"
    )
    it = 0
    last_health_log = 0.0

    while True:
        it += 1
        LOG.log(f"[PML] === Global iteration {it} ===")
        try:
            agent.health.check_and_cool_if_overheated()
            agent.run_one_step(agent_calls=agent_calls)
        except KeyboardInterrupt:
            LOG.log("[PML] Stopped by user.")
            break
        except Exception as e:
            LOG.exception(f"[PML] Loop crash: {e}")
            crashlog(f"loop crash: {e}\n{traceback.format_exc()}")

        # periodic non-blocking temp log (only if readable)
        now = time.time()
        if now - last_health_log > 120:
            last_health_log = now
            gpu_t = agent.health._read_gpu_temp_nvidia_smi()
            cpu_t = agent.health._read_cpu_temp_best_effort()
            if gpu_t is not None or cpu_t is not None:
                LOG.log(f"[HEALTH] temps: gpu={gpu_t if gpu_t is not None else '?'}C cpu={cpu_t if cpu_t is not None else '?'}C")

        time.sleep(max(0, int(sleep_between)))


# --------------------------------------------------------------------------------------
# Argparse
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project Me — PML-only autonomous agent")

    # Compatibility flags
    p.add_argument("--pml-create", action="store_true", help="Alias for pml-loop")
    p.add_argument("--loop-till-stopped", action="store_true", help="Ignored (kept for compatibility)")

    # Common flags
    p.add_argument("--lm-url", default=DEFAULT_LM_URL)
    p.add_argument("--lm-model", default=DEFAULT_LM_MODEL)
    p.add_argument("--ssl-verify", action="store_true", default=DEFAULT_SSL_VERIFY)
    p.add_argument("--no-ssl-verify", action="store_false", dest="ssl_verify")

    p.add_argument("--stream", action="store_true", help="Stream tokens (less blank terminal)")

    p.add_argument("--agent-calls", type=int, default=DEFAULT_AGENT_CALLS_PER_STEP)
    p.add_argument("--sleep-between", type=int, default=DEFAULT_SLEEP_BETWEEN_STEPS)

    p.add_argument("--max-gpu-temp", type=int, default=DEFAULT_MAX_GPU_TEMP_C)
    p.add_argument("--max-cpu-temp", type=int, default=DEFAULT_MAX_CPU_TEMP_C)
    p.add_argument("--cooldown", type=int, default=DEFAULT_OVERHEAT_COOLDOWN_SEC)

    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("warmup")
    sub.add_parser("pml-init")
    sub.add_parser("status")

    sp = sub.add_parser("pml-step")

    lp = sub.add_parser("pml-loop")

    gp = sub.add_parser("gen-steps")
    gp.add_argument("--count", type=int, default=12)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    ensure_pml_layout()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # compat dispatch
    if args.pml_create and not args.cmd:
        args.cmd = "pml-loop"

    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    LOG.log(f"[LM] Using LM Studio at {args.lm_url} model='{args.lm_model}'")

    lm = LMClient(args.lm_url, args.lm_model, ssl_verify=bool(args.ssl_verify))

    if args.cmd == "warmup":
        cmd_warmup(lm)
        return

    if args.cmd == "pml-init":
        cmd_pml_init()
        return

    if args.cmd == "status":
        cmd_status()
        return

    memory = MemoryStore(PML_MEMORY_PATH)
    steps = StepStore(PML_STEPS_PATH)
    docs = PMLDocs()
    tools = ToolRunner()
    health = HealthMonitor(
        max_gpu_temp_c=int(args.max_gpu_temp),
        max_cpu_temp_c=int(args.max_cpu_temp),
        cooldown_seconds=int(args.cooldown),
    )

    agent = PMLAgent(
        lm,
        memory,
        steps,
        docs,
        tools,
        health,
        stream=bool(args.stream),
    )

    if args.cmd == "gen-steps":
        cmd_generate_steps(lm, memory, steps, int(args.count))
        return

    if args.cmd == "pml-step":
        cmd_pml_step(agent, int(args.agent_calls))
        return

    if args.cmd == "pml-loop":
        cmd_pml_loop(agent, agent_calls=int(args.agent_calls), sleep_between=int(args.sleep_between))
        return

    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.log("Stopped.")
    except Exception as e:
        LOG.exception(f"Fatal crash: {e}")
        crashlog(f"fatal: {e}\n{traceback.format_exc()}")
        raise
