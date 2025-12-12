import os
import sys
import time
import json
import argparse
import threading
import subprocess
import shlex
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
MEMORY_PATH = BASE_DIR / "memory.jsonl"
PML_DIR = BASE_DIR / "pml"
PML_OUT_DIR = BASE_DIR / "pml_out"
PML_DOC_PATH = BASE_DIR / "PML_DOCS.md"
STEPS_PATH = BASE_DIR / "steps.json"
HEALTH_LOG_PATH = BASE_DIR / "health_log.jsonl"
PML_STEPS_LOG_PATH = BASE_DIR / "steps_log.jsonl"
PML_CREATE_LOG_PATH = BASE_DIR / "pml_create_log.jsonl"

LM_URL = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
LM_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-coder-30b")

# Optional HTTPS upgrade (e.g., if you front LM Studio with a reverse proxy)
if os.environ.get("PME_FORCE_HTTPS", "0") == "1" and LM_URL.startswith("http://"):
    LM_URL = "https://" + LM_URL[len("http://") :]

DEFAULT_MAX_TOKENS = int(os.environ.get("PME_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.environ.get("PME_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.environ.get("PME_TOP_P", "0.9"))

# Health thresholds + cooldown (seconds)
HEALTH_CPU_MAX_PCT = float(os.environ.get("PME_CPU_MAX_PCT", "92.0"))
HEALTH_RAM_MAX_PCT = float(os.environ.get("PME_RAM_MAX_PCT", "92.0"))
HEALTH_GPU_MEM_MAX_PCT = float(os.environ.get("PME_GPU_MAX_PCT", "92.0"))
HEALTH_COOLDOWN_SECONDS = int(os.environ.get("PME_COOLDOWN_SEC", "600"))  # 10 minutes by default

# Safety for shell commands (no wild system nonsense)
ALLOWED_SHELL_PREFIXES = (
    "python",
    "py",
    "pytest",
    "pip",
    "uv",
    "git",
)

MAX_TOOL_RESULT_CHARS = 4000
MAX_TOOL_REASON_CHARS = 400
MAX_REPO_TREE_ENTRIES = 300
MAX_REPO_TREE_DEPTH = 4
MAX_DOC_CHARS = 20000

_http_session = requests.Session()


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    tid = threading.get_ident()
    print(f"[{ts}][T{tid}] {msg}", flush=True)


def safe_path(rel: str) -> Path:
    p = (BASE_DIR / rel).resolve()
    if not str(p).startswith(str(BASE_DIR)):
        raise ValueError("Path escapes project root")
    return p


def ensure_pml_dirs() -> None:
    PML_DIR.mkdir(parents=True, exist_ok=True)
    PML_OUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_pml_docs_seeded() -> None:
    """Seed human-facing PML docs on first run."""
    if PML_DOC_PATH.exists():
        return
    seed = (
        "# Project Me Language (PML)\n\n"
        "PML is a domain-specific language used only inside this repo to describe full end-to-end systems.\n"
        "- PML specs live in ./pml/*.pml\n"
        "- PML is compiled into Python modules under ./pml_out\n"
        "- PML is meant to describe services, agents, pipelines, and long-lived systems, not one-off scripts.\n"
        "- This file is the canonical human-facing description. The PML agent should keep this updated over time.\n"
    )
    PML_DOC_PATH.write_text(seed, encoding="utf-8")


# -------------------------
# Lightweight JSONL memory
# -------------------------


def read_memory(limit: int = 200) -> List[Dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with MEMORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-limit:]


def append_memory(prompt: str, answer: str, tools: List[Dict[str, Any]]) -> None:
    rec = {
        "ts": time.time(),
        "prompt": prompt,
        "answer": answer,
        "tools": tools,
    }
    try:
        with MEMORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[MEMORY] write failed: {e}")


def _tokenize_for_match(text: str) -> List[str]:
    return [t for t in text.lower().split() if len(t) > 3]


def search_memory(query: str, k: int = 3) -> List[Dict[str, Any]]:
    rows = read_memory()
    if not rows:
        return []
    q_tokens = set(_tokenize_for_match(query))
    if not q_tokens:
        return []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        txt = (r.get("prompt", "") + " " + r.get("answer", ""))[:4000]
        t_tokens = set(_tokenize_for_match(txt))
        if not t_tokens:
            continue
        inter = q_tokens.intersection(t_tokens)
        if not inter:
            continue
        score = len(inter) / len(q_tokens)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]


def build_rag_context(prompt: str) -> str:
    hits = search_memory(prompt, k=5)
    if not hits:
        return ""
    chunks = []
    for h in hits:
        p = h.get("prompt", "")[:400]
        a = h.get("answer", "")[:800]
        chunks.append(f"Prompt: {p}\nAnswer: {a}")
    return "\n\n".join(chunks)


# -------------------------
# PML file + repo tooling
# -------------------------


def tool_pml_list_specs() -> Dict[str, Any]:
    ensure_pml_dirs()
    specs = []
    for p in sorted(PML_DIR.glob("*.pml")):
        specs.append({"name": p.name, "size": p.stat().st_size})
    return {"pml_dir": str(PML_DIR), "specs": specs}


def tool_pml_read_spec(name: str, max_chars: int = 16000) -> Dict[str, Any]:
    ensure_pml_dirs()
    if not name.endswith(".pml"):
        name = name + ".pml"
    p = safe_path(str(PML_DIR / name))
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read(max_chars)
    truncated = p.stat().st_size > max_chars
    return {"path": str(p), "content": data, "truncated": truncated}


def tool_pml_write_spec(name: str, content: str, append: bool = False) -> Dict[str, Any]:
    ensure_pml_dirs()
    if not name.endswith(".pml"):
        name = name + ".pml"
    p = safe_path(str(PML_DIR / name))
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and p.exists() else "w"
    with p.open(mode, encoding="utf-8") as f:
        f.write(content)
    return {"path": str(p), "bytes": len(content), "append": mode == "a"}


def tool_pml_list_generated() -> Dict[str, Any]:
    ensure_pml_dirs()
    files = []
    for p in sorted(PML_OUT_DIR.rglob("*.py")):
        rel = str(p.relative_to(BASE_DIR))
        files.append({"path": rel, "size": p.stat().st_size})
    return {"pml_out_dir": str(PML_OUT_DIR), "files": files}


def tool_pml_read_generated(path: str, max_chars: int = 16000) -> Dict[str, Any]:
    p = safe_path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read(max_chars)
    truncated = p.stat().st_size > max_chars
    return {"path": str(p), "content": data, "truncated": truncated}


def _build_repo_tree(root: Path, max_depth: int, max_entries: int) -> Dict[str, Any]:
    root = root.resolve()
    remaining = [max_entries]

    def walk(path: Path, depth: int) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        if remaining[0] <= 0:
            return nodes
        try:
            children = sorted(path.iterdir(), key=lambda x: x.name.lower())
        except Exception:
            return nodes
        for child in children:
            if remaining[0] <= 0:
                break
            remaining[0] -= 1
            if child.name.startswith("."):
                continue
            node: Dict[str, Any] = {
                "name": child.name,
                "is_dir": child.is_dir(),
            }
            if child.is_file():
                try:
                    node["size"] = child.stat().st_size
                except Exception:
                    node["size"] = 0
            if child.is_dir() and depth < max_depth:
                node["children"] = walk(child, depth + 1)
            nodes.append(node)
        return nodes

    return {"root": str(root), "tree": walk(root, 0)}


def tool_pml_describe_repo(
    max_depth: int = MAX_REPO_TREE_DEPTH,
    max_entries: int = MAX_REPO_TREE_ENTRIES,
) -> Dict[str, Any]:
    tree = _build_repo_tree(BASE_DIR, max_depth=max_depth, max_entries=max_entries)
    return tree


def tool_pml_scan_python_relations() -> Dict[str, Any]:
    root = BASE_DIR
    files: List[Path] = []
    for p in root.rglob("*.py"):
        try:
            if "__pycache__" in str(p):
                continue
            files.append(p)
        except Exception:
            continue
    files = sorted(files, key=lambda x: str(x.relative_to(BASE_DIR)))
    imports_map: Dict[str, List[str]] = {}
    for f in files:
        rel = str(f.relative_to(BASE_DIR))
        imports: List[str] = []
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line_strip = line.strip()
                    if line_strip.startswith("import ") or line_strip.startswith("from "):
                        imports.append(line_strip[:200])
        except Exception:
            continue
        imports_map[rel] = imports
    return {"root": str(root), "python_files": imports_map}


def tool_pml_index() -> Dict[str, Any]:
    ensure_pml_dirs()
    specs = []
    for p in sorted(PML_DIR.glob("*.pml")):
        base = p.stem
        specs.append({"name": p.name, "base": base, "size": p.stat().st_size})
    generated = []
    for p in sorted(PML_OUT_DIR.rglob("*.py")):
        rel = str(p.relative_to(BASE_DIR))
        generated.append({"path": rel, "size": p.stat().st_size})
    return {"specs": specs, "generated": generated}


def tool_pml_search(query: str, max_hits: int = 20) -> Dict[str, Any]:
    ensure_pml_dirs()
    q = query.lower()
    hits: List[Dict[str, Any]] = []
    for p in sorted(PML_DIR.glob("*.pml")):
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    if q in line.lower():
                        hits.append(
                            {
                                "kind": "spec",
                                "path": str(p.relative_to(BASE_DIR)),
                                "line_no": i,
                                "line": line.strip()[:200],
                            }
                        )
                        if len(hits) >= max_hits:
                            return {"query": query, "hits": hits}
        except Exception:
            continue
    for p in sorted(PML_OUT_DIR.rglob("*.py")):
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    if q in line.lower():
                        hits.append(
                            {
                                "kind": "generated",
                                "path": str(p.relative_to(BASE_DIR)),
                                "line_no": i,
                                "line": line.strip()[:200],
                            }
                        )
                        if len(hits) >= max_hits:
                            return {"query": query, "hits": hits}
        except Exception:
            continue
    return {"query": query, "hits": hits}


def tool_pml_read_docs(max_chars: int = MAX_DOC_CHARS) -> Dict[str, Any]:
    ensure_pml_docs_seeded()
    with PML_DOC_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read(max_chars)
    truncated = PML_DOC_PATH.stat().st_size > max_chars
    return {"path": str(PML_DOC_PATH), "content": data, "truncated": truncated}


def tool_pml_write_docs(content: str, append: bool = True) -> Dict[str, Any]:
    ensure_pml_docs_seeded()
    mode = "a" if append else "w"
    with PML_DOC_PATH.open(mode, encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    return {"path": str(PML_DOC_PATH), "bytes": len(content), "append": append}


# -------------------------
# Shell + git backup tools
# -------------------------


def tool_run_shell(cmd: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a shell command with NO timeout (per your request).
    Only allows a fixed safe prefix set (python/py/pytest/pip/uv/git).
    """
    cmd_strip = cmd.strip()
    if not cmd_strip:
        raise ValueError("Empty cmd")
    first = cmd_strip.split()[0].lower()
    if not any(first.startswith(pfx) for pfx in ALLOWED_SHELL_PREFIXES):
        raise ValueError(f"Command not allowed: {first}")
    proc = subprocess.run(
        cmd_strip,
        shell=True,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    return {
        "cmd": cmd_strip,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-MAX_TOOL_RESULT_CHARS:],
        "stderr": proc.stderr[-MAX_TOOL_RESULT_CHARS:],
    }


def tool_git_checkpoint(message: Optional[str] = None, add_all: bool = True) -> Dict[str, Any]:
    """
    Auto-commit current repo state (for GitHub backup, once you push).
    """
    msg = message or f"auto-checkpoint {time.strftime('%Y-%m-%d %H:%M:%S')}"
    status = run_tool("run_shell", {"cmd": "git status --porcelain"})
    if not status.get("ok"):
        return {"ok": False, "stage": "status", "error": status.get("error")}
    out = (status.get("result") or {}).get("stdout", "")
    if not out.strip():
        return {"ok": True, "skipped": True, "reason": "no changes"}
    if add_all:
        add_res = run_tool("run_shell", {"cmd": "git add -A"})
        if not add_res.get("ok"):
            return {"ok": False, "stage": "add", "error": add_res.get("error")}
    safe_msg = msg.replace('"', "'")
    commit_res = run_tool("run_shell", {"cmd": f'git commit -m "{safe_msg}"'})
    if not commit_res.get("ok"):
        return {
            "ok": False,
            "stage": "commit",
            "error": commit_res.get("error"),
            "result": commit_res.get("result"),
        }
    return {"ok": True, "message": msg, "result": commit_res.get("result")}


def tool_pml_run_tests(pattern: str = "test_*.py") -> Dict[str, Any]:
    """
    Hook for running quick tests on the repo.
    """
    cmd = "pytest -q --maxfail=1"
    res = run_tool("run_shell", {"cmd": cmd})
    return {"ok": res.get("ok", False), "result": res.get("result")}


# -------------------------
# LM call + system health
# -------------------------


def _call_lm(
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    payload = {
        "model": LM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    log(f"[LM] POST -> {LM_URL} | max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
    t0 = time.time()
    # NO timeout on HTTP (per your ask) – if it hangs, that's on LM Studio / network
    resp = _http_session.post(LM_URL, json=payload)
    dt = time.time() - t0
    log(f"[LM] Response status: {resp.status_code} in {dt:.2f}s")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _sample_system_health() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"ts": time.time()}
    try:
        if psutil is not None:
            metrics["cpu_pct"] = float(psutil.cpu_percent(interval=0.0))
            vm = psutil.virtual_memory()
            metrics["ram_pct"] = float(vm.percent)
        else:
            metrics["cpu_pct"] = None
            metrics["ram_pct"] = None
    except Exception as e:
        metrics["cpu_error"] = str(e)

    # GPU via nvidia-smi if available
    try:
        nvidia = shutil.which("nvidia-smi")
        if nvidia:
            proc = subprocess.run(
                [
                    nvidia,
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and proc.stdout:
                line = proc.stdout.strip().splitlines()[0]
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    used = float(parts[0])
                    total = float(parts[1]) or 1.0
                    util = float(parts[2])
                    metrics["gpu_mem_used_mb"] = used
                    metrics["gpu_mem_total_mb"] = total
                    metrics["gpu_mem_pct"] = 100.0 * used / total
                    metrics["gpu_util_pct"] = util
    except Exception as e:
        metrics["gpu_error"] = str(e)

    # Log to health_log.jsonl for long-run observability
    try:
        with HEALTH_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception:
        pass
    return metrics


def _maybe_throttle_for_health() -> None:
    """
    If CPU/RAM/GPU usage is above threshold, sleep HEALTH_COOLDOWN_SECONDS.
    This is your “10 min break if overheated” guard.
    """
    metrics = _sample_system_health()
    reasons = []
    cpu = metrics.get("cpu_pct")
    ram = metrics.get("ram_pct")
    gpu_mem = metrics.get("gpu_mem_pct")
    if isinstance(cpu, (int, float)) and cpu >= HEALTH_CPU_MAX_PCT:
        reasons.append(f"cpu={cpu:.1f}%")
    if isinstance(ram, (int, float)) and ram >= HEALTH_RAM_MAX_PCT:
        reasons.append(f"ram={ram:.1f}%")
    if isinstance(gpu_mem, (int, float)) and gpu_mem >= HEALTH_GPU_MEM_MAX_PCT:
        reasons.append(f"gpu_mem={gpu_mem:.1f}%")
    if reasons and HEALTH_COOLDOWN_SECONDS > 0:
        log(f"[HEALTH] High load detected ({', '.join(reasons)}). Cooling for {HEALTH_COOLDOWN_SECONDS}s")
        time.sleep(HEALTH_COOLDOWN_SECONDS)


# -------------------------
# PML compiler + runner
# -------------------------


def _call_pml_compiler(pml_name: str, pml_source: str) -> Dict[str, Any]:
    ensure_pml_dirs()
    repo_overview = tool_pml_describe_repo()
    py_relations = tool_pml_scan_python_relations()
    pml_specs = tool_pml_list_specs()
    ctx = {
        "repo_overview": repo_overview,
        "python_relations": py_relations,
        "pml_specs": pml_specs,
    }
    ctx_str = json.dumps(ctx)[:8000]

    system_msg = (
        "You are the compiler for Project Me Language (PML), a DSL that describes end-to-end systems.\n"
        "You see a JSON snapshot of the repo and existing PML specs so you can generate Python modules into a coherent folder structure.\n"
        "Given a PML spec, output a SINGLE JSON OBJECT (no markdown) with this shape:\n"
        "{\n"
        '  "files": {\n'
        '    "relative/path/module1.py": "<full python code>",\n'
        '    "relative/path/module2.py": "<full python code>"\n'
        "  },\n"
        '  "summary": "short description"\n'
        "}\n"
        "- Target language is Python 3.\n"
        "- Use the existing repo layout when reasonable.\n"
        "- Keep modules cohesive and imported correctly.\n"
        "- No fences, no extra keys, only that JSON object.\n"
    )

    user_msg = (
        f"Repo + PML context (JSON, truncated):\n{ctx_str}\n\n"
        f"PML spec name: {pml_name}\n\n"
        "PML spec content:\n"
        "-----------------\n"
        f"{pml_source}\n"
        "-----------------\n\n"
        "Compile this PML spec into Python modules per the rules."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    content = _call_lm(
        messages,
        max_tokens=int(os.environ.get("PME_PML_COMPILE_MAX_TOKENS", "4096")),
        temperature=0.0,
        top_p=1.0,
    )
    obj = extract_json_block(content)
    if not obj or not isinstance(obj, dict):
        raise ValueError("PML compiler returned invalid JSON")
    files = obj.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("PML compiler JSON missing 'files'")
    summary = str(obj.get("summary", "")).strip()
    written = []
    for rel, code in files.items():
        rel_str = str(rel).lstrip("/\\")
        out_path = safe_path(str(PML_OUT_DIR / rel_str))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(str(code))
        written.append(str(out_path.relative_to(BASE_DIR)))
    return {"written_files": written, "summary": summary}


def tool_pml_compile(name: str) -> Dict[str, Any]:
    ensure_pml_dirs()
    if not name.endswith(".pml"):
        name = name + ".pml"
    p = safe_path(str(PML_DIR / name))
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    log(f"[PML] Compiling {p.name} -> {PML_OUT_DIR}")
    result = _call_pml_compiler(p.name, src)
    return {"pml_name": p.name, **result}


def tool_pml_run(
    name: str,
    entry: str = "main.py",
    python_cmd: str = "python",
) -> Dict[str, Any]:
    ensure_pml_dirs()
    compile_res = tool_pml_compile(name)
    entry_path = safe_path(str(PML_OUT_DIR / entry))
    if not entry_path.exists():
        raise FileNotFoundError(f"Entry file not found after compile: {entry_path}")
    cmd = f"{python_cmd} {entry_path}"
    shell_res = tool_run_shell(cmd)
    return {
        "compiled": compile_res,
        "run": shell_res,
    }


# -------------------------
# Tool registry
# -------------------------


TOOLS: Dict[str, Any] = {
    "pml_list_specs": tool_pml_list_specs,
    "pml_read_spec": tool_pml_read_spec,
    "pml_write_spec": tool_pml_write_spec,
    "pml_list_generated": tool_pml_list_generated,
    "pml_read_generated": tool_pml_read_generated,
    "pml_describe_repo": tool_pml_describe_repo,
    "pml_scan_python_relations": tool_pml_scan_python_relations,
    "pml_index": tool_pml_index,
    "pml_search": tool_pml_search,
    "pml_read_docs": tool_pml_read_docs,
    "pml_write_docs": tool_pml_write_docs,
    "pml_compile": tool_pml_compile,
    "pml_run": tool_pml_run,
    "pml_run_tests": tool_pml_run_tests,
    "run_shell": tool_run_shell,
    "git_checkpoint": tool_git_checkpoint,
}


# -------------------------
# Agent system prompt
# -------------------------


def build_system_prompt(mode: str = "auto") -> str:
    if mode == "design":
        hint = "You are biased toward designing and evolving the PML language itself and authoring .pml specs."
    elif mode == "compiler":
        hint = "You are biased toward compiling PML specs into high-quality Python modules and improving the compilation patterns."
    elif mode == "evolve":
        hint = "You are biased toward iteratively improving existing PML specs and their compiled Python outputs."
    elif mode == "analyze":
        hint = "You are biased toward analyzing existing PML specs and generated Python, finding issues and opportunities to refine PML."
    else:
        hint = "You dynamically choose between PML design, compilation, evolution, and analysis based on the task."

    tools_text = (
        "Available tools (always call them instead of guessing):\n"
        "- pml_list_specs {}\n"
        "- pml_read_spec {name: str, max_chars?: int}\n"
        "- pml_write_spec {name: str, content: str, append?: bool}\n"
        "- pml_list_generated {}\n"
        "- pml_read_generated {path: str, max_chars?: int}\n"
        "- pml_describe_repo {max_depth?: int, max_entries?: int}\n"
        "- pml_scan_python_relations {}\n"
        "- pml_index {}\n"
        "- pml_search {query: str, max_hits?: int}\n"
        "- pml_read_docs {max_chars?: int}\n"
        "- pml_write_docs {content: str, append?: bool}\n"
        "- pml_compile {name: str}\n"
        "- pml_run {name: str, entry?: str='main.py', python_cmd?: str='python'}\n"
        "- pml_run_tests {pattern?: str}\n"
        "- git_checkpoint {message?: str, add_all?: bool}\n"
        "- run_shell {cmd: str, timeout?: int} (only python/py/pytest/pip/uv/git)\n\n"
        "At EACH step you must respond with ONLY ONE JSON OBJECT, no markdown, no surrounding text.\n"
        "The JSON must be EXACTLY ONE of these forms:\n"
        "1) Call a tool:\n"
        '{"action": "tool", "tool_name": "pml_list_specs", "tool_args": {}, "reasoning": "short explanation"}\n'
        "2) Finish with an answer:\n"
        '{"action": "final", "answer": "markdown answer here", "reasoning": "short explanation"}\n'
        "Do NOT wrap this object inside 'tools' or 'action_schema'. Just output that single JSON object.\n"
    )

    pml_text = (
        "You exist ONLY to work with Project Me Language (PML):\n"
        "- Design / evolve the PML syntax and conventions.\n"
        "- Create and edit .pml specs under ./pml.\n"
        "- Keep ./PML_DOCS.md as the canonical PML documentation for humans and for yourself.\n"
        "- Compile PML to Python in ./pml_out via pml_compile / pml_run.\n"
        "- Use repository structure and python relations to keep modules structured and imports consistent.\n"
        "- Over time, converge PML into a powerful, concise DSL for full systems (services, agents, pipelines), not just scripts.\n"
        "- Never step outside PML-related goals.\n"
    )

    return (
        "You are Project Me PML Agent, a specialist focused entirely on Project Me Language (PML).\n"
        + hint
        + "\n\n"
        + pml_text
        + "\n"
        + tools_text
    )


# -------------------------
# JSON handling + agent loop
# -------------------------


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    # Fast path: a single JSON object on one of the lines
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    # Slow path: try the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def normalize_action_obj(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    if "action" in raw_obj:
        return raw_obj
    schema = raw_obj.get("action_schema")
    if isinstance(schema, dict) and "action" in schema:
        log("[AGENT] Normalizing action_schema wrapper from model.")
        return schema
    if "tools" in raw_obj and "action_schema" in raw_obj:
        return normalize_action_obj(raw_obj["action_schema"])
    return raw_obj


def run_tool(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOLS.get(tool_name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {tool_name}"}
    try:
        result = fn(**tool_args)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_agent(
    prompt: str,
    mode: str,
    max_steps: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    rag = build_rag_context(prompt)
    messages: List[Dict[str, str]] = []
    system_text = build_system_prompt(mode)
    messages.append({"role": "system", "content": system_text})
    if rag:
        log("[RAG] Selected memory snippets.")
        messages.append({"role": "system", "content": "Relevant working memory:\n" + rag})
    else:
        log("[RAG] No relevant memory found for this prompt.")
    messages.append({"role": "user", "content": prompt})

    tool_calls: List[Dict[str, Any]] = []
    last_raw = ""
    start_time = time.time()

    for step in range(1, max_steps + 1):
        _maybe_throttle_for_health()
        log(f"[AGENT] Step {step}/{max_steps} calling LM Studio…")
        raw = _call_lm(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        last_raw = raw
        log(f"[AGENT] Raw reply length={len(raw)} chars")

        data = extract_json_block(raw)
        if not data or not isinstance(data, dict):
            answer = raw.strip()
            append_memory(prompt, answer, tool_calls)
            total = time.time() - start_time
            log(f"[AGENT] No valid JSON action, returning raw answer after {step} steps in {total:.2f}s.")
            return answer

        data = normalize_action_obj(data)
        action = str(data.get("action", "final")).lower()

        if action == "final":
            answer = str(data.get("answer") or raw).strip()
            append_memory(prompt, answer, tool_calls)
            total = time.time() - start_time
            log(f"[AGENT] Finished with action=final in {total:.2f}s and {step} steps")
            return answer

        if action == "tool":
            tool_name = str(data.get("tool_name"))
            tool_args = data.get("tool_args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}
            reason = str(data.get("reasoning") or "")[:MAX_TOOL_REASON_CHARS]
            log(f"[AGENT] Tool requested: {tool_name} with args {tool_args} | reason={reason}")
            result = run_tool(tool_name, tool_args)
            tool_calls.append(
                {
                    "step": step,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result,
                }
            )
            # Echo the tool call back as assistant, then inject tool result as a "user" message
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action": "tool",
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                        }
                    ),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool result for "
                        + str(tool_name)
                        + ":\n```json\n"
                        + json.dumps(result, indent=2)[:MAX_TOOL_RESULT_CHARS]
                        + "\n```\nUse this observation and either call another tool or return a 'final' answer."
                    ),
                }
            )
            continue

        # Fallback: unknown action, treat as final answer
        answer = str(data.get("answer") or raw).strip()
        append_memory(prompt, answer, tool_calls)
        total = time.time() - start_time
        log(f"[AGENT] Unknown action '{action}', returning answer after {step} steps in {total:.2f}s")
        return answer

    log("[AGENT] Max steps reached, returning last reply.")
    answer = last_raw.strip()
    append_memory(prompt, answer, tool_calls)
    return answer


# -------------------------
# CLI commands
# -------------------------


def cmd_warmup(args: argparse.Namespace) -> None:
    log("Running warmup call…")
    prompt = "Just reply with the single word: Ready."
    out = run_agent(
        prompt=prompt,
        mode="design",
        max_steps=1,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
    )
    log(f"Warmup output: {out!r}")


def cmd_chat(args: argparse.Namespace) -> None:
    log("Starting PML-focused chat...")
    answer = run_agent(
        prompt=args.message,
        mode=args.mode,
        max_steps=args.steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print()
    print("=== MODEL RESPONSE ===")
    print()
    print(answer)
    print("======================")


def cmd_task(args: argparse.Namespace) -> None:
    log(f"Starting PML task with up to {args.steps} steps…")
    answer = run_agent(
        prompt=args.prompt,
        mode=args.mode,
        max_steps=args.steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print()
    print("=== TASK RESULT ===")
    print()
    print(answer)
    print("===================")


def _load_steps(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    steps_path = path or STEPS_PATH
    if not steps_path.exists():
        raise FileNotFoundError(f"steps.json not found at {steps_path}")
    with steps_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("steps.json must be a JSON array")
    return data


def _run_steps_once(
    steps: List[Dict[str, Any]],
    default_steps: int,
    default_max_tokens: int,
    default_temperature: float,
    default_top_p: float,
    log_path: Path,
) -> None:
    for idx, step in enumerate(steps):
        name = str(step.get("name", f"step_{idx}"))
        prompt = str(step.get("prompt", "")).strip()
        if not prompt:
            log(f"[PML-STEPS] Skipping step {idx} (no prompt).")
            continue
        mode = str(step.get("mode", "auto"))
        max_steps = int(step.get("max_steps", default_steps))
        max_tokens = int(step.get("max_tokens", default_max_tokens))
        temperature = float(step.get("temperature", default_temperature))
        top_p = float(step.get("top_p", default_top_p))

        log(f"[PML-STEPS] Step {idx+1}/{len(steps)} name={name} mode={mode} steps={max_steps}")
        t0 = time.time()
        answer = run_agent(
            prompt=prompt,
            mode=mode,
            max_steps=max_steps,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        dt = time.time() - t0
        rec = {
            "idx": idx,
            "name": name,
            "prompt": prompt,
            "mode": mode,
            "max_steps": max_steps,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "answer": answer,
            "elapsed_sec": dt,
            "ts": time.time(),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"[PML-STEPS] Step {idx+1} finished in {dt:.2f}s. Logged to {log_path}")


def cmd_steps(args: argparse.Namespace) -> None:
    log("[PML-STEPS] Loading steps.json…")
    steps = _load_steps()
    log(f"[PML-STEPS] Loaded {len(steps)} steps.")
    _run_steps_once(
        steps=steps,
        default_steps=args.steps,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
        default_top_p=args.top_p,
        log_path=PML_STEPS_LOG_PATH,
    )


def cmd_pml_create(args: argparse.Namespace) -> None:
    """
    Infinite (or bounded) PML loop: this is basically your
    `python main.py pml-create --loop-sleep 60 --git-checkpoint` mode.
    """
    log("[PML-CREATE] Starting PML create loop…")
    steps_path = Path(args.steps_path).resolve() if args.steps_path else STEPS_PATH
    loop = 0
    while True:
        loop += 1
        log(f"[PML-CREATE] Loop {loop} reading steps from {steps_path}")
        steps = _load_steps(steps_path)
        _run_steps_once(
            steps=steps,
            default_steps=args.steps,
            default_max_tokens=args.max_tokens,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            log_path=PML_CREATE_LOG_PATH,
        )
        if args.git_checkpoint:
            log("[PML-CREATE] Creating git checkpoint…")
            res = tool_git_checkpoint(message=f"pml-create loop {loop}")
            log(f"[PML-CREATE] git checkpoint result: {res}")
        if args.max_loops and loop >= args.max_loops:
            log(f"[PML-CREATE] Reached max_loops={args.max_loops}, stopping.")
            break
        if args.loop_sleep > 0:
            log(f"[PML-CREATE] Sleeping {args.loop_sleep}s before next loop…")
            time.sleep(args.loop_sleep)


def cmd_pml_compile(args: argparse.Namespace) -> None:
    log(f"[PML] Compiling PML spec '{args.name}'…")
    res = tool_pml_compile(args.name)
    print()
    print("=== PML COMPILE RESULT ===")
    print(json.dumps(res, indent=2))
    print("==========================")


def cmd_pml_run(args: argparse.Namespace) -> None:
    log(f"[PML] Compiling and running PML spec '{args.name}' entry='{args.entry}'…")
    res = tool_pml_run(args.name, entry=args.entry, python_cmd=args.python_cmd)
    print()
    print("=== PML RUN RESULT ===")
    print(json.dumps(res, indent=2))
    print("======================")


# -------------------------
# Arg parser + entrypoint
# -------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project_me_pml",
        description="Project Me PML-only agent (design, evolve, compile, and run PML).",
    )
    sub = parser.add_subparsers(dest="command")

    # warmup
    p_warm = sub.add_parser("warmup", help="Prime the model and connection (PML-focused)")
    p_warm.set_defaults(func=cmd_warmup)

    # chat
    p_chat = sub.add_parser("chat", help="Single PML-focused conversation")
    p_chat.add_argument("message", help="PML-related prompt to send")
    p_chat.add_argument("--mode", choices=["auto", "design", "compiler", "evolve", "analyze"], default="auto")
    p_chat.add_argument("--steps", type=int, default=1, help="Max agent steps (default 1 for chat-style)")
    p_chat.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_chat.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_chat.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p_chat.set_defaults(func=cmd_chat)

    # task
    p_task = sub.add_parser("task", help="Run a multi-step PML agent task")
    p_task.add_argument("prompt", help="High-level PML instruction")
    p_task.add_argument("--mode", choices=["auto", "design", "compiler", "evolve", "analyze"], default="auto")
    p_task.add_argument("--steps", type=int, default=4, help="Max agent steps")
    p_task.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_task.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_task.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p_task.set_defaults(func=cmd_task)

    # steps.json one-shot
    p_steps = sub.add_parser("steps", help="Run a sequence of PML tasks defined in steps.json")
    p_steps.add_argument("--steps", type=int, default=4, help="Default max agent steps per item")
    p_steps.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_steps.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_steps.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p_steps.set_defaults(func=cmd_steps)

    # main long-running loop
    p_create = sub.add_parser("pml-create", help="Run PML steps in a loop until stopped")
    p_create.add_argument("--steps-path", default=str(STEPS_PATH), help="Path to steps.json (default: ./steps.json)")
    p_create.add_argument("--steps", type=int, default=4, help="Default max agent steps per item")
    p_create.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_create.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_create.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p_create.add_argument("--loop-sleep", type=int, default=60, help="Seconds to sleep between loops (default: 60)")
    p_create.add_argument("--max-loops", type=int, default=0, help="Stop after N loops (0 = infinite)")
    p_create.add_argument(
        "--git-checkpoint",
        action="store_true",
        help="Create an automatic git commit after each loop for backup.",
    )
    p_create.set_defaults(func=cmd_pml_create)

    # compile-only
    p_pml_c = sub.add_parser("pml-compile", help="Compile a PML spec into Python modules")
    p_pml_c.add_argument("name", help="PML spec name (with or without .pml)")
    p_pml_c.set_defaults(func=cmd_pml_compile)

    # compile + run
    p_pml_r = sub.add_parser("pml-run", help="Compile and run a PML spec")
    p_pml_r.add_argument("name", help="PML spec name (with or without .pml)")
    p_pml_r.add_argument("--entry", default="main.py", help="Entry Python file inside pml_out (default: main.py)")
    p_pml_r.add_argument("--python-cmd", default="python", help="Python command to use (default: python)")
    p_pml_r.set_defaults(func=cmd_pml_run)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(1)
    log(f"[LM] Using LM Studio at {LM_URL} with model '{LM_MODEL}'")
    ensure_pml_dirs()
    ensure_pml_docs_seeded()
    args.func(args)


if __name__ == "__main__":
    main()
