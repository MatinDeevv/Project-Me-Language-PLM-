#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import requests

ActionType = Literal["tool", "final"]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    tid = threading.get_ident()
    print(f"[{_ts()}][T{tid}] {msg}", flush=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def now() -> float:
    return time.time()


def try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


@dataclass
class AppConfig:
    lm_url: str = "http://127.0.0.1:1234/v1/chat/completions"
    lm_model: str = "qwen/qwen3-coder-30b"
    stream: bool = False
    ssl_verify: bool = True

    timeout_s: Optional[float] = None

    ctx_len: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.9

    agent_calls: int = 6
    sleep_between: int = 0
    loop_till_stopped: bool = False

    monitor: bool = True
    monitor_interval_s: int = 10
    max_gpu_temp_c: int = 86
    max_cpu_temp_c: int = 95
    cooldown_s: int = 600

    auto_approve: bool = True
    safe_mode: bool = True

    git_auto: bool = True
    git_push: bool = True
    git_interval_s: int = 300

    rag_k: int = 3
    max_rag_chars: int = 2500
    max_workspace_chars: int = 2500


class ProjectPaths:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.pml_dir = base_dir / "pml"
        self.db_path = self.pml_dir / "pml.db"
        self.docs_path = self.pml_dir / "PML.md"
        self.steps_json_path = self.pml_dir / "steps.json"
        self.heartbeat_path = self.pml_dir / "heartbeat.json"
        self.runlog_path = self.pml_dir / "runlog.jsonl"
        self.lock_path = self.pml_dir / "run.lock"

    def ensure(self) -> None:
        self.pml_dir.mkdir(parents=True, exist_ok=True)


class SingleInstanceLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            raise RuntimeError(
                f"Lock exists at {self.lock_path}. Another run might be active. lock={data}"
            )
        self.lock_path.write_text(jdump({"pid": os.getpid(), "ts": now()}), encoding="utf-8")

    def release(self) -> None:
        try:
            self.lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


class HealthMonitor:
    def __init__(self, cfg: AppConfig, paths: ProjectPaths):
        self.cfg = cfg
        self.paths = paths
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.psutil = try_import_psutil()

    def start(self) -> None:
        if not self.cfg.monitor:
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _gpu_stats(self) -> Dict[str, Any]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if p.returncode != 0:
                return {"ok": False, "err": (p.stderr or "").strip()[:200]}
            line = (p.stdout or "").strip().splitlines()[0]
            parts = [x.strip() for x in line.split(",")]
            temp = int(parts[0])
            mem_used = int(parts[1])
            mem_total = int(parts[2])
            util = int(parts[3])
            return {
                "ok": True,
                "temp_c": temp,
                "mem_used_mb": mem_used,
                "mem_total_mb": mem_total,
                "util_pct": util,
            }
        except Exception:
            return {"ok": False, "err": "nvidia-smi unavailable"}

    def _cpu_ram_stats(self) -> Dict[str, Any]:
        if not self.psutil:
            return {"ok": False, "err": "psutil not installed"}
        try:
            cpu = float(self.psutil.cpu_percent(interval=None))
            vm = self.psutil.virtual_memory()
            return {
                "ok": True,
                "cpu_pct": cpu,
                "ram_pct": float(vm.percent),
                "ram_used_gb": round(vm.used / (1024**3), 2),
                "ram_total_gb": round(vm.total / (1024**3), 2),
            }
        except Exception as e:
            return {"ok": False, "err": str(e)}

    def _disk_stats(self) -> Dict[str, Any]:
        try:
            usage = shutil.disk_usage(self.paths.base_dir)
            return {
                "ok": True,
                "free_gb": round(usage.free / (1024**3), 2),
                "total_gb": round(usage.total / (1024**3), 2),
            }
        except Exception as e:
            return {"ok": False, "err": str(e)}

    def _loop(self) -> None:
        while not self._stop.is_set():
            stats = {
                "ts": now(),
                "cpu": self._cpu_ram_stats(),
                "gpu": self._gpu_stats(),
                "disk": self._disk_stats(),
            }
            try:
                self.paths.pml_dir.mkdir(parents=True, exist_ok=True)
                self.paths.heartbeat_path.write_text(jdump(stats), encoding="utf-8")
            except Exception:
                pass

            gpu = stats.get("gpu") or {}
            temp = gpu.get("temp_c") if gpu.get("ok") else None

            if isinstance(temp, int) and temp >= self.cfg.max_gpu_temp_c:
                log(
                    f"[HEALTH] GPU temp high ({temp}°C >= {self.cfg.max_gpu_temp_c}°C). Cooling {self.cfg.cooldown_s}s"
                )
                slept = 0
                while slept < self.cfg.cooldown_s and not self._stop.is_set():
                    time.sleep(1)
                    slept += 1
                continue

            time.sleep(self.cfg.monitor_interval_s)


class LMStudioClient:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, payload: Dict[str, Any]) -> requests.Response:
        t0 = now()
        log(
            f"[LM] POST -> {self.cfg.lm_url} | max_tokens={payload.get('max_tokens')}, temp={payload.get('temperature')}, top_p={payload.get('top_p')}, stream={payload.get('stream')}"
        )
        resp = self.session.post(
            self.cfg.lm_url,
            json=payload,
            timeout=self.cfg.timeout_s,
            verify=self.cfg.ssl_verify,
            stream=bool(payload.get("stream")),
        )
        dt = now() - t0
        log(f"[LM] Response status: {resp.status_code} in {dt:.2f}s")
        return resp

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.cfg.lm_model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": bool(stream),
        }

        resp = self._request(payload)

        if resp.status_code >= 400:
            try:
                body = resp.text[:4000]
            except Exception:
                body = ""
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} from LM Studio. Body (truncated): {body}",
                response=resp,
            )

        if not stream:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        out = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except Exception:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content")
            if token:
                out.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
        return "".join(out)


class TokenBudgeter:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def estimate_messages(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for m in messages:
            total += self.estimate_tokens(m.get("content", "")) + 8
        return total

    def shrink_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return head + "\n…<snip>…\n" + tail

    def fit(
        self,
        messages: List[Dict[str, str]],
        *,
        reserve_output_tokens: int,
        max_workspace_chars: int,
        max_rag_chars: int,
    ) -> List[Dict[str, str]]:
        ctx = int(self.cfg.ctx_len)
        reserve = int(reserve_output_tokens)
        if reserve >= ctx:
            reserve = max(256, ctx // 2)

        def _hard_trim(ms: List[Dict[str, str]]) -> List[Dict[str, str]]:
            trimmed: List[Dict[str, str]] = []
            for i, m in enumerate(ms):
                c = m.get("content", "")
                if i == 0 and m.get("role") == "system":
                    trimmed.append({"role": "system", "content": self.shrink_text(c, 3000)})
                    continue
                if "PML workspace" in c:
                    trimmed.append({"role": m.get("role", "system"), "content": self.shrink_text(c, max_workspace_chars)})
                elif c.startswith("Relevant memory"):
                    trimmed.append({"role": m.get("role", "system"), "content": self.shrink_text(c, max_rag_chars)})
                else:
                    trimmed.append({"role": m.get("role", "user"), "content": self.shrink_text(c, 6000)})
            return trimmed

        ms = _hard_trim(messages)
        budget = ctx - reserve
        if budget < 512:
            budget = 512

        while self.estimate_messages(ms) > budget:
            if len(ms) <= 2:
                ms = _hard_trim(ms)
                break
            removed = False
            for i in range(1, len(ms) - 1):
                if ms[i].get("role") == "system":
                    ms.pop(i)
                    removed = True
                    break
            if removed:
                continue
            ms.pop(1)

        return ms


class ProjectDB:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._init_schema()
        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL,
              prompt TEXT NOT NULL,
              answer TEXT NOT NULL,
              meta TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
              id TEXT PRIMARY KEY,
              phase TEXT NOT NULL,
              title TEXT NOT NULL,
              body TEXT NOT NULL,
              status TEXT NOT NULL,
              created REAL NOT NULL,
              updated REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL,
              kind TEXT NOT NULL,
              payload TEXT NOT NULL
            )
            """
        )
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(prompt, answer, content='memory', content_rowid='id')
                """
            )
            cur.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                  INSERT INTO memory_fts(rowid, prompt, answer) VALUES (new.id, new.prompt, new.answer);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                  INSERT INTO memory_fts(memory_fts, rowid, prompt, answer) VALUES ('delete', old.id, old.prompt, old.answer);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                  INSERT INTO memory_fts(memory_fts, rowid, prompt, answer) VALUES ('delete', old.id, old.prompt, old.answer);
                  INSERT INTO memory_fts(rowid, prompt, answer) VALUES (new.id, new.prompt, new.answer);
                END;
                """
            )
        except Exception:
            pass
        self.conn.commit()

    def add_memory(self, prompt: str, answer: str, meta: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory(ts,prompt,answer,meta) VALUES (?,?,?,?)",
                (now(), prompt, answer, json.dumps(meta, ensure_ascii=False)),
            )
            self.conn.commit()

    def search_memory(self, query: str, k: int) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        with self._lock:
            try:
                cur = self.conn.execute(
                    """
                    SELECT m.prompt, m.answer, m.meta
                    FROM memory_fts f
                    JOIN memory m ON m.id = f.rowid
                    WHERE memory_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (self._fts_query(q), int(k)),
                )
                rows = cur.fetchall()
            except Exception:
                cur = self.conn.execute(
                    "SELECT prompt, answer, meta FROM memory ORDER BY id DESC LIMIT ?",
                    (int(k),),
                )
                rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for p, a, meta in rows:
            try:
                meta_obj = json.loads(meta)
            except Exception:
                meta_obj = {}
            out.append({"prompt": p, "answer": a, "meta": meta_obj})
        return out

    def _fts_query(self, text: str) -> str:
        tokens = [t for t in re.split(r"\W+", text.lower()) if len(t) >= 3]
        if not tokens:
            return text
        tokens = tokens[:12]
        return " OR ".join(tokens)

    def upsert_step(self, step_id: str, phase: str, title: str, body: str, status: str) -> None:
        ts = now()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO steps(id,phase,title,body,status,created,updated)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  phase=excluded.phase,
                  title=excluded.title,
                  body=excluded.body,
                  status=excluded.status,
                  updated=excluded.updated
                """,
                (step_id, phase, title, body, status, ts, ts),
            )
            self.conn.commit()

    def next_pending_step(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, phase, title, body, status FROM steps WHERE status='todo' ORDER BY created ASC LIMIT 1"
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "phase": row[1], "title": row[2], "body": row[3], "status": row[4]}

    def mark_step(self, step_id: str, status: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE steps SET status=?, updated=? WHERE id=?",
                (status, now(), step_id),
            )
            self.conn.commit()

    def list_steps(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, phase, title, status, created, updated FROM steps ORDER BY created DESC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "phase": r[1],
                    "title": r[2],
                    "status": r[3],
                    "created": r[4],
                    "updated": r[5],
                }
            )
        return out

    def add_event(self, kind: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(ts,kind,payload) VALUES (?,?,?)",
                (now(), kind, json.dumps(payload, ensure_ascii=False)),
            )
            self.conn.commit()


class Workspace:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.exclude = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}

    def safe_path(self, rel: str) -> Path:
        p = (self.paths.base_dir / rel).resolve()
        if not str(p).startswith(str(self.paths.base_dir.resolve())):
            raise ValueError("Path escapes project root")
        return p

    def snapshot_tree(self, max_depth: int = 2, max_entries: int = 200) -> str:
        base = self.paths.base_dir

        def _walk(d: Path, depth: int) -> Iterable[str]:
            if depth > max_depth:
                return
            try:
                entries = sorted(d.iterdir(), key=lambda x: x.name.lower())
            except Exception:
                return
            for e in entries:
                if e.name in self.exclude:
                    continue
                rel = str(e.relative_to(base)).replace("\\", "/")
                if e.is_dir():
                    yield f"{rel}/"
                    yield from _walk(e, depth + 1)
                else:
                    yield rel

        out = []
        for i, line in enumerate(_walk(base, 0)):
            if i >= max_entries:
                out.append("… (tree truncated) …")
                break
            out.append(line)
        return "\n".join(out)


class ToolRunner:
    def __init__(self, cfg: AppConfig, ws: Workspace):
        self.cfg = cfg
        self.ws = ws

    def list_dir(self, path: str = ".") -> Dict[str, Any]:
        p = self.ws.safe_path(path)
        items = []
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if child.name in self.ws.exclude:
                continue
            try:
                st = child.stat()
                size = st.st_size
            except Exception:
                size = 0
            items.append({"name": child.name, "is_dir": child.is_dir(), "size": size})
        return {"cwd": str(p), "items": items}

    def read_file(self, path: str, max_chars: int = 12000) -> Dict[str, Any]:
        p = self.ws.safe_path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))
        data = p.read_text(encoding="utf-8", errors="ignore")
        truncated = False
        if len(data) > max_chars:
            truncated = True
            data = data[:max_chars]
        return {"path": str(p), "content": data, "truncated": truncated}

    def write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        p = self.ws.safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
        return {"path": str(p), "bytes": len(content), "append": append}

    def mkdir(self, path: str, exist_ok: bool = True) -> Dict[str, Any]:
        p = self.ws.safe_path(path)
        p.mkdir(parents=True, exist_ok=exist_ok)
        return {"path": str(p), "exists": True}

    def move_path(self, src: str, dst: str) -> Dict[str, Any]:
        s = self.ws.safe_path(src)
        d = self.ws.safe_path(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)
        return {"src": str(s), "dst": str(d)}

    def delete_path(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        if self.cfg.safe_mode and not self.cfg.auto_approve:
            raise PermissionError("delete_path blocked: safe_mode without auto_approve")
        p = self.ws.safe_path(path)
        if p == self.ws.paths.base_dir:
            raise ValueError("Refusing to delete project root")
        if p.is_dir():
            if not recursive:
                p.rmdir()
            else:
                for child in sorted(p.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink(missing_ok=True)  # type: ignore[arg-type]
                    else:
                        child.rmdir()
                p.rmdir()
        elif p.exists():
            p.unlink()
        return {"path": str(p), "deleted": True}

    def run_shell(self, cmd: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        if self.cfg.safe_mode and not self.cfg.auto_approve:
            raise PermissionError("run_shell blocked: safe_mode without auto_approve")
        cmd_strip = (cmd or "").strip()
        if not cmd_strip:
            raise ValueError("Empty cmd")

        allowed_prefixes = (
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
        )
        first = cmd_strip.split()[0].lower()
        if not any(first.startswith(p) for p in allowed_prefixes):
            raise ValueError("Command not allowed by allowlist")

        p = subprocess.run(
            cmd_strip,
            shell=True,
            cwd=str(self.ws.paths.base_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd_strip,
            "returncode": p.returncode,
            "stdout": (p.stdout or "")[-4000:],
            "stderr": (p.stderr or "")[-4000:],
        }

    def git_status(self) -> Dict[str, Any]:
        return self.run_shell("git status", timeout=60)

    def git_commit_push(self, message: str) -> Dict[str, Any]:
        if not self.cfg.git_auto:
            return {"ok": False, "error": "git_auto disabled"}
        msg = (message or "auto").strip()[:120]
        r1 = self.run_shell("git add -A", timeout=120)
        r2 = self.run_shell(f"git commit -m \"{msg}\"", timeout=120)
        out = {"add": r1, "commit": r2}
        if self.cfg.git_push:
            r3 = self.run_shell("git push", timeout=300)
            out["push"] = r3
        return out

    def tools_schema(self) -> str:
        return (
            "Available tools (call instead of guessing):\n"
            "- list_dir {path?: str='.'}\n"
            "- read_file {path: str, max_chars?: int}\n"
            "- write_file {path: str, content: str, append?: bool}\n"
            "- mkdir {path: str}\n"
            "- move_path {src: str, dst: str}\n"
            "- delete_path {path: str, recursive?: bool}\n"
            "- run_shell {cmd: str, timeout?: int|null} (allowlist: python/py/pytest/pip/uv/dir/ls/type/cat/echo/git)\n"
            "- git_status {}\n"
            "- git_commit_push {message: str}\n"
        )

    def run(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, tool_name, None)
        if not callable(fn):
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            res = fn(**(tool_args or {}))
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def normalize_action(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "action" in obj:
        return obj
    schema = obj.get("action_schema")
    if isinstance(schema, dict) and "action" in schema:
        log("[AGENT] Normalizing action_schema wrapper")
        return schema
    return obj


class Agent:
    def __init__(
        self,
        cfg: AppConfig,
        lm: LMStudioClient,
        db: ProjectDB,
        ws: Workspace,
        tools: ToolRunner,
    ):
        self.cfg = cfg
        self.lm = lm
        self.db = db
        self.ws = ws
        self.tools = tools
        self.budgeter = TokenBudgeter(cfg)

    def _system_prompt(self, mode: str) -> str:
        role_hint = {
            "bootstrap": "Bias: bootstrap multi-file Python projects with clean structure + tests.",
            "refactor": "Bias: minimal-diff refactors; change the least, improve the most.",
            "review": "Bias: review only; do not edit files unless asked.",
            "pml": "Bias: evolve PML (a DSL that compiles to Python) and its runtime, but keep file-count small.",
            "auto": "Bias: choose bootstrap/refactor/review dynamically.",
        }.get(mode, "Bias: auto.")

        return (
            "You are Project Me: a senior full-stack engineer + systems architect.\n"
            f"{role_hint}\n\n"
            "Rules:\n"
            "- ALWAYS use tools to inspect workspace before writing/renaming/deleting.\n"
            "- Keep diffs minimal when editing existing code.\n"
            "- Prefer few files. Avoid file explosions.\n"
            "- Output EXACTLY ONE JSON object per message. No markdown, no extra text.\n\n"
            + self.tools.tools_schema()
            + "\nAction schema (choose ONE):\n"
            + '{"action":"tool","tool_name":"list_dir","tool_args":{},"reasoning":"short"}\n'
            + '{"action":"final","answer":"markdown answer","reasoning":"short"}\n'
        )

    def _rag_context(self, prompt: str) -> str:
        hits = self.db.search_memory(prompt, k=self.cfg.rag_k)
        if not hits:
            return ""
        chunks = []
        for h in hits:
            p = (h.get("prompt") or "")[:300]
            a = (h.get("answer") or "")[:500]
            chunks.append(f"Prompt: {p}\nAnswer: {a}")
        txt = "\n\n".join(chunks)
        if len(txt) > self.cfg.max_rag_chars:
            txt = txt[: self.cfg.max_rag_chars] + "\n…(rag truncated)…"
        return txt

    def _workspace_context(self) -> str:
        tree = self.ws.snapshot_tree(max_depth=2, max_entries=160)
        if len(tree) > self.cfg.max_workspace_chars:
            tree = tree[: self.cfg.max_workspace_chars] + "\n…(tree truncated)…"
        return tree

    def _lm_call(self, messages: List[Dict[str, str]], *, max_tokens: int) -> str:
        fitted = self.budgeter.fit(
            messages,
            reserve_output_tokens=max_tokens,
            max_workspace_chars=self.cfg.max_workspace_chars,
            max_rag_chars=self.cfg.max_rag_chars,
        )

        try:
            return self.lm.chat(
                fitted,
                max_tokens=max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                stream=self.cfg.stream,
            )
        except requests.HTTPError as e:
            msg = str(e)
            if "context" in msg.lower() or "overflow" in msg.lower() or "n_ctx" in msg.lower():
                log("[AGENT] Context overflow detected; retrying with aggressive trimming")
                smaller = []
                for m in fitted:
                    if m.get("role") == "system" and (
                        m.get("content", "").startswith("PML workspace")
                        or m.get("content", "").startswith("Relevant memory")
                    ):
                        continue
                    smaller.append(m)
                smaller = self.budgeter.fit(
                    smaller,
                    reserve_output_tokens=max_tokens,
                    max_workspace_chars=800,
                    max_rag_chars=800,
                )
                return self.lm.chat(
                    smaller,
                    max_tokens=max_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    stream=self.cfg.stream,
                )
            raise

    def run_action_loop(
        self,
        *,
        prompt: str,
        mode: str,
        max_calls: int,
        meta: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        system = self._system_prompt(mode)
        ws_ctx = self._workspace_context()
        rag = self._rag_context(prompt)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "system", "content": "PML workspace tree:\n" + ws_ctx},
        ]
        if rag:
            messages.append({"role": "system", "content": "Relevant memory (RAG):\n" + rag})
        messages.append({"role": "user", "content": prompt})

        tool_calls: List[Dict[str, Any]] = []
        last_raw = ""

        for i in range(1, max_calls + 1):
            log(f"[AGENT] LM call {i}/{max_calls}")
            raw = self._lm_call(messages, max_tokens=self.cfg.max_tokens)
            last_raw = raw
            log(f"[AGENT] Raw reply length={len(raw)} chars")

            obj = extract_first_json_obj(raw)
            if not obj:
                answer = raw.strip()
                self.db.add_memory(prompt, answer, {**meta, "tool_calls": tool_calls, "mode": mode})
                return answer, tool_calls

            obj = normalize_action(obj)
            action = str(obj.get("action", "final")).lower()

            if action == "final":
                answer = str(obj.get("answer") or "").strip() or raw.strip()
                self.db.add_memory(prompt, answer, {**meta, "tool_calls": tool_calls, "mode": mode})
                return answer, tool_calls

            if action == "tool":
                tool_name = str(obj.get("tool_name") or "")
                tool_args = obj.get("tool_args")
                if not isinstance(tool_args, dict):
                    tool_args = {}
                log(f"[AGENT] Tool requested: {tool_name} args={tool_args}")

                result = self.tools.run(tool_name, tool_args)
                tool_calls.append({"i": i, "tool": tool_name, "args": tool_args, "result": result})

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"action": "tool", "tool_name": tool_name, "tool_args": tool_args}),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool result:\n```json\n" + json.dumps(result, indent=2) + "\n```\n"
                        "Use this observation. Either call another tool or return action=final.",
                    }
                )
                continue

            answer = raw.strip()
            self.db.add_memory(prompt, answer, {**meta, "tool_calls": tool_calls, "mode": mode})
            return answer, tool_calls

        answer = last_raw.strip()
        self.db.add_memory(prompt, answer, {**meta, "tool_calls": tool_calls, "mode": mode, "max_calls": max_calls})
        return answer, tool_calls


PML_DOCS = """# PML (Project Me Language)

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

"""


def default_seed_steps() -> List[Dict[str, Any]]:
    return [
        {
            "id": "runtime-0001",
            "phase": "runtime",
            "title": "Bootstrap PML runtime inside this repo",
            "body": "Ensure pml/ exists, pml.db exists, and the CLI can run pml-loop safely for days. Keep files minimal.",
            "status": "todo",
        },
        {
            "id": "spec-0001",
            "phase": "spec",
            "title": "Define PML syntax sketch",
            "body": "Write a compact syntax sketch: modules, contracts, pipelines. Store in pml/PML.md (append).",
            "status": "todo",
        },
        {
            "id": "compiler-0001",
            "phase": "compiler",
            "title": "Compiler plan",
            "body": "Design compilation pipeline to Python: parse -> IR -> emit. Keep it minimal. Define acceptance tests.",
            "status": "todo",
        },
    ]


class PMLSteps:
    def __init__(self, paths: ProjectPaths, db: ProjectDB):
        self.paths = paths
        self.db = db

    def init_files(self) -> None:
        self.paths.ensure()
        if not self.paths.docs_path.exists():
            self.paths.docs_path.write_text(PML_DOCS, encoding="utf-8")
        if not self.paths.steps_json_path.exists():
            seed = default_seed_steps()
            self.paths.steps_json_path.write_text(jdump(seed), encoding="utf-8")
        self._import_steps_json()

    def _import_steps_json(self) -> None:
        try:
            arr = json.loads(self.paths.steps_json_path.read_text(encoding="utf-8"))
        except Exception:
            arr = []
        if not isinstance(arr, list):
            arr = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("id") or "").strip()
            phase = str(it.get("phase") or "pml").strip()
            title = str(it.get("title") or sid).strip()
            body = str(it.get("body") or "").strip()
            status = str(it.get("status") or "todo").strip()
            if not sid:
                continue
            if status not in {"todo", "doing", "done", "blocked"}:
                status = "todo"
            self.db.upsert_step(sid, phase, title, body, status)

    def export_steps_json(self) -> None:
        steps = self.db.list_steps(limit=2000)
        by_id: Dict[str, Dict[str, Any]] = {}
        for s in steps:
            sid = s["id"]
            row = self._get_step_full(sid)
            if row:
                by_id[sid] = row
        ordered = sorted(by_id.values(), key=lambda x: x["id"])
        self.paths.steps_json_path.write_text(jdump(ordered), encoding="utf-8")

    def _get_step_full(self, step_id: str) -> Optional[Dict[str, Any]]:
        with self.db._lock:
            cur = self.db.conn.execute(
                "SELECT id, phase, title, body, status FROM steps WHERE id=?",
                (step_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {"id": r[0], "phase": r[1], "title": r[2], "body": r[3], "status": r[4]}


class GitBackup:
    def __init__(self, cfg: AppConfig, tools: ToolRunner):
        self.cfg = cfg
        self.tools = tools
        self._last_push = 0.0

    def maybe_commit(self, message: str) -> None:
        if not self.cfg.git_auto:
            return
        if now() - self._last_push < self.cfg.git_interval_s:
            return
        self._last_push = now()
        try:
            log(f"[GIT] Auto backup: {message}")
            res = self.tools.git_commit_push(message=message)
            log(f"[GIT] Backup result: ok={res.get('add',{}).get('returncode',0)==0}")
        except Exception as e:
            log(f"[GIT] Backup failed: {e}")


class PMLRunner:
    def __init__(
        self,
        cfg: AppConfig,
        paths: ProjectPaths,
        db: ProjectDB,
        agent: Agent,
        tools: ToolRunner,
        steps: PMLSteps,
        git: GitBackup,
    ):
        self.cfg = cfg
        self.paths = paths
        self.db = db
        self.agent = agent
        self.tools = tools
        self.steps = steps
        self.git = git

    def _append_runlog(self, record: Dict[str, Any]) -> None:
        try:
            self.paths.pml_dir.mkdir(parents=True, exist_ok=True)
            with self.paths.runlog_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _generate_more_steps(self, reason: str) -> None:
        prompt = (
            "Generate 8 new PML steps to progress the language and runtime. "
            "Keep file-count small. Each step must be atomic and executable. "
            "Return action=final where answer is a JSON array of step objects with keys: id, phase, title, body, status='todo'. "
            f"Reason: {reason}"
        )
        answer, _ = self.agent.run_action_loop(
            prompt=prompt,
            mode="pml",
            max_calls=2,
            meta={"kind": "gen_steps", "reason": reason},
        )
        arr = None
        try:
            arr = json.loads(answer)
        except Exception:
            arr = None
        if not isinstance(arr, list):
            log("[PML] Step generator did not return a JSON array; skipping")
            return
        added = 0
        for it in arr:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("id") or "").strip()
            phase = str(it.get("phase") or "pml").strip()
            title = str(it.get("title") or sid).strip()
            body = str(it.get("body") or "").strip()
            status = str(it.get("status") or "todo").strip()
            if not sid or status != "todo":
                continue
            if len(sid) > 64:
                continue
            self.db.upsert_step(sid, phase, title, body, "todo")
            added += 1
        log(f"[PML] Added {added} new steps")
        self.steps.export_steps_json()

    def run_one_step(self, step: Dict[str, Any]) -> None:
        sid = step["id"]
        phase = step["phase"]
        title = step["title"]
        body = step["body"]

        self.db.mark_step(sid, "doing")
        self.steps.export_steps_json()

        prompt = (
            "Execute this PML step now.\n"
            f"id: {sid}\nphase: {phase}\ntitle: {title}\n\n"
            "Constraints:\n"
            "- Keep file count small. Prefer updating existing files.\n"
            "- Use tools for filesystem operations.\n"
            "- At the end, return action=final with: (1) what changed, (2) what to do next, (3) any files edited.\n\n"
            f"Step body:\n{body}\n"
        )

        t0 = now()
        answer, tool_calls = self.agent.run_action_loop(
            prompt=prompt,
            mode="pml",
            max_calls=self.cfg.agent_calls,
            meta={"kind": "pml_step", "step_id": sid, "phase": phase, "title": title},
        )
        dt = now() - t0

        self.db.mark_step(sid, "done")
        self.steps.export_steps_json()

        rec = {
            "ts": now(),
            "step_id": sid,
            "phase": phase,
            "title": title,
            "seconds": round(dt, 3),
            "tool_calls": tool_calls,
            "answer_preview": answer[:800],
        }
        self._append_runlog(rec)
        self.db.add_event("pml_step_done", rec)

        self.git.maybe_commit(f"pml: {sid} {title}")

        log(f"[PML] Step done: {sid} in {dt:.2f}s")

    def loop(self) -> None:
        log(
            f"[PML] Starting loop: agent_calls={self.cfg.agent_calls}, sleep_between={self.cfg.sleep_between}s (Ctrl+C to stop)"
        )

        iteration = 0
        while True:
            iteration += 1
            log(f"[PML] === Global iteration {iteration} ===")

            step = self.db.next_pending_step()
            if not step:
                log("[PML] No pending steps. Generating more.")
                self._generate_more_steps("queue empty")
                step = self.db.next_pending_step()
                if not step:
                    log("[PML] Still no steps after generation. Sleeping 10s")
                    time.sleep(10)
                    continue

            log(f"[PML] Working step={step['id']} phase={step['phase']} title={step['title']}")
            try:
                self.run_one_step(step)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"[EXC] [PML] Step crashed: {e}")
                self.db.add_event(
                    "pml_step_crash",
                    {"step": step, "err": str(e), "ts": now()},
                )
                try:
                    self.db.mark_step(step["id"], "blocked")
                except Exception:
                    pass
                self.git.maybe_commit(f"pml: crash {step['id']}")

            if not self.cfg.loop_till_stopped:
                break

            if self.cfg.sleep_between > 0:
                time.sleep(self.cfg.sleep_between)


def build_global_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lm-url", default=os.environ.get("LMSTUDIO_URL", AppConfig.lm_url))
    p.add_argument("--lm-model", default=os.environ.get("LMSTUDIO_MODEL", AppConfig.lm_model))

    p.add_argument("--ssl-verify", dest="ssl_verify", action="store_true")
    p.add_argument("--no-ssl-verify", dest="ssl_verify", action="store_false")
    p.set_defaults(ssl_verify=True)

    p.add_argument("--stream", dest="stream", action="store_true")
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.set_defaults(stream=False)

    p.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Request timeout seconds. 0 means no timeout.",
    )

    p.add_argument("--ctx-len", type=int, default=int(os.environ.get("LM_CTX_LEN", "4096")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("LM_MAX_TOKENS", "2048")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("LM_TEMPERATURE", "0.2")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("LM_TOP_P", "0.9")))

    p.add_argument("--agent-calls", type=int, default=int(os.environ.get("AGENT_CALLS", "6")))
    p.add_argument("--sleep-between", type=int, default=int(os.environ.get("SLEEP_BETWEEN", "0")))
    p.add_argument("--loop-till-stopped", action="store_true")

    p.add_argument("--monitor", dest="monitor", action="store_true")
    p.add_argument("--no-monitor", dest="monitor", action="store_false")
    p.set_defaults(monitor=True)

    p.add_argument("--max-gpu-temp", type=int, default=int(os.environ.get("MAX_GPU_TEMP", "86")))
    p.add_argument("--cooldown", type=int, default=int(os.environ.get("COOLDOWN", "600")))

    p.add_argument("--auto-approve", dest="auto_approve", action="store_true")
    p.add_argument("--no-auto-approve", dest="auto_approve", action="store_false")
    p.set_defaults(auto_approve=True)

    p.add_argument("--safe-mode", dest="safe_mode", action="store_true")
    p.add_argument("--no-safe-mode", dest="safe_mode", action="store_false")
    p.set_defaults(safe_mode=True)

    p.add_argument("--git-auto", dest="git_auto", action="store_true")
    p.add_argument("--no-git-auto", dest="git_auto", action="store_false")
    p.set_defaults(git_auto=True)

    p.add_argument("--git-push", dest="git_push", action="store_true")
    p.add_argument("--no-git-push", dest="git_push", action="store_false")
    p.set_defaults(git_push=True)

    p.add_argument("--git-interval", type=int, default=int(os.environ.get("GIT_INTERVAL", "300")))

    p.add_argument("--pml-create", action="store_true", help="Compatibility: init PML + start loop")

    return p


def build_cmd_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="project_me", description="Project Me PML agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_w = sub.add_parser("warmup", help="Test LM Studio connectivity")
    p_w.set_defaults(cmd="warmup")

    p_chat = sub.add_parser("chat", help="One-shot question")
    p_chat.add_argument("message")
    p_chat.set_defaults(cmd="chat")

    p_task = sub.add_parser("task", help="Multi-tool task (agent action loop)")
    p_task.add_argument("prompt")
    p_task.add_argument("--mode", choices=["auto", "bootstrap", "refactor", "review", "pml"], default="auto")
    p_task.add_argument("--max-calls", type=int, default=6)
    p_task.set_defaults(cmd="task")

    p_status = sub.add_parser("status", help="Show steps + last heartbeat")
    p_status.set_defaults(cmd="status")

    p_init = sub.add_parser("pml-init", help="Initialize pml/ + seed steps")
    p_init.set_defaults(cmd="pml-init")

    p_step = sub.add_parser("pml-step", help="Run a single step (by id or next pending)")
    p_step.add_argument("--id", dest="step_id", default="")
    p_step.set_defaults(cmd="pml-step")

    p_loop = sub.add_parser("pml-loop", help="Loop executing PML steps")
    p_loop.set_defaults(cmd="pml-loop")

    p_gen = sub.add_parser("gen-steps", help="Ask the model to generate more steps")
    p_gen.add_argument("--reason", default="manual")
    p_gen.set_defaults(cmd="gen-steps")

    return parser


def merge_namespaces(a: argparse.Namespace, b: argparse.Namespace) -> argparse.Namespace:
    out = argparse.Namespace()
    for k, v in vars(a).items():
        setattr(out, k, v)
    for k, v in vars(b).items():
        setattr(out, k, v)
    return out


def load_cfg(global_ns: argparse.Namespace) -> AppConfig:
    cfg = AppConfig()
    cfg.lm_url = str(global_ns.lm_url)
    cfg.lm_model = str(global_ns.lm_model)
    cfg.stream = bool(global_ns.stream)
    cfg.ssl_verify = bool(global_ns.ssl_verify)

    cfg.timeout_s = None if float(global_ns.timeout) == 0 else float(global_ns.timeout)

    cfg.ctx_len = int(global_ns.ctx_len)
    cfg.max_tokens = int(global_ns.max_tokens)
    cfg.temperature = float(global_ns.temperature)
    cfg.top_p = float(global_ns.top_p)

    cfg.agent_calls = int(global_ns.agent_calls)
    cfg.sleep_between = int(global_ns.sleep_between)
    cfg.loop_till_stopped = bool(global_ns.loop_till_stopped)

    cfg.monitor = bool(global_ns.monitor)
    cfg.max_gpu_temp_c = int(global_ns.max_gpu_temp)
    cfg.cooldown_s = int(global_ns.cooldown)

    cfg.auto_approve = bool(global_ns.auto_approve)
    cfg.safe_mode = bool(global_ns.safe_mode)

    cfg.git_auto = bool(global_ns.git_auto)
    cfg.git_push = bool(global_ns.git_push)
    cfg.git_interval_s = int(global_ns.git_interval)

    return cfg


def cmd_warmup(agent: Agent) -> None:
    log("Running warmup call…")
    answer, _ = agent.run_action_loop(
        prompt="Just reply with the single word: Ready.",
        mode="auto",
        max_calls=1,
        meta={"kind": "warmup"},
    )
    log(f"Warmup output: {answer!r}")


def cmd_chat(agent: Agent, message: str) -> None:
    log("Starting one-shot chat…")
    answer, _ = agent.run_action_loop(
        prompt=message,
        mode="auto",
        max_calls=1,
        meta={"kind": "chat"},
    )
    print("\n=== MODEL RESPONSE ===\n")
    print(answer)
    print("\n======================\n")


def cmd_task(agent: Agent, prompt: str, mode: str, max_calls: int) -> None:
    log(f"Starting task: mode={mode} max_calls={max_calls}")
    answer, _ = agent.run_action_loop(
        prompt=prompt,
        mode=mode,
        max_calls=max_calls,
        meta={"kind": "task", "mode": mode},
    )
    print("\n=== TASK RESULT ===\n")
    print(answer)
    print("\n===================\n")


def cmd_status(paths: ProjectPaths, db: ProjectDB) -> None:
    print("\n--- Steps (latest 30) ---")
    for s in db.list_steps(limit=30):
        print(f"{s['id']} [{s['phase']}] {s['status']} - {s['title']}")

    print("\n--- Heartbeat ---")
    if paths.heartbeat_path.exists():
        try:
            hb = json.loads(paths.heartbeat_path.read_text(encoding="utf-8"))
            gpu = hb.get("gpu")
            cpu = hb.get("cpu")
            print("cpu:", cpu)
            print("gpu:", gpu)
        except Exception:
            print(paths.heartbeat_path.read_text(encoding="utf-8")[:1000])
    else:
        print("(no heartbeat yet)")


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    base_dir = Path(__file__).resolve().parent
    paths = ProjectPaths(base_dir)
    paths.ensure()

    lock = SingleInstanceLock(paths.lock_path)
    lock.acquire()

    def _cleanup(*_: Any) -> None:
        lock.release()

    try:
        atexit = __import__("atexit")
        atexit.register(_cleanup)
    except Exception:
        pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: (_cleanup(), sys.exit(0)))
        except Exception:
            pass

    gp = build_global_parser()
    global_ns, remainder = gp.parse_known_args(argv)

    cp = build_cmd_parser()
    cmd_ns = cp.parse_args(remainder)

    ns = merge_namespaces(global_ns, cmd_ns)

    cfg = load_cfg(global_ns)

    log(f"[LM] Using LM Studio at {cfg.lm_url} model='{cfg.lm_model}'")

    db = ProjectDB(paths.db_path)
    ws = Workspace(paths)
    tools = ToolRunner(cfg, ws)
    lm = LMStudioClient(cfg)
    agent = Agent(cfg, lm, db, ws, tools)

    monitor = HealthMonitor(cfg, paths)
    monitor.start()

    steps = PMLSteps(paths, db)
    steps.init_files()
    git = GitBackup(cfg, tools)
    runner = PMLRunner(cfg, paths, db, agent, tools, steps, git)

    try:
        if getattr(ns, "pml_create", False):
            log("[CLI] --pml-create set; starting pml-loop")
            cfg.loop_till_stopped = True
            runner.loop()
            return

        if ns.cmd == "warmup":
            cmd_warmup(agent)
        elif ns.cmd == "chat":
            cmd_chat(agent, ns.message)
        elif ns.cmd == "task":
            cmd_task(agent, ns.prompt, ns.mode, int(ns.max_calls))
        elif ns.cmd == "status":
            cmd_status(paths, db)
        elif ns.cmd == "pml-init":
            log("[PML] Initialized.")
        elif ns.cmd == "pml-step":
            step = None
            if ns.step_id:
                step = {"id": ns.step_id, "phase": "pml", "title": ns.step_id, "body": "", "status": "todo"}
                full = steps._get_step_full(ns.step_id)
                if full:
                    step = full
            else:
                step = db.next_pending_step()
            if not step:
                log("[PML] No pending step.")
                return
            runner.run_one_step(step)
        elif ns.cmd == "pml-loop":
            runner.loop()
        elif ns.cmd == "gen-steps":
            runner._generate_more_steps(ns.reason)
        else:
            raise RuntimeError(f"Unknown cmd: {ns}")
    finally:
        try:
            db.close()
        except Exception:
            pass
        try:
            monitor.stop()
        except Exception:
            pass
        lock.release()


if __name__ == "__main__":
    main()
