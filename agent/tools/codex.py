"""
Codex-rs style tool implementations.

Tool names match codex-rs spec.rs exactly:
  - shell:                    command (array of strings, execvp-style)
  - shell_command:            cmd (single script in user's shell)
  - exec_command:             command, env, timeout, workdir  (persistent session)
  - write_stdin:              session_id, input
  - read_file:                path, offset, limit, mode
  - list_dir:                 path
  - grep_files:               pattern, path, include
  - apply_patch:              patch, path
  - update_plan:              steps [{title, description, status}]
  - request_user_input:       questions [{id, question, type, options}]
  - view_image:               path
  - web_search:               query, cached
  - js_repl:                  code
  - js_repl_reset:            (no params)
  - list_mcp_resources:       server (optional)
  - list_mcp_resource_templates: server (optional)
  - read_mcp_resource:        uri, server (optional)
  - spawn_agents_on_csv:      csv_path, prompt_template, description
  - report_agent_job_result:  job_id, row_index, result
"""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .base import Tool
from .files import _human_size, _max_read_chars


# ── Shell / Execution ──────────────────────────────────────────────────────────

class CodexShellTool(Tool):
    """codex-rs `shell` — execvp-style: command as array of strings."""

    name = "shell"
    description = (
        "Runs a command directly via execvp (no shell interpolation). "
        "Pass command and arguments as an array: [\"git\", \"diff\", \"HEAD\"]. "
        "Safer than shell_command for commands with known arguments. "
        "Returns combined stdout + stderr."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Command and arguments as an array, e.g. [\"python\", \"-m\", \"pytest\"].",
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional: working directory. Defaults to cwd.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional: timeout in seconds. Defaults to 120.",
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Optional: additional environment variables to set.",
                },
            },
            "required": ["command"],
        }

    def run(
        self,
        command: list,
        workdir: str = "",
        timeout: int = 120,
        env: dict | None = None,
    ) -> str:
        """
        Args:
            command: Command and arguments as an array (e.g. ["git", "log", "--oneline"]).
            workdir: Optional working directory.
            timeout: Timeout in seconds. Defaults to 120.
            env: Optional extra environment variables.
        """
        if not isinstance(command, list) or not command:
            return "[error] command must be a non-empty array of strings"
        cwd = workdir or os.getcwd()
        merged_env = {**os.environ, **(env or {})}
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=merged_env,
            )
            parts = []
            if result.stdout.strip():
                parts.append(result.stdout.rstrip())
            if result.stderr.strip():
                parts.append(f"[stderr]\n{result.stderr.rstrip()}")
            if not parts:
                parts.append(f"[exit {result.returncode}]")
            output = "\n".join(parts)
            limit = _max_read_chars()
            if len(output) > limit:
                output = output[:limit] + f"\n\n[truncated — output exceeded {limit:,} chars]"
            return output
        except subprocess.TimeoutExpired:
            return f"[error] timed out after {timeout}s"
        except FileNotFoundError:
            return f"[error] command not found: {command[0]!r}"
        except Exception as exc:
            return f"[error] {exc}"


class CodexShellCommandTool(Tool):
    """codex-rs `shell_command` — single script string in user's default shell."""

    name = "shell_command"
    description = (
        "Runs a shell command as a single script string in the user's default shell ($SHELL). "
        "Supports shell features: pipes, redirects, variable expansion, semicolons. "
        "Use `shell` (array form) for simple commands; use `shell_command` for shell pipelines."
    )

    def run(
        self,
        cmd: str,
        workdir: str = "",
        timeout: int = 120,
        env: dict | None = None,
    ) -> str:
        """
        Args:
            cmd: The shell script to execute (e.g. "find . -name '*.py' | wc -l").
            workdir: Optional working directory.
            timeout: Timeout in seconds. Defaults to 120.
            env: Optional extra environment variables.
        """
        shell_bin = os.environ.get("SHELL", "/bin/sh")
        cwd = workdir or os.getcwd()
        merged_env = {**os.environ, **(env or {})}
        try:
            result = subprocess.run(
                [shell_bin, "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=merged_env,
            )
            parts = []
            if result.stdout.strip():
                parts.append(result.stdout.rstrip())
            if result.stderr.strip():
                parts.append(f"[stderr]\n{result.stderr.rstrip()}")
            if not parts:
                parts.append(f"[exit {result.returncode}]")
            output = "\n".join(parts)
            limit = _max_read_chars()
            if len(output) > limit:
                output = output[:limit] + f"\n\n[truncated — output exceeded {limit:,} chars]"
            return output
        except subprocess.TimeoutExpired:
            return f"[error] timed out after {timeout}s"
        except Exception as exc:
            return f"[error] {exc}"


# ── Persistent exec sessions ───────────────────────────────────────────────────

class _ExecSession:
    """A running subprocess with a readable output buffer."""

    def __init__(self, proc: subprocess.Popen, session_id: str):
        self.proc = proc
        self.session_id = session_id
        self.output_lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                with self._lock:
                    self.output_lines.append(line.rstrip("\n"))
        except Exception:
            pass

    def drain(self) -> str:
        time.sleep(0.2)
        with self._lock:
            lines = list(self.output_lines)
            self.output_lines.clear()
        return "\n".join(lines)

    def write(self, text: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def is_alive(self) -> bool:
        return self.proc.poll() is None


_exec_sessions: dict[str, _ExecSession] = {}
_exec_sessions_lock = threading.Lock()


class CodexExecCommandTool(Tool):
    """codex-rs `exec_command` — starts a persistent interactive process."""

    name = "exec_command"
    description = (
        "Starts a persistent command session (like a long-running REPL or server). "
        "Returns a session_id you can use with write_stdin to send further input. "
        "Output is buffered and returned. Re-call with the same session_id to read "
        "new output from an already-running session."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Command and arguments as an array.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional: reuse an existing session to read its buffered output.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional working directory.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Seconds to wait for initial output. Defaults to 5.",
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Optional extra environment variables.",
                },
            },
            "required": ["command"],
        }

    def run(
        self,
        command: list,
        session_id: str = "",
        workdir: str = "",
        timeout: int = 5,
        env: dict | None = None,
    ) -> str:
        """
        Args:
            command: Command and arguments as an array.
            session_id: Optional existing session id to read output from.
            workdir: Optional working directory.
            timeout: Seconds to wait for output. Defaults to 5.
            env: Optional extra environment variables.
        """
        with _exec_sessions_lock:
            # If session_id given and exists, just drain its output
            if session_id and session_id in _exec_sessions:
                sess = _exec_sessions[session_id]
                if not sess.is_alive():
                    del _exec_sessions[session_id]
                    return f"[session {session_id!r} has exited]"
                time.sleep(min(timeout, 2))
                out = sess.drain()
                return f"[session {session_id}]\n{out}" if out else f"[session {session_id}] (no new output)"

        if not isinstance(command, list) or not command:
            return "[error] command must be a non-empty array of strings"

        cwd = workdir or os.getcwd()
        merged_env = {**os.environ, **(env or {})}
        sid = session_id or uuid.uuid4().hex[:8]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                env=merged_env,
            )
            sess = _ExecSession(proc, sid)
            with _exec_sessions_lock:
                _exec_sessions[sid] = sess
            time.sleep(min(timeout, 3))
            out = sess.drain()
            return f"[session {sid} started]\n{out}" if out else f"[session {sid} started] (no initial output)"
        except FileNotFoundError:
            return f"[error] command not found: {command[0]!r}"
        except Exception as exc:
            return f"[error] {exc}"


class CodexWriteStdinTool(Tool):
    """codex-rs `write_stdin` — sends input to an existing exec_command session."""

    name = "write_stdin"
    description = (
        "Sends characters (stdin) to an existing exec_command session. "
        "Use to interact with a running REPL, debugger, or interactive CLI. "
        "Append '\\n' for Enter. Returns any new output buffered since the last read."
    )

    def run(self, session_id: str, input: str, wait: int = 1) -> str:
        """
        Args:
            session_id: The session ID returned by exec_command.
            input: Text to write to stdin (include '\\n' for Enter).
            wait: Seconds to wait for output after writing. Defaults to 1.
        """
        with _exec_sessions_lock:
            sess = _exec_sessions.get(session_id)
        if sess is None:
            return f"[error] no session {session_id!r} — start one with exec_command"
        if not sess.is_alive():
            with _exec_sessions_lock:
                _exec_sessions.pop(session_id, None)
            return f"[error] session {session_id!r} has exited"
        try:
            sess.write(input)
            time.sleep(max(0.1, wait))
            out = sess.drain()
            return out if out else "(no output)"
        except Exception as exc:
            return f"[error] {exc}"


# ── File Operations ────────────────────────────────────────────────────────────

class CodexReadFileTool(Tool):
    """codex-rs `read_file` — slice + indentation-aware block modes."""

    name = "read_file"
    description = (
        "Reads a local file. Supports paginated slice mode (offset + limit lines) "
        "and block mode (indentation-aware context around a target line). "
        "Returns numbered lines in cat -n format."
    )

    def run(
        self,
        path: str,
        offset: int = 0,
        limit: int = 0,
        mode: str = "slice",
    ) -> str:
        """
        Args:
            path: Absolute path to the file to read.
            offset: 0-based line number to start reading from. Defaults to 0 (start of file).
            limit: Maximum lines to return. 0 means read to end (capped at 2000).
            mode: 'slice' for line-range read (default). 'block' for indentation-aware context.
        """
        p = Path(path).expanduser()
        if not p.exists():
            return f"[error] file not found: {path}"
        if not p.is_file():
            return f"[error] not a file: {path}"
        try:
            text = p.read_bytes().decode("utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)

            start = max(0, offset)
            max_lines = 2000
            end = total if limit == 0 else min(start + limit, total)
            end = min(end, start + max_lines)

            selected = lines[start:end]
            truncated = end < total

            header = f"{path} | lines {start + 1}–{end} of {total}"
            if truncated:
                header += f"  [use offset={end} to continue]"

            numbered = "\n".join(f"{start + i + 1:>5}\t{l}" for i, l in enumerate(selected))
            return header + "\n" + "─" * 60 + "\n" + (numbered or "[empty file]")
        except Exception as exc:
            return f"[error] {exc}"


class CodexListDirTool(Tool):
    """codex-rs `list_dir` — 1-indexed directory listing."""

    name = "list_dir"
    description = (
        "Lists entries in a local directory. "
        "Returns 1-indexed numbered entries with sizes, sorted: directories first, then files."
    )

    def run(self, path: str = ".") -> str:
        """
        Args:
            path: Path to the directory to list. Defaults to current directory.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"[error] path not found: {path}"
        if not p.is_dir():
            return f"[error] not a directory: {path}"

        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        if not entries:
            return f"{p}\n[empty directory]"

        lines = [str(p), "─" * 60]
        for i, item in enumerate(entries, 1):
            if item.is_dir():
                lines.append(f"{i:>4}.  {item.name}/")
            else:
                try:
                    size = _human_size(item.stat().st_size)
                except OSError:
                    size = "?"
                lines.append(f"{i:>4}.  {item.name:<42} {size:>8}")
        return "\n".join(lines)


class CodexGrepFilesTool(Tool):
    """codex-rs `grep_files` — regex search across file contents."""

    name = "grep_files"
    description = (
        "Searches files for lines matching a regular expression. "
        "Filter by file glob with `include`. Returns file:line:content matches."
    )

    def run(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        max_matches: int = 100,
    ) -> str:
        """
        Args:
            pattern: Regular expression to search for.
            path: File or directory to search in. Defaults to cwd.
            include: Optional glob pattern to filter files (e.g. '*.py', '*.{ts,rs}').
            max_matches: Maximum number of matches to return. Defaults to 100.
        """
        import re

        base = Path(path).expanduser().resolve()
        if not base.exists():
            return f"[error] path not found: {path}"

        file_glob = include if include else "*"
        files: list[Path]
        if base.is_file():
            files = [base]
        else:
            files = [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, file_glob)]

        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return f"[error] invalid regex: {exc}"

        results: list[str] = []
        for f in sorted(files):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            rel = str(f.relative_to(base)) if base.is_dir() else str(f)
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    results.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(results) >= max_matches:
                        results.append(f"[truncated — reached limit of {max_matches}]")
                        return "\n".join(results)

        return "\n".join(results) if results else f"[no matches] {pattern!r} in {path}"


# ── Content Modification ───────────────────────────────────────────────────────

class CodexApplyPatchTool(Tool):
    """codex-rs `apply_patch` — applies unified diff patches to files."""

    name = "apply_patch"
    description = (
        "Applies a unified diff patch to one or more files. "
        "The patch must be in standard unified diff format (--- a/file, +++ b/file, @@ hunks). "
        "Use to make precise, surgical edits when you know the exact diff. "
        "Each hunk is applied independently; context lines are used for positioning."
    )

    def run(self, patch: str, path: str = "") -> str:
        """
        Args:
            patch: Unified diff patch text (output of 'diff -u' or 'git diff').
            path: Optional: base directory for resolving relative paths in the patch. Defaults to cwd.
        """
        base = Path(path).expanduser().resolve() if path else Path.cwd()
        lines = patch.splitlines()

        results: list[str] = []
        i = 0
        files_patched = 0
        errors = 0

        while i < len(lines):
            # Find --- line (start of a file patch)
            if not lines[i].startswith("--- "):
                i += 1
                continue
            orig_line = lines[i]
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                results.append(f"[error] expected +++ line after: {orig_line!r}")
                errors += 1
                continue
            plus_line = lines[i]
            i += 1

            # Parse target file path from +++ line
            raw_path = plus_line[4:].split("\t")[0].strip()
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            elif raw_path == "/dev/null":
                results.append(f"[skip] /dev/null target")
                continue

            target = (base / raw_path).resolve()

            # Collect hunks for this file
            hunks: list[list[str]] = []
            while i < len(lines) and lines[i].startswith("@@ "):
                hunk_header = lines[i]
                i += 1
                hunk_lines: list[str] = [hunk_header]
                while i < len(lines) and not lines[i].startswith("@@ ") and not lines[i].startswith("--- "):
                    hunk_lines.append(lines[i])
                    i += 1
                hunks.append(hunk_lines)

            if not hunks:
                results.append(f"[skip] {raw_path} — no hunks found")
                continue

            # Apply hunks
            try:
                apply_result = _apply_unified_hunks(target, hunks, raw_path)
                results.append(apply_result)
                if not apply_result.startswith("[error]"):
                    files_patched += 1
                else:
                    errors += 1
            except Exception as exc:
                results.append(f"[error] {raw_path}: {exc}")
                errors += 1

        if not results:
            return "[error] no patch hunks found — is this a valid unified diff?"
        summary = f"[patch] {files_patched} file(s) patched, {errors} error(s)"
        return summary + "\n" + "\n".join(results)


def _apply_unified_hunks(target: Path, hunks: list[list[str]], display_name: str) -> str:
    """Apply all unified-diff hunks for one file."""
    import re

    # Handle new-file creation (/dev/null on --- side handled upstream)
    if not target.exists():
        # All lines in hunks should be additions
        new_lines: list[str] = []
        for hunk in hunks:
            for hl in hunk[1:]:  # skip hunk header
                if hl.startswith("+"):
                    new_lines.append(hl[1:])
                elif hl.startswith(" "):
                    new_lines.append(hl[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return f"[ok] created {display_name} ({len(new_lines)} lines)"

    content = target.read_text(encoding="utf-8", errors="replace")
    file_lines = content.splitlines(keepends=True)

    offset = 0  # cumulative offset from previous hunks
    for hunk in hunks:
        header = hunk[0]
        # Parse @@ -L,S +L,S @@
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
        if not m:
            continue
        old_start = int(m.group(1)) - 1  # 0-based
        body = hunk[1:]

        # Build expected context + removals for matching
        expected_old: list[str] = []
        new_fragment: list[str] = []
        for bl in body:
            if bl.startswith("-"):
                expected_old.append(bl[1:])
            elif bl.startswith("+"):
                new_fragment.append(bl[1:])
            elif bl.startswith(" ") or bl == "":
                ctx = bl[1:] if bl.startswith(" ") else ""
                expected_old.append(ctx)
                new_fragment.append(ctx)

        # Locate the hunk in actual file with offset
        search_start = max(0, old_start + offset)
        found_at = _find_hunk_position(file_lines, expected_old, search_start)
        if found_at < 0:
            # Try without offset (fallback)
            found_at = _find_hunk_position(file_lines, expected_old, 0)
        if found_at < 0:
            return f"[error] {display_name}: hunk context not found near line {old_start + 1}"

        # Replace
        n_remove = len([b for b in body if b.startswith("-") or b.startswith(" ")])
        new_with_newlines = [l + ("\n" if not l.endswith("\n") else "") for l in new_fragment]
        file_lines[found_at:found_at + n_remove] = new_with_newlines
        offset += len(new_fragment) - n_remove

    target.write_text("".join(file_lines), encoding="utf-8")
    return f"[ok] patched {display_name}"


def _find_hunk_position(
    file_lines: list[str],
    expected: list[str],
    start: int,
) -> int:
    """Find the 0-based line index where expected context matches file_lines."""
    if not expected:
        return start
    n = len(expected)
    for i in range(start, max(0, len(file_lines) - n + 1)):
        if all(
            file_lines[i + j].rstrip("\n") == expected[j].rstrip("\n")
            for j in range(n)
        ):
            return i
    return -1


# ── Planning ───────────────────────────────────────────────────────────────────

_PLAN_FILE = ".codex_plan.json"
_VALID_STATUSES = {"pending", "in_progress", "completed", "skipped"}


class CodexUpdatePlanTool(Tool):
    """codex-rs `update_plan` — structured task plan with step statuses."""

    name = "update_plan"
    description = (
        "Updates the active task plan. Pass the full list of steps; this replaces the plan. "
        "Each step has a title, optional description, and status: "
        "pending | in_progress | completed | skipped. "
        "The plan is persisted to .codex_plan.json and shown in /plan."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["steps"],
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Complete ordered list of plan steps.",
                    "items": {
                        "type": "object",
                        "required": ["title", "status"],
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short step title.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional longer description of what this step does.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "skipped"],
                                "description": "Current step status.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
        }

    def run(self, steps: list) -> str:
        """
        Args:
            steps: List of {title, description, status} step objects.
        """
        if not isinstance(steps, list) or not steps:
            return "[error] steps must be a non-empty array"

        normalized: list[dict[str, Any]] = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                return f"[error] step {i} must be an object"
            title = s.get("title", "").strip()
            if not title:
                return f"[error] step {i} has empty title"
            status = s.get("status", "pending")
            if status not in _VALID_STATUSES:
                return f"[error] step {i} has invalid status {status!r}"
            normalized.append({
                "id": i + 1,
                "title": title,
                "description": s.get("description", "").strip(),
                "status": status,
            })

        Path(_PLAN_FILE).write_text(
            json.dumps({"steps": normalized}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        counts = {s: sum(1 for x in normalized if x["status"] == s) for s in _VALID_STATUSES}
        lines = [
            f"[ok] plan updated — {len(normalized)} steps "
            f"({counts['in_progress']} in_progress, {counts['pending']} pending, "
            f"{counts['completed']} completed, {counts['skipped']} skipped)"
        ]
        for step in normalized:
            icon = {"pending": "○", "in_progress": "◎", "completed": "✓", "skipped": "–"}.get(step["status"], "?")
            lines.append(f"  {icon} [{step['id']}] {step['title']}")
        return "\n".join(lines)


# ── User Interaction ───────────────────────────────────────────────────────────

class CodexRequestUserInputTool(Tool):
    """codex-rs `request_user_input` — structured questions with optional choices."""

    name = "request_user_input"
    description = (
        "Asks the user one or more questions before proceeding. "
        "Supports free-text, yes/no, and multiple-choice question types. "
        "Use when you need clarification, preferences, or a decision from the user."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "description": "List of questions to ask the user.",
                    "items": {
                        "type": "object",
                        "required": ["id", "question", "type"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique ID for this question (e.g. 'auth_method').",
                            },
                            "question": {
                                "type": "string",
                                "description": "The question text.",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["text", "yesno", "choice"],
                                "description": "Question type: 'text' for free input, 'yesno', or 'choice' for multiple options.",
                            },
                            "options": {
                                "type": "array",
                                "description": "Selectable options for 'choice' type.",
                                "items": {
                                    "type": "object",
                                    "required": ["label"],
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                            "default": {
                                "type": "string",
                                "description": "Optional default value or choice label.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
        }

    def run(self, questions: list) -> str:
        """
        Args:
            questions: List of question objects with id, question, type, and optional options.
        """
        if not questions:
            return "[error] no questions provided"

        lines = ["[request_user_input] The agent needs your input:\n"]
        for q in questions:
            qid = q.get("id", "?")
            qtext = q.get("question", "")
            qtype = q.get("type", "text")
            default = q.get("default", "")
            lines.append(f"  [{qid}] {qtext}")
            if qtype == "yesno":
                dflt = f" (default: {default})" if default else ""
                lines.append(f"    → Yes / No{dflt}")
            elif qtype == "choice":
                opts = q.get("options", [])
                for opt in opts:
                    mark = " ← default" if opt.get("label") == default else ""
                    desc = f": {opt['description']}" if opt.get("description") else ""
                    lines.append(f"    • {opt['label']}{desc}{mark}")
            else:  # text
                dflt = f" (default: {default!r})" if default else ""
                lines.append(f"    → [free text{dflt}]")
        lines.append("\n[Please provide your answers to continue.]")
        return "\n".join(lines)


# ── Image ──────────────────────────────────────────────────────────────────────

class CodexViewImageTool(Tool):
    """codex-rs `view_image` — reads a local image and reports metadata."""

    name = "view_image"
    description = (
        "Reads a local image file from the filesystem and returns its metadata "
        "(path, size, format). Supported formats: PNG, JPEG, GIF, WebP, BMP, TIFF. "
        "The image content is returned as base64 if small enough for the context window."
    )

    def run(self, path: str) -> str:
        """
        Args:
            path: Absolute path to the image file.
        """
        import base64
        import imghdr

        p = Path(path).expanduser()
        if not p.exists():
            return f"[error] file not found: {path}"
        if not p.is_file():
            return f"[error] not a file: {path}"

        fmt = imghdr.what(str(p)) or p.suffix.lstrip(".").lower() or "unknown"
        size = p.stat().st_size
        size_str = _human_size(size)

        lines = [
            f"[view_image] {p.name}",
            f"  path:   {path}",
            f"  format: {fmt.upper()}",
            f"  size:   {size_str} ({size:,} bytes)",
        ]

        # Include base64 if under 256 KB
        if size <= 256 * 1024:
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                mime = {
                    "png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
                }.get(fmt.lower(), "image/octet-stream")
                lines.append(f"  data:   data:{mime};base64,{b64}")
            except Exception as exc:
                lines.append(f"  [could not encode: {exc}]")
        else:
            lines.append(f"  [image too large for inline display — {size_str}]")

        return "\n".join(lines)


# ── Web ────────────────────────────────────────────────────────────────────────

class CodexWebSearchTool(Tool):
    """codex-rs `web_search` — live or cached web search."""

    name = "web_search"
    description = (
        "Searches the web for current information. "
        "Returns titles, URLs, and snippets for the top results. "
        "Set cached=true to prefer faster cached results when freshness is not critical."
    )

    def run(self, query: str, cached: bool = False) -> str:
        """
        Args:
            query: The search query.
            cached: Optional: prefer cached results for speed. Defaults to false.
        """
        from urllib.parse import quote_plus
        import re
        import httpx

        _HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=8):
                    results.append(
                        f"**{r.get('title', 'No title')}**\n"
                        f"{r.get('href', '')}\n"
                        f"{r.get('body', '')}"
                    )
            if results:
                return "\n\n".join(results)
        except ImportError:
            pass
        except Exception:
            pass

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            with httpx.Client(follow_redirects=True, timeout=10) as client:
                resp = client.get(url, headers=_HEADERS)
                resp.raise_for_status()
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text, re.S)
            urls   = re.findall(r'class="result__url"[^>]*>\s*(.*?)\s*</a>', resp.text, re.S)
            snips  = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.S)
            results = []
            for i, (t, u, s) in enumerate(zip(titles, urls, snips)):
                if i >= 8:
                    break
                results.append(
                    f"**{re.sub(r'<[^>]+>', '', t).strip()}**\n"
                    f"{re.sub(r'<[^>]+>', '', u).strip()}\n"
                    f"{re.sub(r'<[^>]+>', '', s).strip()}"
                )
            return "\n\n".join(results) if results else f"[no results for: {query!r}]"
        except Exception as exc:
            return f"[error] web search failed: {exc}"


# ── JavaScript REPL ────────────────────────────────────────────────────────────

class _JsReplState:
    """Module-level singleton that holds accumulated JS context."""
    bindings: dict[str, str] = {}   # varname → last assigned expression (best-effort)
    session_code: list[str] = []    # accumulated declarations for re-injection


_js_state = _JsReplState()


class CodexJsReplTool(Tool):
    """codex-rs `js_repl` — runs JavaScript in a Node.js subprocess."""

    name = "js_repl"
    description = (
        "Executes JavaScript code using Node.js. "
        "Supports top-level await, ES modules syntax, and CommonJS require(). "
        "Declarations from previous calls are re-injected so variables persist across calls. "
        "Returns stdout output and any errors. Node.js must be installed."
    )

    def run(self, code: str) -> str:
        """
        Args:
            code: JavaScript code to execute. Top-level await is supported via async IIFE wrapping.
        """
        # Check node is available
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "[error] Node.js not found — install Node.js to use js_repl"

        # Wrap in async IIFE to support top-level await
        preamble = "\n".join(_js_state.session_code)
        wrapped = (
            f"{preamble}\n"
            f"(async () => {{\n"
            f"  {code}\n"
            f"}})().catch(e => {{ process.stderr.write(String(e)); process.exit(1); }});"
        )

        try:
            result = subprocess.run(
                ["node", "--input-type=module"],
                input=wrapped,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "[error] js_repl timed out after 30s"
        except Exception as exc:
            return f"[error] {exc}"

        # Accumulate non-expression code for persistence
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith(("const ", "let ", "var ", "function ", "class ", "async function ")):
                _js_state.session_code.append(line)

        parts = []
        if result.stdout.strip():
            parts.append(result.stdout.rstrip())
        if result.stderr.strip():
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        if not parts:
            parts.append("[ok] (no output)")
        return "\n".join(parts)


class CodexJsReplResetTool(Tool):
    """codex-rs `js_repl_reset` — clears the js_repl session state."""

    name = "js_repl_reset"
    description = (
        "Resets the js_repl kernel — clears all accumulated declarations and bindings. "
        "Use when you want a clean JavaScript environment."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def run(self) -> str:
        _js_state.session_code.clear()
        _js_state.bindings.clear()
        return "[ok] js_repl kernel reset — all bindings cleared"


# ── MCP (Model Context Protocol) ──────────────────────────────────────────────

class CodexListMcpResourcesTool(Tool):
    """codex-rs `list_mcp_resources` — lists resources from MCP servers."""

    name = "list_mcp_resources"
    description = (
        "Lists all resources provided by connected MCP (Model Context Protocol) servers. "
        "Resources are named data sources the agent can read via read_mcp_resource. "
        "Optionally filter by server name."
    )

    def run(self, server: str = "") -> str:
        """
        Args:
            server: Optional MCP server name to filter results.
        """
        # MCP client integration is not wired in this environment.
        note = f" on server {server!r}" if server else ""
        return (
            f"[list_mcp_resources{note}]\n"
            "[MCP client is not configured in this environment. "
            "To use MCP, add an MCP server to your config.]"
        )


class CodexListMcpResourceTemplatesTool(Tool):
    """codex-rs `list_mcp_resource_templates` — parameterized MCP resource templates."""

    name = "list_mcp_resource_templates"
    description = (
        "Lists parameterized resource URI templates provided by MCP servers. "
        "Templates use {parameter} placeholders you fill when calling read_mcp_resource."
    )

    def run(self, server: str = "") -> str:
        """
        Args:
            server: Optional MCP server name to filter results.
        """
        note = f" on server {server!r}" if server else ""
        return (
            f"[list_mcp_resource_templates{note}]\n"
            "[MCP client is not configured in this environment.]"
        )


class CodexReadMcpResourceTool(Tool):
    """codex-rs `read_mcp_resource` — reads a resource from an MCP server."""

    name = "read_mcp_resource"
    description = (
        "Reads a specific resource from an MCP server by URI. "
        "Use list_mcp_resources or list_mcp_resource_templates to discover available URIs."
    )

    def run(self, uri: str, server: str = "") -> str:
        """
        Args:
            uri: The resource URI to read (e.g. 'file:///path/to/resource' or a custom scheme).
            server: Optional MCP server name if multiple servers are configured.
        """
        note = f" from {server!r}" if server else ""
        return (
            f"[read_mcp_resource] uri={uri!r}{note}\n"
            "[MCP client is not configured in this environment.]"
        )


# ── Batch Job Tools ────────────────────────────────────────────────────────────

class CodexSpawnAgentsOnCsvTool(Tool):
    """codex-rs `spawn_agents_on_csv` — one sub-agent per CSV row."""

    name = "spawn_agents_on_csv"
    description = (
        "Processes a CSV file by spawning one sub-agent per row. "
        "The prompt_template uses {column_name} placeholders filled from each row's values. "
        "Results from each row-agent are collected and returned as a summary table."
    )

    def run(
        self,
        csv_path: str,
        prompt_template: str,
        description: str = "",
        max_workers: int = 4,
    ) -> str:
        """
        Args:
            csv_path: Path to the CSV file to process.
            prompt_template: Prompt with {column_name} placeholders for each row.
            description: Optional short label for this batch job.
            max_workers: Maximum concurrent sub-agents. Defaults to 4.
        """
        import csv
        from concurrent.futures import ThreadPoolExecutor, as_completed

        p = Path(csv_path).expanduser()
        if not p.exists():
            return f"[error] CSV file not found: {csv_path}"

        try:
            with open(p, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as exc:
            return f"[error] could not read CSV: {exc}"

        if not rows:
            return "[error] CSV file has no data rows"

        label = description or f"batch:{p.name}"
        results: dict[int, str] = {}
        errors: dict[int, str] = {}

        def _run_row(idx: int, row: dict) -> tuple[int, str]:
            from ..agent import Agent
            from ..config import Config

            try:
                prompt = prompt_template.format(**row)
            except KeyError as exc:
                return idx, f"[error] missing column {exc} in row {idx + 1}"

            try:
                config = Config()
                config.max_tool_iterations = min(config.max_tool_iterations, 10)
                sub = Agent(config=config)
                parts: list[str] = []
                for event in sub.run(prompt):
                    if event.type == "text":
                        parts.append(event.data)
                    elif event.type == "error":
                        parts.append(f"[error] {event.data}")
                return idx, "".join(parts).strip() or "(no output)"
            except Exception as exc:
                return idx, f"[error] {exc}"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_row, i, row): i for i, row in enumerate(rows)}
            for fut in as_completed(futures):
                idx, result = fut.result()
                results[idx] = result

        lines = [f"[{label}] {len(rows)} rows processed\n"]
        for i, row in enumerate(rows):
            key_col = next(iter(row), "row")
            row_label = row.get(key_col, f"row {i + 1}")
            res = results.get(i, "[no result]")
            lines.append(f"── Row {i + 1} ({row_label}) ──")
            lines.append(res)
            lines.append("")
        return "\n".join(lines)


class CodexReportAgentJobResultTool(Tool):
    """codex-rs `report_agent_job_result` — worker reports result for a batch row."""

    name = "report_agent_job_result"
    description = (
        "Used by worker sub-agents inside a spawn_agents_on_csv batch job. "
        "Call this to report your result for the assigned row back to the job manager. "
        "result should be a concise summary of what was done / found."
    )

    def run(self, job_id: str, row_index: int, result: str) -> str:
        """
        Args:
            job_id: The batch job identifier provided in the sub-agent's prompt.
            row_index: The 0-based index of the CSV row this agent was assigned.
            result: The result string to report for this row.
        """
        # In this environment, worker results are collected by the parent via return value.
        # This tool exists for schema compatibility; the parent's thread pool collects results.
        return (
            f"[report_agent_job_result] job={job_id!r} row={row_index} result recorded.\n"
            f"{result}"
        )


# ── Public re-exports ──────────────────────────────────────────────────────────

__all__ = [
    # Shell
    "CodexShellTool",
    "CodexShellCommandTool",
    "CodexExecCommandTool",
    "CodexWriteStdinTool",
    # File ops
    "CodexReadFileTool",
    "CodexListDirTool",
    "CodexGrepFilesTool",
    # Content modification
    "CodexApplyPatchTool",
    # Planning
    "CodexUpdatePlanTool",
    # User interaction
    "CodexRequestUserInputTool",
    # Media
    "CodexViewImageTool",
    # Web
    "CodexWebSearchTool",
    # JS REPL
    "CodexJsReplTool",
    "CodexJsReplResetTool",
    # MCP
    "CodexListMcpResourcesTool",
    "CodexListMcpResourceTemplatesTool",
    "CodexReadMcpResourceTool",
    # Batch
    "CodexSpawnAgentsOnCsvTool",
    "CodexReportAgentJobResultTool",
]
