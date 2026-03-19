from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .base import Tool, ToolRegistry
from .claude import (
    BashTool, EditTool, LSTool, ReadTool, WriteTool,
    GlobTool as _ClaudeGlob,
    GrepTool as _ClaudeGrep,
    MultiEditTool as _ClaudeMultiEdit,
    NotebookEditTool, NotebookReadTool,
    ExitPlanModeTool, TodoReadTool, TodoWriteTool, TaskTool,
    WebFetchTool, WebSearchTool,
)
from .files import GlobTool, GrepTool, ListDirTool, MultiEditTool, ReadFileTool, WriteFileTool, ReadManyFilesTool
from .gemini import (
    GeminiActivateSkillTool,
    GeminiAskUserTool,
    GeminiEnterPlanModeTool,
    GeminiExitPlanModeTool,
    GeminiGetInternalDocsTool,
    GeminiGlobTool,
    GeminiGrepTool,
    GeminiListDirTool,
    GeminiReadFileTool,
    GeminiReadManyFilesTool,
    GeminiReplaceTool,
    GeminiSaveMemoryTool,
    GeminiShellTool,
    GeminiWebFetchTool,
    GeminiWebSearchTool,
    GeminiWriteTodosTool,
)
from .plan import WritePlanTool
from .qwen import (
    QwenEditTool,
    QwenGlobTool,
    QwenGrepTool,
    QwenListDirTool,
    QwenLspTool,
    QwenReadFileTool,
    QwenSaveMemoryTool,
    QwenShellTool,
    QwenSkillTool,
    QwenTaskTool,
    QwenTodoWriteTool,
    QwenWebFetchTool,
    QwenWebSearchTool,
)
from .codex import (
    CodexApplyPatchTool,
    CodexExecCommandTool,
    CodexGrepFilesTool,
    CodexJsReplResetTool,
    CodexJsReplTool,
    CodexListDirTool,
    CodexListMcpResourceTemplatesTool,
    CodexListMcpResourcesTool,
    CodexReadFileTool,
    CodexReadMcpResourceTool,
    CodexReportAgentJobResultTool,
    CodexRequestUserInputTool,
    CodexShellCommandTool,
    CodexShellTool,
    CodexSpawnAgentsOnCsvTool,
    CodexUpdatePlanTool,
    CodexViewImageTool,
    CodexWebSearchTool,
    CodexWriteStdinTool,
)
from .opencode import (
    OpencodeApplyPatchTool,
    OpencodeBashTool,
    OpencodeBatchTool,
    OpencodeCodeSearchTool,
    OpencodeEditTool,
    OpencodeGlobTool,
    OpencodeGrepTool,
    OpencodeListTool,
    OpencodeLspTool,
    OpencodeMultiEditTool,
    OpencodePlanExitTool,
    OpencodeQuestionTool,
    OpencodeReadTool,
    OpencodeSkillTool,
    OpencodeTaskTool,
    OpencodeTodoReadTool,
    OpencodeTodoWriteTool,
    OpencodeWebFetchTool,
    OpencodeWebSearchTool,
    OpencodeWriteTool,
)
from .multi_agents import (
    CloseAgentTool,
    ListAgentsTool,
    ResumeAgentTool,
    SendInputTool,
    SpawnAgentTool,
    WaitTool,
)
from .shell import OpenTerminalTool, ShellTool


@dataclass
class ToolProfile:
    """A named, reusable collection of tools."""
    name: str
    description: str
    # Factories instead of instances so each build_registry() gets fresh objects
    _factories: list[Callable[[], Tool]] = field(default_factory=list, repr=False)

    def build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(*[f() for f in self._factories])
        return registry

    def tool_names(self) -> list[str]:
        return [f().name for f in self._factories]


# ── Profile definitions ────────────────────────────────────────────────────────

_PROFILES: dict[str, ToolProfile] = {

    # All built-in tools (default)
    "default": ToolProfile(
        name="default",
        description="All built-in tools: shell, file read/write, read_many_files, multi_edit, glob, grep, list_dir",
        _factories=[
            ShellTool, ReadFileTool, WriteFileTool, ReadManyFilesTool,
            MultiEditTool, GlobTool, GrepTool, ListDirTool,
        ],
    ),

    # Claude Code style: full Claude Code tool suite
    "claude": ToolProfile(
        name="claude",
        description=(
            "Claude Code style — full tool suite matching Claude Code: "
            "bash, glob, grep, ls, read, edit, multi_edit, write, "
            "notebook_read, notebook_edit, web_fetch, web_search, "
            "exit_plan_mode, todo_read, todo_write, task."
        ),
        _factories=[
            BashTool, _ClaudeGlob, _ClaudeGrep, LSTool,
            ReadTool, EditTool, _ClaudeMultiEdit, WriteTool,
            NotebookReadTool, NotebookEditTool,
            WebFetchTool, WebSearchTool,
            ExitPlanModeTool,
            TodoReadTool, TodoWriteTool,
            TaskTool,
            OpenTerminalTool,
            # Multi-agent tools
            SpawnAgentTool, SendInputTool, WaitTool,
            CloseAgentTool, ResumeAgentTool, ListAgentsTool,
        ],
    ),

    # Gemini CLI style: matches the actual gemini-cli DEFAULT_LEGACY_SET / GEMINI_3_SET
    # read_file(file_path, start_line, end_line)
    # replace(file_path, instruction, old_string, new_string, allow_multiple)
    # glob(pattern, dir_path, case_sensitive)
    # grep_search(pattern, dir_path, include, names_only, total_max_matches)
    # list_directory(dir_path, ignore)
    # run_shell_command(command, description)
    # (save_memory, read_many_files, get_internal_docs, ask_user, activate_skill omitted)
    "gemini": ToolProfile(
        name="gemini",
        description=(
            "Gemini CLI style — full tool suite matching gemini-cli exactly. "
            "replace(file_path, instruction, old_string, new_string, allow_multiple), "
            "read_file(file_path, start_line, end_line), read_many_files(include, exclude), "
            "grep_search(pattern, dir_path, include, names_only, total_max_matches), "
            "google_web_search(query), web_fetch(prompt), write_todos(todos), "
            "save_memory(fact), ask_user(questions), enter_plan_mode(reason), exit_plan_mode."
        ),
        _factories=[
            GeminiShellTool, GeminiReadFileTool, WriteFileTool,
            GeminiGlobTool, GeminiGrepTool, GeminiListDirTool,
            GeminiReplaceTool,
            GeminiReadManyFilesTool,
            GeminiWebSearchTool, GeminiWebFetchTool,
            GeminiWriteTodosTool,
            GeminiSaveMemoryTool,
            GeminiGetInternalDocsTool,
            GeminiAskUserTool,
            GeminiEnterPlanModeTool,
            GeminiExitPlanModeTool,
            GeminiActivateSkillTool,
        ],
    ),

    # GPT / OpenAI style: conservative, avoids raw shell
    "gpt": ToolProfile(
        name="gpt",
        description=(
            "GPT style — conservative set. "
            "Prefers structured file tools over raw shell; no write_plan."
        ),
        _factories=[
            ReadFileTool, WriteFileTool, ReadManyFilesTool, MultiEditTool,
            GlobTool, GrepTool, ListDirTool, ShellTool,
        ],
    ),

    # Read-only: safe for exploration, no writes or shell
    "readonly": ToolProfile(
        name="readonly",
        description="Read-only — no shell, no writes. Safe for untrusted models or exploration.",
        _factories=[ReadFileTool, GlobTool, GrepTool, ListDirTool],
    ),

    # Minimal: shell + read only
    "minimal": ToolProfile(
        name="minimal",
        description="Minimal — shell and read_file only.",
        _factories=[ShellTool, ReadFileTool],
    ),

    # Qwen Coder: matches the actual qwen-code tool set
    # read_file(absolute_path, offset, limit)
    # edit(file_path, old_string, new_string, replace_all)
    # glob(pattern, path)
    # grep_search(pattern, path, glob, limit)
    # list_directory(dir_path)
    # run_shell_command(command, is_background, timeout, description, directory)
    # (lsp and skill have no Python equivalents and are omitted)
    "qwen": ToolProfile(
        name="qwen",
        description=(
            "Qwen Coder style — full tool suite matching qwen-code exactly. "
            "read_file(absolute_path, offset, limit), "
            "edit(file_path, old_string, new_string, replace_all), "
            "run_shell_command(command, is_background, timeout, description, directory), "
            "web_fetch(url, prompt), web_search(query), todo_write(todos), "
            "save_memory(fact, scope), task(description, prompt, subagent_type), "
            "skill(skill), lsp(operation, ...)."
        ),
        _factories=[
            QwenReadFileTool, WriteFileTool,
            QwenEditTool,
            QwenGlobTool, QwenGrepTool, QwenListDirTool,
            QwenShellTool,
            QwenWebFetchTool, QwenWebSearchTool,
            QwenTodoWriteTool,
            QwenSaveMemoryTool,
            QwenTaskTool,
            QwenSkillTool,
            QwenLspTool,
            ExitPlanModeTool,
        ],
    ),

    # Codex-rs: matches the codex-rs tool set from codex/codex-rs
    # shell([cmd, args...]), shell_command(cmd), exec_command(command, ...), write_stdin(session_id, input)
    # read_file(path, offset, limit, mode), list_dir(path), grep_files(pattern, path, include)
    # apply_patch(patch), update_plan(steps), request_user_input(questions), view_image(path)
    # web_search(query, cached), js_repl(code), js_repl_reset()
    # list_mcp_resources, list_mcp_resource_templates, read_mcp_resource(uri)
    # spawn_agents_on_csv(csv_path, prompt_template), report_agent_job_result(job_id, row_index, result)
    # + multi-agent: spawn_agent, send_input, wait, close_agent, resume_agent
    "codex": ToolProfile(
        name="codex",
        description=(
            "Codex-rs style — full tool suite matching codex-rs spec.rs. "
            "shell([cmd,args]), shell_command(cmd), exec_command(command), write_stdin(session_id, input), "
            "read_file(path, offset, limit, mode), list_dir(path), grep_files(pattern, path, include), "
            "apply_patch(patch), update_plan(steps), request_user_input(questions), view_image(path), "
            "web_search(query, cached), js_repl(code), js_repl_reset(), "
            "list_mcp_resources, list_mcp_resource_templates, read_mcp_resource(uri), "
            "spawn_agents_on_csv(csv_path, prompt_template), report_agent_job_result(job_id, row_index, result)."
        ),
        _factories=[
            # Shell / exec
            CodexShellTool, CodexShellCommandTool,
            CodexExecCommandTool, CodexWriteStdinTool,
            # File ops
            CodexReadFileTool, WriteFileTool, CodexListDirTool, CodexGrepFilesTool,
            # Content modification
            CodexApplyPatchTool,
            # Planning & interaction
            CodexUpdatePlanTool, CodexRequestUserInputTool,
            # Media
            CodexViewImageTool,
            # Web
            CodexWebSearchTool,
            # JS REPL
            CodexJsReplTool, CodexJsReplResetTool,
            # MCP
            CodexListMcpResourcesTool, CodexListMcpResourceTemplatesTool, CodexReadMcpResourceTool,
            # Batch
            CodexSpawnAgentsOnCsvTool, CodexReportAgentJobResultTool,
            # Multi-agent (shared with claude/opencode)
            SpawnAgentTool, SendInputTool, WaitTool,
            CloseAgentTool, ResumeAgentTool, ListAgentsTool,
        ],
    ),

    # OpenCode: matches the actual opencode tool set
    "opencode": ToolProfile(
        name="opencode",
        description=(
            "OpenCode style — full tool suite matching opencode. "
            "read(filePath, offset, limit), write(content, filePath), "
            "list(path, ignore), grep(pattern, path, include), "
            "edit(filePath, oldString, newString, replaceAll), "
            "bash(command, timeout, workdir, description), "
            "webfetch(url, format, timeout), websearch(query, numResults, ...), "
            "todowrite(todos), todoread(), plan_exit(), task(...), "
            "apply_patch(patchText), codesearch(query), lsp(...), "
            "multiedit(filePath, edits), question(questions), skill(name), batch(tool_calls)."
        ),
        _factories=[
            OpencodeReadTool, OpencodeWriteTool, OpencodeListTool,
            OpencodeGlobTool, OpencodeGrepTool, OpencodeEditTool,
            OpencodeBashTool, OpencodeWebFetchTool, OpencodeWebSearchTool,
            OpencodeTodoWriteTool, OpencodeTodoReadTool,
            OpencodePlanExitTool, OpencodeTaskTool,
            OpencodeApplyPatchTool, OpencodeCodeSearchTool,
            OpencodeLspTool, OpencodeMultiEditTool,
            OpencodeQuestionTool, OpencodeSkillTool, OpencodeBatchTool,
            # Multi-agent tools
            SpawnAgentTool, SendInputTool, WaitTool,
            CloseAgentTool, ResumeAgentTool, ListAgentsTool,
        ],
    ),
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_profile(name: str) -> ToolProfile:
    """
    Return a ToolProfile by name.
    Falls back to 'default' and prints a warning if name is unknown.
    """
    profile = _PROFILES.get(name.lower())
    if profile is None:
        known = ", ".join(_PROFILES)
        print(f"[warning] unknown tool profile {name!r}, using 'default'. Known: {known}")
        return _PROFILES["default"]
    return profile


def list_profiles() -> list[ToolProfile]:
    return list(_PROFILES.values())


def register_profile(profile: ToolProfile) -> None:
    """Add or replace a profile at runtime (for custom setups)."""
    _PROFILES[profile.name.lower()] = profile


def infer_profile(model_name: str) -> str:
    """
    Guess the best profile from a model name when tool_profile='auto'.

    Examples:
        "claude-opus-4-6"          → "claude"
        "gemini-2.0-flash"         → "gemini"
        "opencode-dev"             → "opencode"
        "gpt-4o"                   → "gpt"
        "llama3.2" / "mistral"     → "default"
    """
    m = model_name.lower()
    if "claude" in m:
        return "claude"
    if "gemini" in m:
        return "gemini"
    if "opencode" in m or "open code" in m:
        return "opencode"
    if "codex" in m:
        return "codex"
    if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
        return "gpt"
    if "qwen" in m:
        return "qwen"
    return "default"
