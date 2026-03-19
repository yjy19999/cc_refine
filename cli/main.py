from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent import Agent, Config
from agent.telemetry import SessionMetrics, TokenUsageStats
from agent.tools.profiles import get_profile, list_profiles
from cli.input import InputPrompt
from cli.terminal import open_terminal
console = Console()

# ── Tool display name mappings ───────────────────────────────────────
# Maps the actual tool name (what the LLM calls) → a shorter display name
# shown in the terminal. Covers both the gemini and qwen profiles where
# the raw name would be noisy or misleading.

GEMINI_TOOL_DISPLAY_NAMES: dict[str, str] = {
    "read_file":         "ReadFile",
    "write_file":        "WriteFile",
    "replace":           "Replace",
    "glob":              "Glob",
    "grep_search":       "Grep",
    "list_directory":    "ListDirectory",
    "run_shell_command": "Shell",
    "google_web_search": "WebSearch",   # differs most from display name
    "web_fetch":         "WebFetch",
    "write_todos":       "WriteTodos",
    "save_memory":       "SaveMemory",
    "read_many_files":   "ReadManyFiles",
    "get_internal_docs": "GetInternalDocs",
    "ask_user":          "AskUser",
    "enter_plan_mode":   "EnterPlanMode",
    "exit_plan_mode":    "ExitPlanMode",
    "activate_skill":    "ActivateSkill",
}

QWEN_TOOL_DISPLAY_NAMES: dict[str, str] = {
    "read_file":         "ReadFile",
    "write_file":        "WriteFile",
    "edit":              "Edit",
    "glob":              "Glob",
    "grep_search":       "Grep",
    "list_directory":    "ListDirectory",
    "run_shell_command": "Shell",
    "web_fetch":         "WebFetch",
    "web_search":        "WebSearch",
    "todo_write":        "TodoWrite",
    "save_memory":       "SaveMemory",
    "task":              "Task",
    "skill":             "Skill",
    "lsp":               "Lsp",
    "exit_plan_mode":    "ExitPlanMode",
}

OPENCODE_TOOL_DISPLAY_NAMES: dict[str, str] = {
    "read": "Read",
    "write": "Write",
    "list": "List",
    "glob": "Glob",
    "grep": "Grep",
    "edit": "Edit",
    "bash": "Bash",
    "webfetch": "WebFetch",
    "websearch": "WebSearch",
    "todowrite": "TodoWrite",
    "todoread": "TodoRead",
    "plan_exit": "PlanExit",
    "task": "Task",
    "apply_patch": "ApplyPatch",
    "codesearch": "CodeSearch",
    "lsp": "Lsp",
    "multiedit": "MultiEdit",
    "question": "Question",
    "skill": "Skill",
    "batch": "Batch",
}

# Combined lookup used by the renderer — gemini entries first so that
# any name shared between profiles resolves consistently.
_TOOL_DISPLAY_NAMES: dict[str, str] = {
    **GEMINI_TOOL_DISPLAY_NAMES,
    **QWEN_TOOL_DISPLAY_NAMES,
    **OPENCODE_TOOL_DISPLAY_NAMES,
}


def _display_name(tool_name: str) -> str:
    """Return the display name for a tool, falling back to the raw name."""
    return _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)


def _shorten_path(path: Path, max_width: int) -> str:
    """Return the full path, shrinking from the left with ... if too long."""
    full = str(path)
    if len(full) <= max_width:
        return full
    parts = path.parts
    for i in range(1, len(parts)):
        candidate = ".../" + "/".join(parts[i:])
        if len(candidate) <= max_width:
            return candidate
    # Last resort: hard truncate
    return "..." + full[-(max_width - 3):]


def _make_status_line(agent: "Agent", elapsed: float | None = None) -> str:
    """Single status line: path on the left, optional ⏱ timer centered, tokens on the right."""
    cols = shutil.get_terminal_size((80, 24)).columns
    total = agent.metrics.get_summary().get("total_tokens", 0)
    right = f" tokens: {total:,} " if total else " tokens: — "

    if elapsed is not None:
        mid = f" ⏱  {elapsed:.1f}s "
        # Divide remaining width evenly between left and right padding
        remaining = cols - len(mid) - len(right)
        left_width = max(10, remaining // 2)
        left = " " + _shorten_path(Path(os.getcwd()), left_width - 1)
        pad = remaining - len(left)
        return left + " " * max(0, pad) + mid + right
    else:
        max_left = max(10, cols - len(right) - 2)
        left = " " + _shorten_path(Path(os.getcwd()), max_left - 1)
        gap = max(1, cols - len(left) - len(right))
        return left + " " * gap + right


def _make_token_toolbar(agent: "Agent") -> str:
    """Border line + status bar: full cwd path on the left, token count on the right."""
    cols = shutil.get_terminal_size((80, 24)).columns
    return "─" * cols + "\n" + _make_status_line(agent)


HELP_TEXT = """
[bold]Commands:[/bold]
  [cyan]/help[/cyan]              Show this help
  [cyan]/plan[/cyan]              Toggle plan-then-execute mode
  [cyan]/verbose[/cyan]           Toggle verbose mode (full tool I/O + live thinking)
  [cyan]/profile [name][/cyan]    Show or switch tool profile
  [cyan]/reset[/cyan]             Clear conversation history
  [cyan]/history[/cyan]           Show conversation history
  [cyan]/tools[/cyan]             List available tools
  [cyan]/model <name>[/cyan]      Switch model
  [cyan]/stats[/cyan]             Show session token usage & stats
  [cyan]/sessions[/cyan]          List saved sessions
  [cyan]/resume <id>[/cyan]       Resume a saved session
  [cyan]/delete <id>[/cyan]       Delete a saved session
  [cyan]/terminal[/cyan]          Open an embedded interactive shell (exit or Ctrl+D to return)
  [cyan]/exit[/cyan]              Quit (also: Ctrl+C, Ctrl+D)
""".strip()


# ── Formatting helpers ──────────────────────────────────────────────


def _format_duration(ms: float) -> str:
    """Format milliseconds into a human-readable duration string."""
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def _format_number(n: int) -> str:
    """Format a number with thousands separators."""
    return f"{n:,}"


# ── Stats display ───────────────────────────────────────────────────


def _print_session_stats_simple(metrics: SessionMetrics, title: str = "Session Stats") -> None:
    """Print stats panel using only Rich text (no inline Table rendering issues)."""
    summary = metrics.get_summary()

    lines = []

    # ── Performance ─────────────────────────────────────────────────
    wall_time = _format_duration(summary["session_duration_ms"])
    api_time = _format_duration(summary["total_api_time_ms"])
    tool_time = _format_duration(summary["total_tool_time_ms"])
    active_time_ms = summary["total_api_time_ms"] + summary["total_tool_time_ms"]
    active_time = _format_duration(active_time_ms)
    api_pct = (summary["total_api_time_ms"] / active_time_ms * 100) if active_time_ms > 0 else 0
    tool_pct = (summary["total_tool_time_ms"] / active_time_ms * 100) if active_time_ms > 0 else 0

    lines.append("[bold]Performance[/bold]")
    lines.append(f"  Wall Time:       {wall_time}")
    lines.append(f"  Agent Active:    {active_time}")
    lines.append(f"    » API Time:    {api_time} [dim]({api_pct:.1f}%)[/dim]")
    lines.append(f"    » Tool Time:   {tool_time} [dim]({tool_pct:.1f}%)[/dim]")
    lines.append("")

    # ── Token Usage ─────────────────────────────────────────────────
    lines.append("[bold]Token Usage[/bold]")
    lines.append(f"  Total Tokens:    {_format_number(summary['total_tokens'])}")
    lines.append(f"    » Input:       {_format_number(summary['total_input_tokens'])}")
    lines.append(f"    » Output:      {_format_number(summary['total_output_tokens'])}")
    if summary["total_cached_tokens"] > 0:
        lines.append(f"    » Cached:      {_format_number(summary['total_cached_tokens'])}")
    lines.append(f"  API Requests:    {summary['total_api_requests']}")
    lines.append("")

    # ── Model breakdown ─────────────────────────────────────────────
    models_data = summary.get("models", {})
    if models_data:
        lines.append("[bold]Model Usage[/bold]")
        for model_name, m in models_data.items():
            lines.append(f"  [cyan]{model_name}[/cyan]")
            lines.append(
                f"    Reqs: {m['requests']}  "
                f"In: {_format_number(m['input_tokens'])}  "
                f"Out: {_format_number(m['output_tokens'])}  "
                f"Total: {_format_number(m['total_tokens'])}"
            )
            if m["cached_tokens"] > 0:
                lines.append(
                    f"    Cache: {_format_number(m['cached_tokens'])} "
                    f"({m['cache_hit_rate']:.1f}% hit rate)"
                )
            lines.append(f"    Avg Latency: {_format_duration(m['avg_latency_ms'])}")
        lines.append("")

    # ── Tool Usage ──────────────────────────────────────────────────
    tools_data = summary.get("tools", {})
    if tools_data["total_calls"] > 0:
        lines.append("[bold]Tool Usage[/bold]")
        lines.append(
            f"  Calls: [green]✓ {tools_data['total_success']}[/green]  "
            f"[red]✗ {tools_data['total_fail']}[/red]  "
            f"({tools_data['total_calls']} total, "
            f"{tools_data['success_rate']:.1f}% success)"
        )
        by_name = tools_data.get("by_name", {})
        if by_name:
            for tn, ts in sorted(by_name.items(), key=lambda x: x[1]["count"], reverse=True):
                lines.append(
                    f"    [cyan]{tn:<20}[/cyan] "
                    f"×{ts['count']:>3}  "
                    f"[green]✓{ts['success']}[/green] [red]✗{ts['fail']}[/red]  "
                    f"[dim]{_format_duration(ts['duration_ms'])}[/dim]"
                )

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold green]{title}[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


# ── Banner ──────────────────────────────────────────────────────────


def print_banner(agent: Agent, plan_mode: bool) -> None:
    plan_tag = "  [yellow]plan-first: on[/yellow]" if plan_mode else ""
    profile = getattr(agent, "profile_name", "custom")
    console.print(Panel(
        f"[bold green]AI Agent[/bold green]\n"
        f"[dim]model:[/dim] [cyan]{agent.config.model}[/cyan]  "
        f"[dim]url:[/dim] [cyan]{agent.config.base_url}[/cyan]  "
        f"[dim]profile:[/dim] [cyan]{profile}[/cyan]  "
        f"[dim]tools:[/dim] [cyan]{len(agent.registry)}[/cyan]"
        f"{plan_tag}\n"
        f"[dim]Type [/dim][cyan]/help[/cyan][dim] for commands, [/dim]"
        f"[cyan]Ctrl+C[/cyan][dim] to quit.[/dim]",
        border_style="green",
    ))


# ── Command handler ─────────────────────────────────────────────────


def handle_command(
    line: str, agent: Agent, plan_mode: bool, verbose: bool = False, input_prompt: InputPrompt = None
) -> tuple[bool, bool, bool]:
    """
    Handle slash commands.
    Returns (was_command, new_plan_mode, new_verbose).
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/help":
        console.print(HELP_TEXT)

    elif cmd == "/plan":
        plan_mode = not plan_mode
        status = "[yellow]on[/yellow]" if plan_mode else "[dim]off[/dim]"
        console.print(f"Plan-first mode: {status}")

    elif cmd == "/verbose":
        verbose = not verbose
        status = "[yellow]on[/yellow]" if verbose else "[dim]off[/dim]"
        console.print(
            f"Verbose mode: {status}"
            + ("  [dim](live thinking · full tool I/O)[/dim]" if verbose else "")
        )

    elif cmd == "/profile":
        if not arg:
            # Show all profiles, highlight current
            current = getattr(agent, "profile_name", "custom")
            for p in list_profiles():
                marker = "[green]●[/green]" if p.name == current else " "
                names = ", ".join(p.tool_names())
                console.print(f"  {marker} [cyan]{p.name:<10}[/cyan] {p.description}")
                console.print(f"             [dim]tools: {names}[/dim]")
        else:
            profile = get_profile(arg)
            agent.registry = profile.build_registry()
            agent.profile_name = profile.name
            agent.reset()
            names = ", ".join(profile.tool_names())
            console.print(
                f"[dim]Switched to profile [cyan]{profile.name}[/cyan] "
                f"({len(agent.registry)} tools: {names}). History cleared.[/dim]"
            )

    elif cmd == "/reset":
        agent.reset()
        console.print("[dim]Conversation history cleared.[/dim]")

    elif cmd == "/history":
        for msg in agent.history:
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            if role == "system":
                console.print(f"[dim][system] {str(content)[:80]}...[/dim]")
            elif role == "user":
                console.print(f"[bold cyan][you][/bold cyan] {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                    console.print(f"[bold green][agent][/bold green] [dim](called: {names})[/dim]")
                elif content:
                    console.print(f"[bold green][agent][/bold green] {str(content)[:200]}")
            elif role == "tool":
                result = str(content)[:100]
                console.print(f"[dim][tool:{msg.get('name')}] {result}[/dim]")

    elif cmd == "/tools":
        for schema in agent.registry.schemas():
            fn = schema["function"]
            first_line = fn['description'].split('\n')[0]
            console.print(f"  [cyan]{fn['name']}[/cyan] — {first_line[:80]}")

    elif cmd == "/model":
        if not arg:
            console.print(f"Current model: [cyan]{agent.config.model}[/cyan]")
        else:
            agent.config.model = arg
            agent.client.config.model = arg
            console.print(f"[dim]Switched to model: [cyan]{arg}[/cyan][/dim]")

    elif cmd in ("/stats", "/usage"):
        _print_session_stats_simple(agent.metrics, title="Session Stats")

    elif cmd == "/sessions":
        sessions = agent.list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
        else:
            console.print("[bold]Saved Sessions:[/bold]")
            for i, sess in enumerate(sessions, 1):
                console.print(
                    f"  [cyan]{i}.[/cyan] [bold]{sess['id'][:8]}[/bold] "
                    f"({sess['message_count']} messages, {sess['last_updated'][:10]})"
                )
                if sess['summary']:
                    console.print(f"     [dim]{sess['summary'][:60]}...[/dim]")

    elif cmd == "/resume":
        if not arg:
            console.print("[yellow]Usage: /resume <session_id>[/yellow]")
        else:
            record = agent.resume_session(arg)
            if record:
                console.print(
                    f"[green]✓ Resumed session {record.session_id[:8]} "
                    f"with {len(record.messages)} messages[/green]"
                )
            else:
                console.print(f"[red]✗ Session not found: {arg}[/red]")

    elif cmd == "/delete":
        if not arg:
            console.print("[yellow]Usage: /delete <session_id>[/yellow]")
        else:
            if agent.delete_session(arg):
                console.print(f"[green]✓ Deleted session {arg}[/green]")
            else:
                console.print(f"[red]✗ Failed to delete session {arg}[/red]")

    elif cmd == "/terminal":
        shell = arg.strip() or None
        open_terminal(console, shell=shell)
        # Reset prompt_toolkit session after pty.spawn
        if input_prompt:
            input_prompt.reset()

    elif cmd in ("/exit", "/quit", "/q"):
        agent.save_session()
        _print_session_stats_simple(agent.metrics, title="Goodbye — Session Summary")
        sys.exit(0)

    else:
        console.print(f"[red]Unknown command:[/red] {cmd}  (type /help)")

    return True, plan_mode, verbose


_VERBOSE_MAX_RESULT_LINES = 25


def render_event(event, verbose: bool = False) -> None:
    """Render a single TurnEvent to the console."""
    if event.type == "text":
        print(event.data, end="", flush=True)
    elif event.type == "tool_start":
        name = _display_name(event.data["name"])
        if verbose:
            console.print(f"\n[dim]  ▶ {name}[/dim]")
            for k, v in event.data["arguments"].items():
                v_str = json.dumps(v) if not isinstance(v, str) else repr(v)
                console.print(f"[dim]      {k} = {v_str}[/dim]")
        else:
            arg_str = _format_args(event.data["arguments"])
            console.print(f"\n[dim]  ▶ {name}({arg_str})[/dim]")
    elif event.type == "tool_end":
        name = _display_name(event.data.get("name", ""))
        result = event.data["result"]
        lines = result.splitlines()
        if verbose:
            label = f"  ◀ {name}" if name else "  ◀"
            console.print(f"[dim]{label}  ({len(lines)} line{'s' if len(lines) != 1 else ''})[/dim]")
            shown = lines[:_VERBOSE_MAX_RESULT_LINES]
            for line in shown:
                console.print(f"[dim]    {line}[/dim]")
            if len(lines) > _VERBOSE_MAX_RESULT_LINES:
                console.print(f"[dim]    ... +{len(lines) - _VERBOSE_MAX_RESULT_LINES} more lines[/dim]")
        else:
            preview = lines[0][:120] if lines else ""
            suffix = f"  [dim]+{len(lines)-1} more lines[/dim]" if len(lines) > 1 else ""
            console.print(f"[dim]  ◀ {preview}{suffix}[/dim]")
    elif event.type == "usage":
        data = event.data
        if isinstance(data, dict) and data.get("final"):
            stats: TokenUsageStats = data["stats"]
            console.print(f" [dim]({stats.input_tokens}in, {stats.output_tokens}out tokens, {stats.latency_ms/1000:.1f}s)[/dim]")
    elif event.type == "compressed":
        d = event.data
        saved = d["original_tokens"] - d["new_tokens"]
        console.print(
            f"\n[dim]  ⚡ context compressed  "
            f"{d['original_tokens']:,} → {d['new_tokens']:,} tokens  "
            f"(-{saved:,})[/dim]"
        )
    elif event.type == "error":
        console.print(f"\n[red]{event.data}[/red]")


_ANSI_GREY  = "\033[38;5;242m"
_ANSI_RESET = "\033[0m"


def _stream_events(
    events,
    capture_plan: bool = False,
    verbose: bool = False,
    agent: "Agent | None" = None,
    pre_label: str | None = None,
) -> dict | None:
    """
    Consume TurnEvents, rendering text as Markdown (normal) or live stream (verbose).

    Normal mode  — text is buffered and displayed via Rich Live as rendered
                   Markdown once each batch completes.
    Verbose mode — text tokens stream immediately so you can watch the model
                   think in real-time; tool I/O shows full args and results.

    When agent is provided the timer rewrites the status bar line in-place so
    the elapsed clock appears centred inside the bar. pre_label (Rich markup)
    is printed once the first event arrives, just before normal rendering.

    Returns plan data dict if capture_plan=True and a plan_ready event arrives.
    """
    text_buffer = ""
    live: Live | None = None
    showed_label = False
    plan_data = None
    got_first_event = False

    t0 = time.monotonic()
    timer_stop = threading.Event()

    def _tick() -> None:
        while not timer_stop.wait(0.1):
            elapsed = time.monotonic() - t0
            if agent is not None:
                line = _make_status_line(agent, elapsed)
                # \033[1A  move up one line (onto the status line)
                # \r\033[2K clear it, then rewrite with timer, then \n back down
                sys.stdout.write(f"\033[1A\r\033[2K{line}\n")
            else:
                sys.stdout.write(f"\r\033[2K{_ANSI_GREY}⏱  {elapsed:.1f}s...{_ANSI_RESET}")
            sys.stdout.flush()

    timer = threading.Thread(target=_tick, daemon=True)
    timer.start()

    try:
        for event in events:
            if not got_first_event:
                got_first_event = True
                timer_stop.set()
                timer.join(timeout=0.3)
                if agent is not None:
                    # Restore the status line without the timer
                    line = _make_status_line(agent)
                    sys.stdout.write(f"\033[1A\r\033[2K{line}\n")
                else:
                    sys.stdout.write("\r\033[2K")
                sys.stdout.flush()
                if pre_label:
                    console.print(pre_label)
            if event.type == "text":
                if not showed_label:
                    if verbose:
                        console.print("\n[bold green]agent[/bold green] [dim](thinking):[/dim]")
                    else:
                        console.print("\n[bold green]agent:[/bold green]")
                    showed_label = True

                if verbose:
                    # Stream raw tokens immediately — thinking visible in real-time
                    print(event.data, end="", flush=True)
                else:
                    # Buffer and render as live Markdown
                    text_buffer += event.data
                    if live is None:
                        live = Live(
                            RichMarkdown(text_buffer),
                            console=console,
                            refresh_per_second=12,
                            transient=False,
                        )
                        live.start()
                    else:
                        live.update(RichMarkdown(text_buffer))

            elif event.type == "plan_ready" and capture_plan:
                plan_data = event.data

            elif event.type == "done":
                if live is not None:
                    live.stop()
                    live = None
                elif verbose and showed_label:
                    print()  # final newline after streamed text
                elapsed = time.monotonic() - t0
                console.print(f"[dim]⏱  {elapsed:.1f}s[/dim]")
                break

            else:
                # Stop live display / flush streamed text before inline events
                if live is not None:
                    live.stop()
                    live = None
                    text_buffer = ""
                elif verbose and showed_label:
                    print()  # newline between thinking text and tool line
                    showed_label = False
                render_event(event, verbose=verbose)

    finally:
        timer_stop.set()
        if live is not None:
            live.stop()
        # Ensure cursor is on a clean line after any interruption
        if not got_first_event:
            # Timer was still running — clear the status/timer line
            if agent is not None:
                line = _make_status_line(agent)
                sys.stdout.write(f"\033[1A\r\033[2K{line}\n")
            else:
                sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

    return plan_data


def run_turn(agent: Agent, user_input: str, verbose: bool = False) -> None:
    """Normal mode: send input and stream response."""
    _stream_events(agent.run(user_input), verbose=verbose, agent=agent)


def run_plan_turn(agent: Agent, user_input: str, verbose: bool = False) -> None:
    """Plan-then-execute mode: generate plan, ask approval, then execute."""

    # ── Phase 1: planning ──────────────────────────────────────────────
    plan = _stream_events(
        agent.generate_plan(user_input),
        capture_plan=True,
        verbose=verbose,
        agent=agent,
        pre_label="\n[yellow]Planning...[/yellow]",
    )

    if plan is None:
        console.print("[red]No plan was produced.[/red]")
        return

    # ── Show plan & ask for approval ───────────────────────────────────
    steps = plan.get("steps", [])
    summary = plan.get("summary", "")

    lines = []
    if summary:
        lines.append(f"[bold]{summary}[/bold]")
        lines.append("")
    for i, step in enumerate(steps, 1):
        lines.append(f"  [cyan]{i}.[/cyan] {step}")

    console.print(Panel("\n".join(lines), title="[yellow]Proposed Plan[/yellow]", border_style="yellow"))

    try:
        answer = console.input("\n[yellow]Proceed with this plan? (y/n):[/yellow] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Cancelled.[/dim]")
        agent.reset()
        return

    if answer not in ("y", "yes"):
        console.print("[dim]Plan rejected. History cleared.[/dim]")
        agent.reset()
        return

    # ── Phase 2: execution ─────────────────────────────────────────────
    _stream_events(
        agent.execute(),
        verbose=verbose,
        agent=agent,
        pre_label="\n[green]Executing plan...[/green]",
    )


def _format_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        v_str = json.dumps(v) if not isinstance(v, str) else repr(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "..."
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


def main() -> None:
    config = Config()
    agent = Agent(config=config)
    plan_mode = False
    verbose = False

    input_prompt = InputPrompt()
    print_banner(agent, plan_mode)

    while True:
        try:
            tags = []
            if plan_mode:
                tags.append("(plan)")
            if verbose:
                tags.append("(verbose)")
            tag_str = " ".join(tags) + " " if tags else ""
            print()
            user_input = input_prompt.get_input(
                f"you {tag_str}: ",
                bottom_toolbar=lambda: _make_token_toolbar(agent),
            ).strip()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted (Ctrl+C).[/dim]")
            agent.save_session()
            _print_session_stats_simple(agent.metrics, title="Goodbye — Session Summary")
            break
        except EOFError:
            console.print("\n[dim]EOF received (Ctrl+D).[/dim]")
            agent.save_session()
            _print_session_stats_simple(agent.metrics, title="Goodbye — Session Summary")
            break
        except Exception as e:
            console.print(f"\n[red]Unexpected loop error:[/red] {e}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            try:
                _, plan_mode, verbose = handle_command(user_input, agent, plan_mode, verbose, input_prompt)
            except Exception as exc:
                console.print(f"\n[red]Command error:[/red] {exc}")
            continue

        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if not cmd:
                continue
            try:
                subprocess.run(cmd, shell=True, check=False)
            except Exception as e:
                console.print(f"[red]Error executing shell command:[/red] {e}")
            continue

        console.print(_make_token_toolbar(agent))
        try:
            if plan_mode:
                run_plan_turn(agent, user_input, verbose=verbose)
            else:
                run_turn(agent, user_input, verbose=verbose)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")


if __name__ == "__main__":
    main()
