"""
CodingTaskRunner — automated code + test + fix loop.

Usage::

    from agent import CodingTaskRunner, Config

    runner = CodingTaskRunner(
        workspace="/tmp/my_project",
        config=Config(model="gpt-4o", base_url="...", api_key="..."),
    )
    result = runner.run("Write a Python module that implements a binary search tree")

    print(result.status)        # "passed" | "failed" | "error"
    print(result.code_files)    # ["bst.py"]
    print(result.test_files)    # ["test_bst.py"]
    print(result.test_output)   # final pytest output
    print(result.iterations)    # how many fix cycles it took

Streaming::

    for event in runner.run_stream("Write a calculator module"):
        if event.type == "text":
            print(event.data, end="", flush=True)
        elif event.type == "phase":
            print(f"\\n--- {event.data} ---")

All file operations are sandboxed to the workspace folder.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import Agent, TurnEvent
from .config import Config
from .sandbox import SandboxedRegistry
from .tools.profiles import get_profile, infer_profile


@dataclass
class TaskResult:
    """Outcome of a coding task run."""
    task: str
    status: str = ""  # "passed" | "failed" | "error"
    iterations: int = 0
    code_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    doc_files: list[str] = field(default_factory=list)
    test_output: str = ""
    error: str = ""
    conversation_log: list[dict[str, Any]] = field(default_factory=list)


_CODING_SYSTEM_PROMPT = """\
You are a coding agent. You write clean, working Python code and tests.

IMPORTANT RULES:
1. All code and test files MUST be created inside the current working directory.
   Use relative paths only (e.g. "calculator.py", "tests/test_calculator.py").
   NEVER use absolute paths or paths starting with "/" or "..".
2. Write production-quality code with proper error handling.
3. When asked to write tests, use pytest. Each test function must start with "test_".
4. Keep file names simple and descriptive.
5. If fixing code after a test failure, read the failing test output carefully
   and make targeted fixes. Do not rewrite everything from scratch.
"""

_TASK_INTAKE_PROMPT = """\
Task: {task}

Do NOT write any code yet. First, analyse and understand the task thoroughly.

Write a file called `Task.md` with the following sections:

## Task Goal
What exactly needs to be built? State the objective clearly in 1-2 sentences.

## Inputs and Outputs
- What inputs does the code accept? (arguments, files, data formats, etc.)
- What outputs does it produce? (return values, files, side effects, etc.)

## Constraints
- Language/framework requirements
- Performance requirements (if any)
- Compatibility or environment constraints
- Any explicit restrictions from the task description

## Modification Scope
- Which files need to be created?
- What classes, functions, or modules are required?
- Estimated complexity (small / medium / large)

## Risks
- Potential pitfalls or ambiguities in the task
- Edge cases that could cause issues
- Dependencies or assumptions that might not hold

## Success Criteria
- How do we know the task is complete?
- What should the tests verify?
- What does "working correctly" look like?

Use relative file paths only. Be concise and specific — this document will guide \
the implementation that follows.
"""

_REPO_RECON_PROMPT = """\
Do NOT write any code yet. First, understand the current workspace / codebase.

Use the available tools (list directories, read files, grep, glob) to explore \
the workspace and answer the following:

1. **Entry files** — What are the main entry points (scripts, __main__.py, etc.)?
2. **Relevant modules** — Which modules or packages exist? What does each one do?
3. **Existing tests** — Are there any test files? What framework do they use?
4. **Configuration files** — Any config files (.env, pyproject.toml, setup.cfg, \
   Makefile, tox.ini, etc.)?
5. **Build & run** — How is the project built, installed, and run?
6. **Similar implementations** — Any existing code that overlaps with the task \
   described in Task.md?
7. **Project conventions** — What patterns are used for linting, typing, testing, \
   naming, imports?

After exploring, write a file called `Repo.md` with these sections:

## Relevant Files
List the files that matter for the task.

## Dependency Graph
How the key modules depend on each other.

## Candidate Modification Points
Where new code should be added or existing code changed.

## Risky / Sensitive Modules
Modules that are fragile, tightly coupled, or have many dependents.

Keep it concise and actionable — this document will guide the implementation.
Use relative file paths only.
"""

_PLAN_PROMPT = """\
Do NOT write any code yet. Based on Task.md and Repo.md, create a solution plan.

Write a file called `Plan.md` with the following sections:

## Files to Modify / Create
List each file that will be created or modified, with a one-line description of \
what it will contain.

## Changes per File
For each file, describe the specific changes:
- New classes, functions, or methods to add
- Existing code to modify (if any)
- Key logic and algorithms

## Test Plan
- What test files will be created
- What test cases will cover (happy path, edge cases, error cases)
- Any fixtures or helpers needed

## Execution Order
Number the steps in the order they should be done:
1. What to implement first (core logic, base classes, etc.)
2. What depends on what
3. What to do last

## Expected Risks
- What could go wrong during implementation
- Tricky parts that need extra care
- Assumptions that might not hold

Keep it concise and actionable. Use relative file paths only.
"""

_WRITE_CODE_PROMPT = """\
Now write the code following the plan in Plan.md.
Create the necessary Python file(s) in the execution order specified.
Use relative file paths only. Do NOT write tests yet — just the implementation.
After writing, briefly list the files you created.
"""

_WRITE_TESTS_PROMPT = """\
Now write pytest tests for the code you just created.
- Put tests in a file named test_<module>.py (or tests/test_<module>.py).
- Cover the main functionality, edge cases, and error cases.
- Use relative file paths only.
- After writing, list the test files you created.
"""

_FIX_PROMPT = """\
The tests failed. Here is the output:

```
{test_output}
```

Read the relevant source and test files, understand the failures,
and fix the code. You may fix either the implementation or the tests
(or both) as appropriate. After fixing, briefly describe what you changed.
"""

_REVIEW_PROMPT = """\
Review the code and tests you have written against the requirements in Task.md.

Read all the source files and test files, then evaluate:

1. **Correctness** — Does the code implement all requirements from Task.md?
2. **Completeness** — Are all functions/classes/methods present? Any missing features?
3. **Edge cases** — Are edge cases handled in the code and covered by tests?
4. **Code quality** — Is the code clean, well-structured, and following conventions?
5. **Test coverage** — Do tests cover the success criteria from Task.md?

After your review, you MUST write a file called `Review.md` with your findings \
and a final verdict. The LAST line of Review.md MUST be exactly one of:

    VERDICT: PASS
    VERDICT: FAIL

Use PASS if the code and tests are acceptable. Use FAIL if there are issues \
that require rewriting tests or fixing code.
If FAIL, clearly list what needs to be fixed so the next iteration can address it.
Use relative file paths only.
"""

_WRITE_DOCS_PROMPT = """\
Now write documentation for the code you created. Produce four separate files:

### 1. README.md — User-facing documentation
From the user's perspective:
- Brief description of what the module does
- Installation / dependencies (if any)
- Usage examples showing how to import and use the main classes/functions
- API reference: list each public class/function with its parameters and return values
- Known limitations or caveats

### 2. CHANGES.md — Change summary
From the developer's perspective:
- What files were created or modified
- What was the design approach and why
- Key implementation decisions and trade-offs
- Any deviations from the original plan (Plan.md)

### 3. TESTS.md — Test summary
From the validation perspective:
- List of all test files and what they cover
- Summary of test results (which passed, which failed, if any)
- Test coverage gaps (if any)
- How to run the tests

### 4. FOLLOWUPS.md — Limitations and follow-ups
Be explicit and honest about:
- What problems remain unresolved
- Which parts are temporary workarounds
- Known edge cases not yet handled
- Performance or scalability concerns
- Suggested next steps and improvements

Even if some tests fail, this documentation makes the delivery valuable by being \
transparent about the current state.
Keep all files concise and practical. Use relative file paths only.
"""

_TEST_CMD = "python -m pytest -x -v --tb=short 2>&1"


class CodingTaskRunner:
    """
    Orchestrates: write code → write tests → run tests → fix → repeat.

    All file operations are sandboxed to the specified workspace folder.
    """

    def __init__(
        self,
        workspace: str | Path,
        config: Config | None = None,
        max_fix_iterations: int = 5,
        max_review_iterations: int = 2,
        test_command: str = _TEST_CMD,
        session_id: str | None = None,
        logs_dir: str | Path | None = None,
        memory_log_dir: str | Path | None = None,
    ):
        """
        Args:
            workspace: Folder where all code will be created. Created if missing.
            config: LLM configuration. Uses defaults from .env if not provided.
            max_fix_iterations: Max fix→test cycles before giving up.
            max_review_iterations: Max review→rewrite cycles before proceeding.
            test_command: Shell command to run tests (default: pytest).
            session_id: Optional session ID. Passed to Agent so the trace
                        file and workspace folder share the same identifier.
            logs_dir: Directory for trajectory/trace files. Defaults to api_logs/.
            memory_log_dir: Directory for compression memory logs. Defaults to memory_logs/.
        """
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or Config()
        self.max_fix_iterations = max_fix_iterations
        self.max_review_iterations = max_review_iterations
        self.test_command = test_command
        self.session_id = session_id
        self.logs_dir = str(logs_dir) if logs_dir else None
        self.memory_log_dir = str(memory_log_dir) if memory_log_dir else None

    def _make_agent(self) -> Agent:
        """Create a sandboxed Agent."""
        # Build a sandboxed registry from the appropriate profile
        profile_name = self.config.tool_profile
        if profile_name == "auto":
            profile_name = infer_profile(self.config.model)

        base_registry = get_profile(profile_name).build_registry()
        sandbox = SandboxedRegistry(self.workspace)
        for schema in base_registry.schemas():
            tool = base_registry.get(schema["function"]["name"])
            if tool:
                sandbox.register(tool)

        agent = Agent(
            config=Config(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                model=self.config.model,
                stream=self.config.stream,
                max_tool_iterations=self.config.max_tool_iterations,
                tool_profile=self.config.tool_profile,
                context_limit=self.config.context_limit,
                compression_threshold=self.config.compression_threshold,
                system_prompt=_CODING_SYSTEM_PROMPT,
            ),
            registry=sandbox,
            session_id=self.session_id,
            logs_dir=self.logs_dir,
            memory_log_dir=self.memory_log_dir,
        )
        return agent

    # ------------------------------------------------------------------
    # Main blocking interface
    # ------------------------------------------------------------------

    def run(self, task: str, verbose: bool = True) -> TaskResult:
        """
        Run the full code → test → fix loop. Returns a TaskResult.

        When *verbose* is True (default), prints real-time progress to
        the terminal so you can monitor what the agent is doing.

        Args:
            task: Description of the coding task.
            verbose: Print live progress to stdout. Default True.

        Returns:
            TaskResult with status, files, and test output.
        """
        printer = _ProgressPrinter() if verbose else None
        result = TaskResult(task=task)
        try:
            for event in self.run_stream(task):
                if printer:
                    printer.handle(event)
                if event.type == "result":
                    return event.data
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            if printer:
                printer.error(str(e))
        return result

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def run_stream(self, task: str) -> Iterator[TurnEvent]:
        """
        Run the full loop, yielding TurnEvent objects for progress tracking.

        Extra event types:
            - ``phase`` — data is a string like "write_code", "write_tests",
              "run_tests", "fix_N"
            - ``test_result`` — data is {"passed": bool, "output": str}
            - ``result`` — data is the final TaskResult

        Args:
            task: Description of the coding task.

        Yields:
            TurnEvent objects.
        """
        agent = self._make_agent()
        result = TaskResult(task=task)

        # Phase 0: Task intake — analyse before coding
        yield TurnEvent(type="phase", data="task_intake")
        for event in agent.run(_TASK_INTAKE_PROMPT.format(task=task)):
            yield event

        # Phase 1: Repo reconnaissance — explore the workspace
        yield TurnEvent(type="phase", data="repo_recon")
        for event in agent.run(_REPO_RECON_PROMPT):
            yield event

        # Phase 2: Plan / solution design
        yield TurnEvent(type="phase", data="plan_design")
        for event in agent.run(_PLAN_PROMPT):
            yield event

        # Phase 3: Write code
        yield TurnEvent(type="phase", data="write_code")
        for event in agent.run(_WRITE_CODE_PROMPT):
            yield event

        review_passed = False
        for review_round in range(1, self.max_review_iterations + 1):
            # Phase 4: Write tests
            yield TurnEvent(type="phase", data="write_tests")
            for event in agent.run(_WRITE_TESTS_PROMPT):
                yield event

            # Discover created files
            result.code_files = self._find_files("*.py", exclude_prefix="test_")
            result.test_files = self._find_files("test_*.py")

            # Phase 5: Run tests → fix loop
            for iteration in range(1, self.max_fix_iterations + 1):
                result.iterations = iteration

                yield TurnEvent(type="phase", data=f"run_tests_{iteration}")
                test_passed, test_output = self._run_tests()
                result.test_output = test_output
                yield TurnEvent(type="test_result", data={
                    "passed": test_passed,
                    "output": test_output,
                    "iteration": iteration,
                })

                if test_passed:
                    result.status = "passed"
                    break

                # Fix phase
                yield TurnEvent(type="phase", data=f"fix_{iteration}")
                for event in agent.run(_FIX_PROMPT.format(test_output=test_output)):
                    yield event
            else:
                # Exhausted iterations without passing
                result.status = "failed"

            # Re-scan files in case fixes added new ones
            result.code_files = self._find_files("*.py", exclude_prefix="test_")
            result.test_files = self._find_files("test_*.py")

            # Phase 6: Review
            yield TurnEvent(type="phase", data=f"review_{review_round}")
            for event in agent.run(_REVIEW_PROMPT):
                yield event

            review_passed = self._check_review_verdict()
            yield TurnEvent(type="review_result", data={
                "passed": review_passed,
                "round": review_round,
            })

            if review_passed:
                break
            # Review failed — loop back to write_tests
        # end review loop

        if not review_passed:
            result.status = "failed"

        # Phase 7: Write documentation
        yield TurnEvent(type="phase", data="write_docs")
        for event in agent.run(_WRITE_DOCS_PROMPT):
            yield event
        result.doc_files = self._find_files("*.md")

        yield TurnEvent(type="result", data=result)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_review_verdict(self) -> bool:
        """Read Review.md and check if the last VERDICT line says PASS."""
        review_file = self.workspace / "Review.md"
        if not review_file.exists():
            return False
        try:
            text = review_file.read_text()
            for line in reversed(text.strip().splitlines()):
                line = line.strip().upper()
                if line.startswith("VERDICT:"):
                    return "PASS" in line
            return False
        except Exception:
            return False

    def _run_tests(self) -> tuple[bool, str]:
        """Run the test command in the workspace. Returns (passed, output)."""
        try:
            proc = subprocess.run(
                self.test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.workspace),
            )
            output = proc.stdout
            if proc.stderr:
                output += "\n" + proc.stderr
            return proc.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, "[error] tests timed out after 120s"
        except Exception as exc:
            return False, f"[error] {exc}"

    def _find_files(
        self, pattern: str, exclude_prefix: str | None = None
    ) -> list[str]:
        """Find files matching a glob pattern in the workspace."""
        results = []
        for p in self.workspace.rglob(pattern):
            if p.is_file():
                rel = str(p.relative_to(self.workspace))
                if exclude_prefix and p.name.startswith(exclude_prefix):
                    continue
                # Skip common noise
                if "__pycache__" in rel or ".pytest_cache" in rel:
                    continue
                results.append(rel)
        return sorted(results)


# ---------------------------------------------------------------------------
# Rich console progress printer
# ---------------------------------------------------------------------------

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

_PHASE_ICONS = {
    "task_intake":  ("clipboard",  "Phase 0 — Task Intake"),
    "repo_recon":   ("magnifier",  "Phase 1 — Repo Reconnaissance"),
    "plan_design":  ("blueprint",  "Phase 2 — Plan / Solution Design"),
    "write_code":   ("keyboard",   "Phase 3 — Writing Code"),
    "write_tests":  ("test_tube",  "Phase 4 — Writing Tests"),
    "write_docs":   ("memo",       "Phase 7 — Writing Documentation"),
}


class _ProgressPrinter:
    """Rich-powered real-time progress printer for task runs."""

    def __init__(self):
        self._console = Console()
        self._in_text = False
        self._text_buf: list[str] = []
        self._tool_count = 0

    def handle(self, event: TurnEvent) -> None:
        if event.type == "phase":
            self._flush_text()
            self._tool_count = 0
            label = self._phase_label(event.data)
            icon = self._phase_icon(event.data)
            self._console.print()
            self._console.rule(
                f"[bold cyan]{icon} {label}[/]", style="cyan",
            )

        elif event.type == "text":
            self._in_text = True
            self._text_buf.append(event.data)

        elif event.type == "tool_start":
            self._flush_text()
            self._tool_count += 1
            name = event.data.get("name", "?")
            args = event.data.get("arguments", {})
            summary = self._summarize_tool(name, args)
            self._console.print(
                f"  [dim]{self._tool_count}.[/] [bold yellow]{name}[/]"
                f"[dim]({summary})[/]"
            )

        elif event.type == "tool_end":
            result = event.data.get("result", "")
            name = event.data.get("name", "")
            first_line = result.split("\n")[0][:120] if result else ""
            if first_line.startswith("[ok]"):
                self._console.print(f"     [green]{first_line}[/]")
            elif first_line.startswith("[error]"):
                self._console.print(f"     [red]{first_line}[/]")
            else:
                self._console.print(f"     [dim]{first_line}[/]")

        elif event.type == "test_result":
            self._flush_text()
            passed = event.data.get("passed", False)
            output = event.data.get("output", "")
            iteration = event.data.get("iteration", "?")
            self._console.print()
            if passed:
                self._console.print(Panel(
                    f"[bold green]PASSED[/]  [dim](iteration {iteration})[/]",
                    border_style="green",
                    padding=(0, 2),
                ))
            else:
                lines = output.strip().splitlines()
                tail = lines[-min(15, len(lines)):]
                fail_text = Text()
                for line in tail:
                    if "FAILED" in line or "ERROR" in line:
                        fail_text.append(line + "\n", style="bold red")
                    elif "PASSED" in line:
                        fail_text.append(line + "\n", style="green")
                    elif line.startswith(">") or line.startswith("E "):
                        fail_text.append(line + "\n", style="red")
                    else:
                        fail_text.append(line + "\n", style="dim")
                self._console.print(Panel(
                    fail_text,
                    title=f"[bold red]FAILED[/] [dim](iteration {iteration})[/]",
                    border_style="red",
                    padding=(0, 1),
                ))

        elif event.type == "review_result":
            self._flush_text()
            passed = event.data.get("passed", False)
            round_num = event.data.get("round", "?")
            self._console.print()
            if passed:
                self._console.print(Panel(
                    f"[bold green]REVIEW PASSED[/]  [dim](round {round_num})[/]",
                    border_style="green",
                    padding=(0, 2),
                ))
            else:
                self._console.print(Panel(
                    f"[bold yellow]REVIEW FAILED[/]  [dim](round {round_num}) — looping back to write tests[/]",
                    border_style="yellow",
                    padding=(0, 2),
                ))

        elif event.type == "result":
            self._flush_text()
            r = event.data
            self._print_result(r)

        elif event.type == "usage":
            pass

        elif event.type == "error":
            self._flush_text()
            self._console.print(f"  [bold red]Error:[/] {event.data}")

    def error(self, msg: str) -> None:
        self._flush_text()
        self._console.print(Panel(
            f"[bold red]{msg}[/]",
            title="[red]Exception[/]",
            border_style="red",
        ))

    def _flush_text(self) -> None:
        if self._text_buf:
            text = "".join(self._text_buf).strip()
            if text:
                self._console.print()
                self._console.print(
                    Markdown(text),
                    style="white",
                )
            self._text_buf.clear()
            self._in_text = False

    def _print_result(self, r: TaskResult) -> None:
        self._console.print()

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            expand=False,
        )
        table.add_column("key", style="bold", no_wrap=True)
        table.add_column("value")

        if r.status == "passed":
            table.add_row("Status", "[bold green]PASSED[/]")
        else:
            table.add_row("Status", f"[bold red]{r.status.upper()}[/]")

        table.add_row("Iterations", str(r.iterations))

        if r.code_files:
            files_text = "\n".join(f"[cyan]{f}[/]" for f in r.code_files)
            table.add_row("Code Files", files_text)

        if r.test_files:
            files_text = "\n".join(f"[cyan]{f}[/]" for f in r.test_files)
            table.add_row("Test Files", files_text)

        if r.doc_files:
            files_text = "\n".join(f"[cyan]{f}[/]" for f in r.doc_files)
            table.add_row("Doc Files", files_text)

        if r.test_output:
            lines = r.test_output.strip().splitlines()
            if lines:
                summary = lines[-1]
                if "passed" in summary.lower():
                    table.add_row("Tests", f"[green]{summary}[/]")
                else:
                    table.add_row("Tests", f"[red]{summary}[/]")

        self._console.print(Panel(
            table,
            title="[bold]Result[/]",
            border_style="cyan" if r.status == "passed" else "red",
            padding=(1, 2),
        ))

    @staticmethod
    def _phase_label(phase: str) -> str:
        _, label = _PHASE_ICONS.get(phase, (None, None))
        if label:
            return label
        if phase.startswith("run_tests"):
            n = phase.split("_")[-1]
            return f"Running Tests (iteration {n})"
        if phase.startswith("fix_"):
            n = phase.split("_")[-1]
            return f"Fixing Code (attempt {n})"
        if phase.startswith("review_"):
            n = phase.split("_")[-1]
            return f"Phase 6 — Review (round {n})"
        return phase

    @staticmethod
    def _phase_icon(phase: str) -> str:
        if phase in ("task_intake", "repo_recon", "plan_design", "write_code", "write_tests", "write_docs"):
            return "[bold]>>>[/]"
        if phase.startswith(("run_tests", "fix_", "review_")):
            return "[bold]>>>[/]"
        return ">>>"

    @staticmethod
    def _summarize_tool(name: str, args: dict) -> str:
        """One-line summary of tool arguments."""
        if name in ("shell", "Bash"):
            cmd = args.get("command", "")
            if len(cmd) > 80:
                cmd = cmd[:77] + "..."
            return cmd
        if "path" in args:
            return args["path"]
        if "file_path" in args:
            return args["file_path"]
        if "pattern" in args:
            return args["pattern"]
        for v in args.values():
            if isinstance(v, str):
                if len(v) > 60:
                    return v[:57] + "..."
                return v
        return ""
