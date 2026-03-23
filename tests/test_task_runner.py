"""
Tests for CodingTaskRunner (agent/task_runner.py).

Mocks LLMClient and SessionRecordingService so no real API calls happen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import CodingTaskRunner, TaskResult, TurnEvent, Config
from agent.client import ChatResponse, ToolCall
from agent.sandbox import SandboxedRegistry
from agent.telemetry import TokenUsageStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_response(text: str = "", tool_calls: list[ToolCall] | None = None) -> ChatResponse:
    resp = ChatResponse()
    resp.content = text
    resp.tool_calls = tool_calls or []
    resp.usage = TokenUsageStats(input_tokens=10, output_tokens=5, total_tokens=15)
    return resp


def make_write_file_call(filename: str, content: str) -> ToolCall:
    """Create a tool call that writes a file."""
    return ToolCall(
        id=f"tc_{filename}",
        name="write_file",
        arguments={"path": filename, "content": content},
    )


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_defaults(self):
        r = TaskResult(task="test")
        assert r.status == ""
        assert r.task == "test"
        assert r.iterations == 0
        assert r.code_files == []
        assert r.test_files == []
        assert r.test_output == ""

    def test_fields(self):
        r = TaskResult(
            task="build thing",
            status="passed",
            iterations=2,
            code_files=["thing.py"],
            test_files=["test_thing.py"],
            test_output="1 passed",
        )
        assert r.status == "passed"
        assert r.iterations == 2


# ---------------------------------------------------------------------------
# CodingTaskRunner — construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_creates_workspace(self, tmp_path):
        ws = tmp_path / "new_project"
        runner = CodingTaskRunner(workspace=ws)
        assert ws.exists()
        assert runner.workspace == ws.resolve()

    def test_default_config(self, tmp_path):
        runner = CodingTaskRunner(workspace=tmp_path)
        assert runner.config is not None

    def test_custom_config(self, tmp_path):
        cfg = Config(model="test-model", api_key="test")
        runner = CodingTaskRunner(workspace=tmp_path, config=cfg)
        assert runner.config.model == "test-model"

    def test_custom_max_iterations(self, tmp_path):
        runner = CodingTaskRunner(workspace=tmp_path, max_fix_iterations=3)
        assert runner.max_fix_iterations == 3

    def test_custom_test_command(self, tmp_path):
        runner = CodingTaskRunner(workspace=tmp_path, test_command="pytest -q")
        assert runner.test_command == "pytest -q"


# ---------------------------------------------------------------------------
# CodingTaskRunner — _make_agent
# ---------------------------------------------------------------------------

class TestMakeAgent:
    def test_agent_uses_sandboxed_registry(self, tmp_path):
        runner = CodingTaskRunner(
            workspace=tmp_path,
            config=Config(model="test", api_key="test"),
        )
        with (
            patch("agent.task_runner.Agent") as mock_agent_cls,
            patch("agent.agent.SessionRecordingService"),
            patch("agent.agent.LLMClient"),
        ):
            mock_agent_cls.return_value = MagicMock()
            runner._make_agent()
            # Agent should have been called with a SandboxedRegistry
            call_kwargs = mock_agent_cls.call_args
            registry = call_kwargs.kwargs.get("registry") or call_kwargs[1].get("registry")
            assert isinstance(registry, SandboxedRegistry)


# ---------------------------------------------------------------------------
# CodingTaskRunner — run_stream phases
# ---------------------------------------------------------------------------

class TestRunStream:
    def _make_runner(self, tmp_path, responses, test_results=None):
        """
        Create a runner with mocked LLM and test execution.

        responses: list of ChatResponse to return from LLM
        test_results: list of (passed: bool, output: str) for each test run
        """
        runner = CodingTaskRunner(
            workspace=tmp_path,
            config=Config(model="test", api_key="test", stream=False),
            max_fix_iterations=3,
            max_review_iterations=2,
        )

        # Patch _run_tests to return controlled results
        if test_results is None:
            test_results = [(True, "1 passed")]
        test_iter = iter(test_results)
        runner._run_tests = lambda: next(test_iter)

        # Default: review always passes (tests that care override this)
        runner._check_review_verdict = lambda: True

        # Patch _make_agent to return a mocked agent
        original_make = runner._make_agent

        def patched_make():
            with (
                patch("agent.agent.SessionRecordingService") as mock_rec_cls,
                patch("agent.agent.LLMClient") as mock_client_cls,
            ):
                mock_rec_cls.return_value = MagicMock()
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.chat.side_effect = list(responses)
                agent = original_make()
                agent.client = mock_client
                agent.recorder = mock_rec_cls.return_value
                return agent

        runner._make_agent = patched_make
        return runner

    def test_emits_phase_events(self, tmp_path):
        """The runner emits write_code, write_tests, and run_tests phase events."""
        runner = self._make_runner(tmp_path, [
            make_response("Task analysed."),  # task_intake
            make_response("Repo scanned."),   # repo_recon
            make_response("Plan ready."),     # plan_design
            make_response("Code written."),   # write_code
            make_response("Tests written."),  # write_tests
            make_response("Review done."),    # review
            make_response("Docs written."),   # write_docs
        ])
        events = list(runner.run_stream("build X"))
        phase_events = [e for e in events if e.type == "phase"]
        phase_names = [e.data for e in phase_events]
        assert "write_code" in phase_names
        assert "write_tests" in phase_names
        assert any(p.startswith("run_tests") for p in phase_names)

    def test_passes_on_first_try(self, tmp_path):
        """When tests pass immediately, status is 'passed' with 1 iteration."""
        runner = self._make_runner(
            tmp_path,
            [
                make_response("Task analysed."),  # task_intake
                make_response("Repo scanned."),   # repo_recon
                make_response("Plan ready."),     # plan_design
                make_response("Code."),           # write_code
                make_response("Tests."),          # write_tests
                make_response("Review done."),    # review
                make_response("Docs written."),   # write_docs
            ],
            test_results=[(True, "1 passed")],
        )
        events = list(runner.run_stream("build X"))
        result_events = [e for e in events if e.type == "result"]
        assert result_events
        result = result_events[0].data
        assert result.status == "passed"
        assert result.iterations == 1

    def test_fix_and_retry(self, tmp_path):
        """When tests fail then pass, the fix cycle works."""
        runner = self._make_runner(
            tmp_path,
            [
                make_response("Task analysed."),   # task_intake
                make_response("Repo scanned."),    # repo_recon
                make_response("Plan ready."),      # plan_design
                make_response("Code."),            # write_code
                make_response("Tests."),           # write_tests
                make_response("Fixed the code."),  # fix_1
                make_response("Review done."),     # review
                make_response("Docs written."),    # write_docs
            ],
            test_results=[
                (False, "FAILED test_foo - AssertionError"),  # iter 1
                (True, "1 passed"),                            # iter 2
            ],
        )
        events = list(runner.run_stream("build X"))
        result = next(e.data for e in events if e.type == "result")
        assert result.status == "passed"
        assert result.iterations == 2

    def test_exhausts_iterations(self, tmp_path):
        """When tests never pass, status is 'failed' after max iterations."""
        runner = self._make_runner(
            tmp_path,
            [
                make_response("Task analysed."),  # task_intake
                make_response("Repo scanned."),   # repo_recon
                make_response("Plan ready."),     # plan_design
                make_response("Code."),           # write_code
                make_response("Tests."),          # write_tests
                make_response("Fix 1."),          # fix_1
                make_response("Fix 2."),          # fix_2
                make_response("Fix 3."),          # fix_3
                make_response("Review done."),    # review
                make_response("Docs written."),   # write_docs
            ],
            test_results=[
                (False, "FAILED"),
                (False, "FAILED"),
                (False, "FAILED"),
            ],
        )
        events = list(runner.run_stream("build X"))
        result = next(e.data for e in events if e.type == "result")
        assert result.status == "failed"
        assert result.iterations == 3

    def test_review_failure_overrides_passed_tests(self, tmp_path):
        """A final failed review must not leak out as status='passed'."""
        runner = self._make_runner(
            tmp_path,
            [
                make_response("Task intake."),
                make_response("Repo."),
                make_response("Plan."),
                make_response("Code."),
                make_response("Tests."),
                make_response("Review failed."),
                make_response("Tests rewrite."),
                make_response("Review failed again."),
                make_response("Docs."),
            ],
            test_results=[
                (True, "1 passed"),
                (True, "1 passed"),
            ],
        )

        review_states = iter([False, False])
        runner._check_review_verdict = lambda: next(review_states)

        events = list(runner.run_stream("build X"))
        result = next(e.data for e in events if e.type == "result")
        review_events = [e for e in events if e.type == "review_result"]

        assert result.status == "failed"
        assert len(review_events) == 2
        assert all(event.data["passed"] is False for event in review_events)

    def test_test_result_events_emitted(self, tmp_path):
        """test_result events contain pass/fail info and output."""
        runner = self._make_runner(
            tmp_path,
            [
                make_response("Task analysed."),  # task_intake
                make_response("Repo scanned."),   # repo_recon
                make_response("Plan ready."),     # plan_design
                make_response("Code."),           # write_code
                make_response("Tests."),          # write_tests
                make_response("Review done."),    # review
                make_response("Docs written."),   # write_docs
            ],
            test_results=[(True, "1 passed in 0.1s")],
        )
        events = list(runner.run_stream("build X"))
        test_events = [e for e in events if e.type == "test_result"]
        assert test_events
        assert test_events[0].data["passed"] is True
        assert "1 passed" in test_events[0].data["output"]

    def test_text_events_from_agent(self, tmp_path):
        """Text events from the underlying agent are yielded."""
        runner = self._make_runner(
            tmp_path,
            [
                make_response("Task analysed."),  # task_intake
                make_response("Repo scanned."),   # repo_recon
                make_response("Plan ready."),     # plan_design
                make_response("Hello from agent"),  # write_code
                make_response("Tests here"),      # write_tests
                make_response("Review done."),    # review
                make_response("Docs written."),   # write_docs
            ],
        )
        events = list(runner.run_stream("build X"))
        text_events = [e for e in events if e.type == "text"]
        assert text_events
        full_text = "".join(e.data for e in text_events)
        assert "Hello from agent" in full_text


# ---------------------------------------------------------------------------
# CodingTaskRunner — run (blocking)
# ---------------------------------------------------------------------------

class TestRunBlocking:
    def test_returns_task_result(self, tmp_path):
        runner = CodingTaskRunner(
            workspace=tmp_path,
            config=Config(model="test", api_key="test", stream=False),
            max_review_iterations=1,
        )
        runner._run_tests = lambda: (True, "1 passed")
        runner._check_review_verdict = lambda: True

        with (
            patch("agent.agent.SessionRecordingService") as mock_rec_cls,
            patch("agent.agent.LLMClient") as mock_client_cls,
        ):
            mock_rec_cls.return_value = MagicMock()
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.chat.side_effect = [
                make_response("Task analysed."),  # task_intake
                make_response("Repo scanned."),   # repo_recon
                make_response("Plan ready."),     # plan_design
                make_response("Code."),           # write_code
                make_response("Tests."),          # write_tests
                make_response("Review done."),    # review
                make_response("Docs written."),   # write_docs
            ]
            result = runner.run("build thing")

        assert isinstance(result, TaskResult)
        assert result.status == "passed"


# ---------------------------------------------------------------------------
# CodingTaskRunner — _find_files
# ---------------------------------------------------------------------------

class TestFindFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "test_main.py").write_text("def test_x(): pass")
        (tmp_path / "utils.py").write_text("y = 2")

        runner = CodingTaskRunner(workspace=tmp_path)
        code = runner._find_files("*.py", exclude_prefix="test_")
        tests = runner._find_files("test_*.py")

        assert "main.py" in code
        assert "utils.py" in code
        assert "test_main.py" not in code
        assert "test_main.py" in tests

    def test_finds_nested_files(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_app.py").write_text("def test_x(): pass")

        runner = CodingTaskRunner(workspace=tmp_path)
        code = runner._find_files("*.py", exclude_prefix="test_")
        tests = runner._find_files("test_*.py")

        assert any("app.py" in f for f in code)
        assert any("test_app.py" in f for f in tests)

    def test_excludes_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "main.cpython-311.pyc").write_text("")
        (tmp_path / "main.py").write_text("x = 1")

        runner = CodingTaskRunner(workspace=tmp_path)
        files = runner._find_files("*.py*", exclude_prefix="test_")
        assert all("__pycache__" not in f for f in files)


# ---------------------------------------------------------------------------
# CodingTaskRunner — _run_tests (real subprocess)
# ---------------------------------------------------------------------------

class TestRunTests:
    def test_passing_tests(self, tmp_path):
        (tmp_path / "test_ok.py").write_text("def test_pass(): assert True")
        runner = CodingTaskRunner(workspace=tmp_path)
        passed, output = runner._run_tests()
        assert passed is True
        assert "passed" in output

    def test_failing_tests(self, tmp_path):
        (tmp_path / "test_fail.py").write_text("def test_fail(): assert 1 == 2")
        runner = CodingTaskRunner(workspace=tmp_path)
        passed, output = runner._run_tests()
        assert passed is False
        assert "FAILED" in output or "failed" in output

    def test_no_tests_found(self, tmp_path):
        runner = CodingTaskRunner(workspace=tmp_path)
        passed, output = runner._run_tests()
        # pytest returns non-zero when no tests collected
        assert passed is False

    def test_custom_test_command(self, tmp_path):
        (tmp_path / "test_ok.py").write_text("def test_pass(): assert True")
        runner = CodingTaskRunner(workspace=tmp_path, test_command="python -m pytest -q 2>&1")
        passed, output = runner._run_tests()
        assert passed is True
