#!/usr/bin/env python3
"""
Quick test script for CodingTaskRunner.

Usage:
    python task_run.py                          # run default task
    python task_run.py "Build a stack class"    # run custom task
    python task_run.py --quiet "Build X"        # no live output
"""
import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from agent import CodingTaskRunner, Config

DEFAULT_TASK = (
    "Write a Python module called `calculator.py` that implements a Calculator class "
    "with methods: add, subtract, multiply, divide. "
    "The divide method should raise ValueError on division by zero."
)

OUTPUT_ROOT = Path(__file__).parent / "output"


def main():
    parser = argparse.ArgumentParser(description="Test the CodingTaskRunner")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="Coding task description")
    parser.add_argument("--workspace", default=None, help="Override the files/ subfolder path")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max fix iterations")
    parser.add_argument("--clean", action="store_true", help="Remove output folder before starting")
    parser.add_argument("--quiet", action="store_true", help="No live output")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    args = parser.parse_args()

    # Each run gets its own timestamped folder under output/
    session_id = uuid.uuid4().hex[:12]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"{ts}_{session_id}"

    output_dir   = OUTPUT_ROOT / run_folder
    workspace    = Path(args.workspace) if args.workspace else output_dir / "files"
    logs_dir     = output_dir / "trajectory"
    memory_dir   = output_dir / "memory"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleaned output: {output_dir}")

    # Create all subdirectories up front
    workspace.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    if args.model:
        config = Config(
            base_url=config.base_url,
            api_key=config.api_key,
            model=args.model,
            stream=config.stream,
            max_tool_iterations=config.max_tool_iterations,
            context_limit=config.context_limit,
        )

    print(f"Model:      {config.model}")
    print(f"Output dir: {output_dir}")
    print(f"  files/      → {workspace}")
    print(f"  trajectory/ → {logs_dir}")
    print(f"  memory/     → {memory_dir}")
    print(f"Task:       {args.task[:80]}{'...' if len(args.task) > 80 else ''}")
    print()

    runner = CodingTaskRunner(
        workspace=workspace,
        config=config,
        max_fix_iterations=args.max_iterations,
        session_id=session_id,
        logs_dir=logs_dir,
        memory_log_dir=memory_dir,
    )

    result = runner.run(args.task, verbose=not args.quiet)

    # Print final summary regardless of verbose mode
    print()
    print("=" * 60)
    print(f"Status:     {result.status.upper()}")
    print(f"Iterations: {result.iterations}")
    print(f"Code files: {result.code_files}")
    print(f"Test files: {result.test_files}")
    print(f"Doc files:  {result.doc_files}")
    if result.error:
        print(f"Error:      {result.error}")
    print("=" * 60)

    # Show the generated files
    if result.code_files or result.test_files:
        print("\nGenerated files:")
        for f in result.code_files + result.test_files + result.doc_files:
            p = workspace / f
            if p.exists():
                size = p.stat().st_size
                lines = p.read_text().count("\n") + 1
                print(f"  {f:<40} {lines:>4} lines  {size:>6} bytes")

    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
