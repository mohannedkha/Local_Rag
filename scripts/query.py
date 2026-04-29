#!/usr/bin/env python3
"""Interactive CLI for querying the RAG system.

Usage:
    python scripts/query.py                         # interactive REPL
    python scripts/query.py --question "What is X?" # single-shot
    python scripts/query.py --session my_session    # resume saved session
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.rag.factory import build_pipeline

console = Console()


def run_repl(pipeline, session_path: str | None):
    console.print(Panel("[bold cyan]Local RAG System[/bold cyan]\nType [bold]exit[/bold] or [bold]quit[/bold] to stop. [bold]/clear[/bold] resets conversation history.", expand=False))

    if session_path and Path(session_path).exists():
        pipeline.memory.load_session(session_path)
        console.print(f"[dim]Session restored from {session_path}[/dim]")

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "/clear":
            pipeline.memory._short_term.clear()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        console.print("\n[bold blue]Assistant:[/bold blue]")
        collected: list[str] = []
        for token in pipeline.query(user_input, stream=True):
            print(token, end="", flush=True)
            collected.append(token)
        print()

    if session_path:
        pipeline.memory.save_session(session_path)
        console.print(f"[dim]Session saved to {session_path}[/dim]")

    console.print("\n[dim]Goodbye.[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Query the local RAG system")
    parser.add_argument("--question", "-q", help="Single question (non-interactive)")
    parser.add_argument("--session", "-s", help="Path to session JSON file for persistence")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    pipeline = build_pipeline(args.config)

    if not pipeline.llm.is_available():
        console.print(f"[red]Ollama model '{pipeline.llm.model}' not available.[/red]")
        console.print("Run: [bold]ollama pull " + pipeline.llm.model + "[/bold]")
        sys.exit(1)

    if args.question:
        response = pipeline.query(args.question, stream=False)
        console.print(Markdown(response))
    else:
        run_repl(pipeline, args.session)


if __name__ == "__main__":
    main()
