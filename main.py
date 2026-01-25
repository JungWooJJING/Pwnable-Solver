"""
PWN Solver - Multi-Agent CTF Solver

Usage:
    python main.py                    # Interactive mode
    python main.py --binary /path/to/vuln  # Direct mode
"""

import sys
import os
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from LangGraph.state import init_state, SolverState
from LangGraph.workflow import create_app

console = Console()

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def check_env() -> bool:
    """Check if required environment variables are set"""
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    if not has_openai and not has_gemini:
        console.print("[bold red]Error: No LLM API key found![/bold red]")
        console.print("Please set at least one of the following:")
        console.print("  export OPENAI_API_KEY=<your-key>")
        console.print("  export GEMINI_API_KEY=<your-key>")
        return False

    return True


def get_challenge_info() -> dict:
    """Interactive prompt to get challenge information"""
    console.print(Panel("Challenge Information", style="bold green"))

    # Title
    console.print("[blue]Enter the challenge title:[/blue]")
    title = input("> ").strip()

    # Description
    console.print("[blue]Enter the challenge description (Press <<<END>>> to finish):[/blue]")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if "<<<END>>>" in line:
            prefix = line.split("<<<END>>>", 1)[0]
            if prefix:
                lines.append(prefix)
            break
        lines.append(line)
    description = "\n".join(lines).rstrip("\n")

    # Flag format
    console.print("[blue]Enter the flag format (e.g., FLAG{...}):[/blue]")
    flag_format = input("> ").strip() 

    # Binary path
    console.print("[blue]Enter the binary file path (absolute path):[/blue]")
    binary_path = input("> ").strip()

    # Validate binary
    if binary_path and not Path(binary_path).exists():
        console.print(f"[yellow]Warning: Binary not found at {binary_path}[/yellow]")

    return {
        "title": title,
        "description": description,
        "flag_format": flag_format,
        "binary_path": binary_path,
    }


def display_summary(state: SolverState) -> None:
    """Display final summary of the solving session"""
    console.print(Panel("Session Summary", style="bold cyan"))

    # Challenge info
    challenge = state.get("challenge", {})
    console.print(f"[bold]Challenge:[/bold] {challenge.get('title', 'Unknown')}")
    console.print(f"[bold]Binary:[/bold] {state.get('binary_path', 'N/A')}")

    # Iterations
    console.print(f"[bold]Iterations:[/bold] {state.get('iteration_count', 0)}")

    # Tasks
    tasks = state.get("tasks", [])
    done_tasks = [t for t in tasks if t.get("status") == "done"]
    console.print(f"[bold]Tasks Completed:[/bold] {len(done_tasks)}/{len(tasks)}")

    # Exploit readiness
    readiness = state.get("exploit_readiness", {})
    score = readiness.get("score", 0.0)
    console.print(f"[bold]Exploit Readiness:[/bold] {score:.1%}")

    # Analysis summary
    analysis = state.get("analysis", {})

    table = Table(title="Analysis Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Checksec", "✓" if analysis.get("checksec", {}).get("done") else "✗")
    table.add_row("Decompile", "✓" if analysis.get("decompile", {}).get("done") else "✗")
    table.add_row("Vulnerabilities", str(len(analysis.get("vulnerabilities", []))))
    table.add_row("Gadgets", str(len(analysis.get("gadgets", []))))
    table.add_row("Libc", "✓" if analysis.get("libc", {}).get("detected") else "✗")

    console.print(table)

    # Exploit output
    exploit_output = state.get("exploit_output", {})
    if exploit_output.get("json", {}).get("exploit_code"):
        console.print("[bold green]✓ Exploit generated![/bold green]")
        binary_path = state.get("binary_path", "")
        if binary_path:
            out_dir = Path(__file__).resolve().parent / "Challenge" / _slug(Path(binary_path).stem)
            console.print(f"[dim]Saved to: {out_dir}/exploit.py[/dim]")

    # Flag
    if state.get("flag_detected"):
        console.print(f"[bold green]FLAG: {state.get('detected_flag')}[/bold green]")


def run_solver(state: SolverState) -> SolverState:
    """Run the PWN solver workflow"""
    console.print(Panel("Starting PWN Solver", style="bold magenta"))

    # Create and run workflow
    app = create_app()

    # Stream execution with updates
    console.print("[cyan]Running workflow...[/cyan]\n")

    final_state = None
    for output in app.stream(state):
        # output is a dict with node_name: state
        for node_name, node_state in output.items():
            console.print(f"[dim]Completed: {node_name}[/dim]")
            final_state = node_state

    return final_state or state


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PWN Solver - Multi-Agent CTF Solver")
    parser.add_argument("--binary", "-b", help="Path to binary file")
    parser.add_argument("--title", "-t", help="Challenge title")
    parser.add_argument("--description", "-d", help="Challenge description (direct mode)")
    parser.add_argument("--flag-format", "-f", help="Flag format", default="FLAG{...}")
    args = parser.parse_args()

    # Check environment
    if not check_env():
        sys.exit(1)

    # Initialize state
    state = init_state()

    # Get challenge info (interactive or from args)
    if args.binary:
        # Direct mode
        state["challenge"] = {
            "title": args.title or Path(args.binary).stem,
            "description": args.description or "",
            "flag_format": args.flag_format,
        }
        state["binary_path"] = args.binary
    else:
        # Interactive mode
        info = get_challenge_info()
        state["challenge"] = {
            "title": info["title"],
            "description": info["description"],
            "flag_format": info["flag_format"],
        }
        state["binary_path"] = info["binary_path"]

    # Validate binary exists
    binary_path = state.get("binary_path", "")
    if binary_path and not Path(binary_path).exists():
        console.print(f"[red]Error: Binary not found: {binary_path}[/red]")
        sys.exit(1)

    # Run solver
    try:
        final_state = run_solver(state)
        display_summary(final_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        display_summary(state)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
