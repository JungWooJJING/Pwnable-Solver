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

# Ensure project root is in sys.path for package imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

    # Staged Exploit progress
    staged = state.get("staged_exploit", {})
    stages = staged.get("stages", [])
    if stages:
        console.print("\n[bold cyan]Staged Exploit Progress:[/bold cyan]")
        for s in stages:
            status = "[green]✓[/green]" if s.get("verified") else "[red]✗[/red]"
            refine_info = f" (refined {s.get('refinement_attempts', 0)}x)" if s.get("refinement_attempts", 0) > 0 else ""
            console.print(f"  {status} Stage {s.get('stage_index', 0)+1}: {s.get('stage_id', '?')} — {s.get('description', '')}{refine_info}")

    # Exploit output
    exploit_output = state.get("exploit_output", {})
    has_exploit = exploit_output.get("json", {}).get("exploit_code") or any(s.get("code") for s in stages)
    if has_exploit:
        console.print("[bold green]✓ Exploit generated![/bold green]")
        binary_path = state.get("binary_path", "")
        if binary_path:
            out_dir = Path(__file__).resolve().parent / "Challenge" / _slug(Path(binary_path).stem)
            console.print(f"[dim]Saved to: {out_dir}/exploit.py[/dim]")

        # Exploit verification status
        attempts = state.get("exploit_attempts", 0)
        verified = state.get("exploit_verified", False)

        if verified:
            console.print(f"[bold green]✓ Exploit verified! (attempts: {attempts})[/bold green]")
        elif attempts > 0:
            max_attempts = state.get("max_exploit_attempts", 3)
            console.print(f"[yellow]✗ Exploit verification failed ({attempts}/{max_attempts} attempts)[/yellow]")
            if state.get("exploit_error"):
                error_preview = state["exploit_error"][:200]
                if len(state["exploit_error"]) > 200:
                    error_preview += "..."
                console.print(f"[dim]Last error: {error_preview}[/dim]")

    # Flag
    if state.get("flag_detected"):
        console.print(f"[bold green]FLAG: {state.get('detected_flag')}[/bold green]")


def ask_continue(iteration: int, state: SolverState) -> bool:
    """Ask user if they want to continue after certain iterations"""
    readiness = state.get("exploit_readiness", {})
    score = readiness.get("score", 0.0)

    console.print(Panel(
        f"[yellow]Iteration {iteration} completed[/yellow]\n"
        f"Readiness Score: {score:.1%}\n"
        f"Tasks Done: {len([t for t in state.get('tasks', []) if t.get('status') == 'done'])}/{len(state.get('tasks', []))}",
        title="Progress Check",
        style="yellow"
    ))

    console.print("[bold]Continue solving? (y/n/s)[/bold]")
    console.print("  [green]y[/green] - Continue")
    console.print("  [red]n[/red] - Stop and show summary")
    console.print("  [cyan]s[/cyan] - Skip to exploit generation")

    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if choice in ("y", "yes", ""):
            return True
        elif choice in ("n", "no"):
            return False
        elif choice in ("s", "skip"):
            # Force exploit generation
            state["loop"] = False
            state["exploit_readiness"]["recommend_exploit"] = True
            return True
        else:
            console.print("[yellow]Please enter y, n, or s[/yellow]")


def ask_after_exploit(state: SolverState) -> str:
    """Ask user what to do after exploit verification"""
    verified = state.get("exploit_verified", False)
    flag_detected = state.get("flag_detected", False)
    attempts = state.get("exploit_attempts", 0)
    max_attempts = state.get("max_exploit_attempts", 3)

    if flag_detected:
        console.print(Panel(
            f"[bold green]FLAG FOUND: {state.get('detected_flag')}[/bold green]",
            title="Success!",
            style="green"
        ))
    elif verified:
        # Show staged progress if available
        staged = state.get("staged_exploit", {})
        stages = staged.get("stages", [])
        stage_info = ""
        if stages:
            verified_count = sum(1 for s in stages if s.get("verified"))
            stage_info = f"\nStages: {verified_count}/{len(stages)} verified"

        console.print(Panel(
            f"[green]Exploit verified (shell access detected)[/green]\n"
            f"Attempts: {attempts}{stage_info}",
            title="Exploit Success",
            style="green"
        ))
    else:
        console.print(Panel(
            f"[yellow]Exploit verification failed[/yellow]\n"
            f"Attempts: {attempts}/{max_attempts}",
            title="Exploit Failed",
            style="yellow"
        ))

    console.print("[bold]What would you like to do?[/bold]")
    console.print("  [green]c[/green] - Continue (back to analysis)")
    console.print("  [cyan]r[/cyan] - Retry exploit generation")
    console.print("  [red]s[/red] - Stop and show summary")

    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "stop"

        if choice in ("c", "continue"):
            return "continue"
        elif choice in ("r", "retry"):
            return "retry"
        elif choice in ("s", "stop", ""):
            return "stop"
        else:
            console.print("[yellow]Please enter c, r, or s[/yellow]")


def run_solver(state: SolverState, ask_interval: int = 5) -> SolverState:
    """
    Run the PWN solver workflow

    Args:
        state: Initial solver state
        ask_interval: Ask user every N iterations (0 = never ask)
    """
    console.print(Panel("Starting PWN Solver", style="bold magenta"))

    # Create and run workflow
    app = create_app()

    # Stream execution with updates
    console.print("[cyan]Running workflow...[/cyan]\n")

    final_state = None
    last_asked_iteration = 0

    restart_workflow = False

    while True:
        restart_workflow = False

        for output in app.stream(state, config={"recursion_limit": 9999}):
            if restart_workflow:
                break

            # output is a dict with node_name: state
            for node_name, node_state in output.items():
                console.print(f"[dim]Completed: {node_name}[/dim]")
                final_state = node_state
                state = node_state  # Update state for next iteration

                # Check if we should ask user to continue (after feedback)
                if ask_interval > 0 and node_name == "feedback":
                    iteration = node_state.get("iteration_count", 0)

                    # Ask every ask_interval iterations
                    if iteration > 0 and iteration % ask_interval == 0 and iteration != last_asked_iteration:
                        last_asked_iteration = iteration

                        if not ask_continue(iteration, node_state):
                            console.print("[yellow]Stopping by user request[/yellow]")
                            return node_state

                # Check after exploit verification (not in auto mode)
                if ask_interval > 0 and node_name == "verify":
                    choice = ask_after_exploit(node_state)

                    if choice == "stop":
                        console.print("[yellow]Stopping by user request[/yellow]")
                        return node_state
                    elif choice == "continue":
                        # Go back to analysis loop
                        console.print("[cyan]Returning to analysis...[/cyan]")
                        state["loop"] = True
                        state["exploit_readiness"]["recommend_exploit"] = False
                        state["exploit_attempts"] = 0
                        state["exploit_verified"] = False
                        restart_workflow = True
                        break
                    elif choice == "retry":
                        # Retry exploit generation
                        console.print("[cyan]Retrying exploit generation...[/cyan]")
                        state["loop"] = False
                        state["exploit_readiness"]["recommend_exploit"] = True
                        state["exploit_attempts"] = 0
                        state["exploit_verified"] = False
                        state["exploit_path"] = ""
                        restart_workflow = True
                        break

        if not restart_workflow:
            # Stream completed normally
            break

        # Restart workflow with updated state
        app = create_app()

    return final_state or state


def setup_docker_env(binary_path: str, port: int = 1337) -> bool:
    """Setup Docker environment for exploit testing"""
    from Tool.tool import Tool

    console.print(Panel("Setting up Docker Environment", style="bold blue"))

    try:
        tool = Tool(binary_path=binary_path)
        result = tool.Docker_setup(port=port)
        console.print(result)
        return "[SUCCESS]" in result
    except Exception as e:
        console.print(f"[red]Docker setup failed: {e}[/red]")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PWN Solver - Multi-Agent CTF Solver")
    parser.add_argument("--binary", "-b", help="Path to binary file")
    parser.add_argument("--title", "-t", help="Challenge title")
    parser.add_argument("--description", "-d", help="Challenge description (direct mode)")
    parser.add_argument("--flag-format", "-f", help="Flag format", default="FLAG{...}")
    parser.add_argument("--docker", action="store_true", help="Auto-setup Docker environment for testing")
    parser.add_argument("--docker-port", type=int, default=1337, help="Docker port (default: 1337)")
    parser.add_argument("--ask-interval", type=int, default=5, help="Ask to continue every N iterations (0=never)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max exploit verification attempts (default: 3)")
    parser.add_argument("--auto", action="store_true", help="Fully automatic mode (no user prompts during solving)")
    args = parser.parse_args()

    # Check environment
    if not check_env():
        sys.exit(1)

    # Initialize state
    state = init_state()
    state["max_exploit_attempts"] = args.max_attempts

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

    # Setup Docker if requested
    if args.docker and binary_path:
        if not setup_docker_env(binary_path, args.docker_port):
            console.print("[yellow]Warning: Docker setup failed, continuing without Docker[/yellow]")
        else:
            console.print(f"[green]Docker environment ready on port {args.docker_port}[/green]")
            state["docker_port"] = args.docker_port

    # Run solver
    try:
        ask_interval = 0 if args.auto else args.ask_interval
        final_state = run_solver(state, ask_interval=ask_interval)
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
