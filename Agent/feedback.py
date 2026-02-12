"""
Feedback Agent - Evaluates progress and provides feedback.

Responsibilities:
- Code-based readiness score computation (deterministic)
- LLM for qualitative feedback (what to do next)
- Decide whether to loop or go to Exploit
"""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response
from Templates.prompt import build_messages
from LangGraph.state import compute_readiness

console = Console()


class FeedbackAgent(BaseAgent):
    """
    Feedback Agent for PWN Solver.

    - Code-based readiness score (not LLM-dependent)
    - LLM provides qualitative feedback only
    - Decide whether to loop or go to Exploit
    """

    MAX_ITERATIONS = 10
    READINESS_THRESHOLD = 0.5

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_iterations: int = 10,
        readiness_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(name="FeedbackAgent", model=model, provider=provider, **kwargs)
        self.max_iterations = max_iterations
        self.readiness_threshold = readiness_threshold

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run Feedback Agent."""
        console.print(Panel("Feedback Agent", style="bold green"))

        self.clear_history()

        # --- Step 1: Code-based readiness score ---
        analysis = state.get("analysis", {})
        runs = state.get("runs", [])
        code_score, completed, missing = compute_readiness(analysis, runs=runs)

        console.print(f"[cyan]Code-based Readiness: {code_score:.0%}[/cyan]")
        if completed:
            console.print(f"[green]  Completed: {', '.join(completed)}[/green]")
        if missing:
            console.print(f"[yellow]  Missing: {', '.join(missing)}[/yellow]")

        # --- Step 2: LLM for qualitative feedback ---
        parsing_output = state.get("parsing_output", {})
        key_findings = parsing_output.get("json", {}).get("key_findings", [])

        tasks = state.get("tasks", [])
        completed_tasks = [
            {"id": t["id"], "title": t.get("title", ""), "success": True}
            for t in tasks if t.get("status") == "done"
        ][-5:]

        messages = build_messages(
            agent="feedback",
            state=state,
            include_initial=False,
            completed_tasks=completed_tasks,
            key_findings=key_findings,
        )

        if messages:
            self.set_system_prompt(messages[0]["content"])
            for msg in messages[1:]:
                if msg["role"] == "user":
                    self.add_user_message(msg["content"])

        console.print("[cyan]Calling LLM for qualitative feedback...[/cyan]")
        response_text = self.call_llm()

        parsed = parse_json_response(response_text)

        state["feedback_output"] = {
            "agent": "feedback",
            "text": response_text,
            "json": parsed,
            "created_at": time.time(),
        }

        # --- Step 3: Use CODE score, not LLM score ---
        readiness_score = code_score  # Override LLM score

        # LLM can suggest exploit but we gate on code score
        llm_recommend = parsed.get("recommend_exploit", False)
        llm_loop = parsed.get("loop_continue", True)

        # Recommend exploit if code score meets threshold
        recommend_exploit = readiness_score >= self.readiness_threshold
        # Also allow if LLM recommends and score is close
        if llm_recommend and readiness_score >= (self.readiness_threshold - 0.15):
            recommend_exploit = True

        state["exploit_readiness"] = {
            "score": readiness_score,
            "components": completed,
            "missing": missing,
            "recommend_exploit": recommend_exploit,
        }

        # Update analysis readiness score
        if "analysis" in state:
            state["analysis"]["readiness_score"] = readiness_score

        # Show feedback
        console.print(f"[cyan]Final Readiness Score: {readiness_score:.0%} (code-based)[/cyan]")
        console.print(f"[cyan]Recommend Exploit: {recommend_exploit}[/cyan]")

        if "feedback_to_plan" in parsed:
            console.print(Panel(
                parsed["feedback_to_plan"],
                title="Feedback to Plan",
                border_style="yellow"
            ))

        # --- Step 4: Decision ---
        iteration_count = state.get("iteration_count", 0)

        if recommend_exploit:
            console.print("[green]Ready for exploitation![/green]")
            state["loop"] = False
        else:
            # Always continue until max iterations or exploit ready
            if missing:
                console.print(f"[yellow]  Still missing: {', '.join(missing)} — continuing[/yellow]")
                if "feedback_to_plan" not in parsed or not parsed["feedback_to_plan"]:
                    parsed["feedback_to_plan"] = ""
                parsed["feedback_to_plan"] += (
                    f"\n\n[AUTO] Focus on resolving missing items: {', '.join(missing)}"
                )
                state["feedback_output"]["json"] = parsed
            state["loop"] = True

        return state
