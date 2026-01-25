"""
Feedback Agent - Evaluates progress and provides feedback.

Responsibilities:
- Evaluate progress toward exploitation
- Calculate readiness score
- Provide feedback to Plan agent
- Decide whether to loop or go to Exploit
"""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response

console = Console()


class FeedbackAgent(BaseAgent):
    """
    Feedback Agent for PWN Solver.
    
    - Evaluate progress toward exploitation
    - Calculate readiness score
    - Provide feedback to Plan agent
    - Decide whether to loop or go to Exploit
    """
    
    MAX_ITERATIONS = 10
    READINESS_THRESHOLD = 0.8
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_iterations: int = 10,
        readiness_threshold: float = 0.8,
        **kwargs
    ):
        super().__init__(name="FeedbackAgent", model=model, provider=provider, **kwargs)
        self.max_iterations = max_iterations
        self.readiness_threshold = readiness_threshold
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run Feedback Agent."""
        console.print(Panel("Feedback Agent", style="bold green"))
        
        from Templates.prompt import build_messages
        
        # Get completed tasks and findings for this iteration
        parsing_output = state.get("parsing_output", {})
        key_findings = parsing_output.get("json", {}).get("key_findings", [])
        
        # Get recently completed tasks
        tasks = state.get("tasks", [])
        completed_tasks = [
            {"id": t["id"], "title": t.get("title", ""), "success": True}
            for t in tasks if t.get("status") == "done"
        ][-5:]  # Last 5 completed
        
        # Build messages
        messages = build_messages(
            agent="feedback",
            state=state,
            include_initial=False,
            completed_tasks=completed_tasks,
            key_findings=key_findings,
        )
        
        # Set system prompt and add user message
        if messages:
            self.set_system_prompt(messages[0]["content"])
            for msg in messages[1:]:
                if msg["role"] == "user":
                    self.add_user_message(msg["content"])
        
        console.print("[cyan]Calling LLM for feedback...[/cyan]")
        response_text = self.call_llm()
        
        # Parse response
        parsed = parse_json_response(response_text)
        
        state["feedback_output"] = {
            "agent": "feedback",
            "text": response_text,
            "json": parsed,
            "created_at": time.time(),
        }
        
        # Update exploit readiness
        readiness_score = parsed.get("readiness_score", 0.0)
        recommend_exploit = parsed.get("recommend_exploit", False)
        loop_continue = parsed.get("loop_continue", True)
        
        state["exploit_readiness"] = {
            "score": readiness_score,
            "components": parsed.get("completed_components", []),
            "missing": parsed.get("missing_components", []),
            "recommend_exploit": recommend_exploit,
        }
        
        # Update analysis readiness score
        if "analysis" in state:
            state["analysis"]["readiness_score"] = readiness_score
        
        # Show feedback
        console.print(f"[cyan]Readiness Score: {readiness_score:.1%}[/cyan]")
        console.print(f"[cyan]Recommend Exploit: {recommend_exploit}[/cyan]")
        console.print(f"[cyan]Continue Loop: {loop_continue}[/cyan]")
        
        if "feedback_to_plan" in parsed:
            console.print(Panel(
                parsed["feedback_to_plan"],
                title="Feedback to Plan",
                border_style="yellow"
            ))
        
        # Decision: continue loop or go to exploit
        iteration_count = state.get("iteration_count", 0)
        
        if iteration_count >= self.max_iterations:
            console.print(f"[yellow]Max iterations ({self.max_iterations}) reached[/yellow]")
            state["loop"] = False
        elif recommend_exploit and readiness_score >= self.readiness_threshold:
            console.print("[green]Ready for exploitation![/green]")
            state["loop"] = False
        elif not loop_continue:
            console.print("[yellow]Loop terminated by feedback[/yellow]")
            state["loop"] = False
        else:
            state["loop"] = True
        
        return state
