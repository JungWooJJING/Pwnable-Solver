"""
Plan Agent - Analyzes state and creates tasks for exploitation.

Responsibilities:
- Analyze current Analysis Document
- Identify missing information ([NOT YET] sections)
- Create 2-4 prioritized tasks to fill gaps
- Update Analysis Document with findings
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response
from Templates.prompt import build_messages
from LangGraph.state import upsert_tasks, merge_analysis_updates

console = Console()


class PlanAgent(BaseAgent):
    """
    Plan Agent for PWN Solver.
    
    First iteration: Initialize analysis
    Subsequent iterations: Analyze feedback and create/update tasks
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name="PlanAgent", model=model, provider=provider, **kwargs)
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run Plan Agent."""
        console.print(Panel("Plan Agent", style="bold magenta"))

        # #2/#7: 실패 컨텍스트가 있으면 표시
        failure_reason = state.get("analysis_failure_reason", "")
        if failure_reason:
            console.print(Panel(
                f"[bold red]Re-analysis triggered:[/bold red]\n{failure_reason}",
                title="Exploit Failure Context",
                border_style="red"
            ))

        self.clear_history()

        # Build messages
        messages = build_messages(
            agent="plan",
            state=state,
            include_initial=(state.get("iteration_count", 0) <= 1),
        )
        
        # Set system prompt and add user message
        if messages:
            self.set_system_prompt(messages[0]["content"])
            for msg in messages[1:]:
                if msg["role"] == "user":
                    self.add_user_message(msg["content"])
        
        console.print("[cyan]Calling LLM for planning...[/cyan]")
        response_text = self.call_llm()
        
        # Parse response
        parsed = parse_json_response(response_text)
        
        # Update state with plan output
        state["plan_output"] = {
            "agent": "plan",
            "text": response_text,
            "json": parsed,
            "created_at": time.time(),
        }
        
        # Process tasks from plan
        if "tasks" in parsed:
            new_tasks = []
            for t in parsed["tasks"]:
                task = {
                    "id": t.get("id", f"task_{int(time.time()*1000)}"),
                    "title": t.get("title", ""),
                    "objective": t.get("objective", ""),
                    "actions_hint": t.get("actions_hint", []),
                    "tool_hint": t.get("tool_hint", ""),
                    "depends_on": t.get("depends_on", []),
                    "priority": t.get("priority", 0.5),
                    "status": "pending",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
                new_tasks.append(task)
            
            upsert_tasks(state, new_tasks)
            console.print(f"[green]Created/updated {len(new_tasks)} tasks[/green]")
        
        # Process analysis updates
        if "analysis_updates" in parsed:
            merge_analysis_updates(state, parsed["analysis_updates"])

        # #6: blockers/hypotheses 업데이트
        if "blockers" in parsed and isinstance(parsed["blockers"], list):
            existing = {b.get("question", "") for b in state.get("blockers", [])}
            for b in parsed["blockers"]:
                if b.get("question", "") not in existing:
                    state.setdefault("blockers", []).append(b)
            console.print(f"[yellow]Blockers: {len(state.get('blockers', []))} total[/yellow]")

        if "hypotheses" in parsed and isinstance(parsed["hypotheses"], list):
            existing = {h.get("statement", "") for h in state.get("hypotheses", [])}
            for h in parsed["hypotheses"]:
                if h.get("statement", "") not in existing:
                    state.setdefault("hypotheses", []).append(h)

        # 실패 컨텍스트를 Plan이 처리했으면 클리어
        if state.get("analysis_failure_reason"):
            state["analysis_failure_reason"] = ""
            state["exploit_failure_context"] = {}

        # Show reasoning
        if "reasoning" in parsed:
            console.print(Panel(parsed["reasoning"], title="Reasoning", border_style="blue"))

        return state
