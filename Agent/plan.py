"""
Plan Agent - Analyzes state and creates tasks for exploitation.

Responsibilities:
- Analyze current Analysis Document
- Identify missing information ([NOT YET] sections)
- Create 2-4 prioritized tasks to fill gaps
- Update Analysis Document with findings
"""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response

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
        
        # Import here to avoid circular imports
        from Templates.prompt import build_messages
        from LangGraph.state import upsert_tasks
        
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
            self._merge_analysis_updates(state, parsed["analysis_updates"])
        
        # Show reasoning
        if "reasoning" in parsed:
            console.print(Panel(parsed["reasoning"], title="Reasoning", border_style="blue"))
        
        return state
    
    def _merge_analysis_updates(self, state: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Merge analysis updates into state."""
        from LangGraph.state import init_analysis
        
        analysis = state.get("analysis", init_analysis())
        
        # Checksec
        if "checksec" in updates:
            analysis["checksec"].update(updates["checksec"])
        
        # Decompile
        if "decompile" in updates:
            if updates["decompile"].get("done"):
                analysis["decompile"]["done"] = True
            
            new_funcs = updates["decompile"].get("functions", [])
            existing_names = {f["name"] for f in analysis["decompile"].get("functions", [])}
            
            for func in new_funcs:
                if func.get("name") not in existing_names:
                    analysis["decompile"]["functions"].append(func)
        
        # Disasm
        if "disasm" in updates:
            analysis["disasm"].update(updates["disasm"])
        
        # Vulnerabilities (append, don't overwrite)
        if "vulnerabilities" in updates:
            existing_vulns = {(v.get("type"), v.get("function")) for v in analysis.get("vulnerabilities", [])}
            
            for vuln in updates["vulnerabilities"]:
                key = (vuln.get("type"), vuln.get("function"))
                if key not in existing_vulns:
                    analysis["vulnerabilities"].append(vuln)
        
        # Strategy
        if "strategy" in updates and updates["strategy"]:
            analysis["strategy"] = updates["strategy"]
        
        # Libc
        if "libc" in updates:
            analysis["libc"].update(updates["libc"])
        
        # Gadgets (append)
        if "gadgets" in updates:
            existing_addrs = {g.get("address") for g in analysis.get("gadgets", [])}
            
            for gadget in updates["gadgets"]:
                if gadget.get("address") not in existing_addrs:
                    analysis["gadgets"].append(gadget)
        
        # Boolean flags
        for flag in ["leak_primitive", "control_hijack", "payload_ready"]:
            if flag in updates:
                analysis[flag] = updates[flag]
        
        state["analysis"] = analysis
