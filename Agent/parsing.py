"""
Parsing Agent - Parses execution results and extracts structured information.

Responsibilities:
- Parse raw output from tool executions
- Extract structured information for Analysis Document
- Identify vulnerabilities and key findings
"""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response

console = Console()


class ParsingAgent(BaseAgent):
    """
    Parsing Agent for PWN Solver.
    
    - Parse execution results from Instruction agent
    - Extract structured information for Analysis Document
    - Identify vulnerabilities and key findings
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name="ParsingAgent", model=model, provider=provider, **kwargs)
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run Parsing Agent."""
        console.print(Panel("Parsing Agent", style="bold yellow"))
        
        from Templates.prompt import build_messages
        from LangGraph.state import mark_task_status
        
        instruction_output = state.get("instruction_output", {})
        executions = instruction_output.get("executions", [])
        
        if not executions:
            console.print("[yellow]No executions to parse[/yellow]")
            return state
        
        # Build messages with execution results
        messages = build_messages(
            agent="parsing",
            state=state,
            include_initial=False,
            executions=executions,
        )
        
        # Set system prompt and add user message
        if messages:
            self.set_system_prompt(messages[0]["content"])
            for msg in messages[1:]:
                if msg["role"] == "user":
                    self.add_user_message(msg["content"])
        
        console.print("[cyan]Calling LLM for parsing...[/cyan]")
        response_text = self.call_llm()
        
        # Parse response
        parsed = parse_json_response(response_text)
        
        state["parsing_output"] = {
            "agent": "parsing",
            "text": response_text,
            "json": parsed,
            "created_at": time.time(),
        }
        
        # Merge analysis updates
        if "analysis_updates" in parsed:
            from LangGraph.state import merge_analysis_updates
            merge_analysis_updates(state, parsed["analysis_updates"])
            console.print("[green]Analysis document updated[/green]")

        # Store key findings
        key_findings = parsed.get("key_findings", [])
        if key_findings:
            console.print("[cyan]Key Findings:[/cyan]")
            for finding in key_findings:
                console.print(f"  • {finding}")

            # Update latest runs with findings
            runs = state.get("runs", [])
            if runs:
                runs[-1]["key_findings"] = key_findings

        # Mark tasks as done
        instruction_json = instruction_output.get("json", {})
        for task_id in instruction_json.get("selected_tasks", []):
            mark_task_status(state, task_id, "done")

        return state
