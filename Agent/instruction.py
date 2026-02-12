"""
Instruction Agent - Converts tasks to tool calls and executes them.

Responsibilities:
- Select pending tasks based on priority
- Convert tasks to specific tool calls
- Execute tools and collect results
"""

import time
import hashlib
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response
from Templates.prompt import build_messages
from LangGraph.state import mark_task_status, add_run, TaskRun
from Tool.tool import Tool

console = Console()


def compute_cmd_hash(tool_name: str, args: Dict[str, Any]) -> str:
    """Compute hash for command deduplication."""
    import json
    cmd_str = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
    return hashlib.md5(cmd_str.encode()).hexdigest()[:12]


class InstructionAgent(BaseAgent):
    """
    Instruction Agent for PWN Solver.
    
    - Select pending tasks based on priority
    - Convert tasks to tool calls
    - Execute tools and collect results
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name="InstructionAgent", model=model, provider=provider, **kwargs)
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run Instruction Agent."""
        console.print(Panel("Instruction Agent", style="bold cyan"))

        self.clear_history()

        binary_path = state.get("binary_path", "")
        if not binary_path:
            console.print("[red]No binary path set[/red]")
            return state
        
        # Initialize tool
        try:
            tool = Tool(binary_path=binary_path)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return state
        
        # Build messages
        messages = build_messages(
            agent="instruction",
            state=state,
            include_initial=False,
        )
        
        # Set system prompt and add user message
        if messages:
            self.set_system_prompt(messages[0]["content"])
            for msg in messages[1:]:
                if msg["role"] == "user":
                    self.add_user_message(msg["content"])
        
        console.print("[cyan]Calling LLM for instruction...[/cyan]")
        response_text = self.call_llm()
        
        # Parse response
        parsed = parse_json_response(response_text)
        
        # Execute commands (parallel for independent tools, sequential for state-mutating ones)
        executions = []
        commands = parsed.get("commands", [])
        seen_hashes = state.get("seen_cmd_hashes", [])

        # Separate state-mutating tools from parallelizable ones
        STATE_MUTATING_TOOLS = {"Pwninit", "Run", "Pwndbg"}
        parallel_cmds = []
        sequential_cmds = []

        for cmd in commands:
            tool_name = cmd.get("tool", "")
            args = cmd.get("args", {})
            cmd_hash = compute_cmd_hash(tool_name, args)
            if cmd_hash in seen_hashes:
                console.print(f"[yellow]Skipping duplicate: {tool_name}[/yellow]")
                continue
            seen_hashes.append(cmd_hash)

            if tool_name in STATE_MUTATING_TOOLS:
                sequential_cmds.append(cmd)
            else:
                parallel_cmds.append(cmd)

        # Execute parallelizable tools concurrently
        if parallel_cmds:
            console.print(f"[cyan]Executing {len(parallel_cmds)} tools in parallel...[/cyan]")
            with ThreadPoolExecutor(max_workers=min(len(parallel_cmds), 4)) as executor:
                future_to_cmd = {}
                for cmd in parallel_cmds:
                    tool_name = cmd.get("tool", "")
                    args = cmd.get("args", {})
                    console.print(f"[green]  → {tool_name}({args})[/green]")
                    future = executor.submit(self._execute_tool, tool, tool_name, args, state)
                    future_to_cmd[future] = cmd

                for future in as_completed(future_to_cmd):
                    cmd = future_to_cmd[future]
                    result = future.result()
                    result["task_id"] = cmd.get("task_id", "")
                    result["purpose"] = cmd.get("purpose", "")
                    executions.append(result)

                    stdout = result.get("stdout", "")
                    stdout_preview = stdout[:500] + "..." if len(stdout) > 500 else stdout
                    console.print(Panel(
                        stdout_preview,
                        title=f"{cmd.get('tool', '')} Output",
                        border_style="green" if result.get("success") else "red"
                    ))
                    if cmd.get("task_id"):
                        mark_task_status(state, cmd["task_id"], "in_progress")

        # Execute state-mutating tools sequentially
        for cmd in sequential_cmds:
            tool_name = cmd.get("tool", "")
            args = cmd.get("args", {})
            task_id = cmd.get("task_id", "")
            purpose = cmd.get("purpose", "")

            console.print(f"[green]Executing: {tool_name}({args})[/green]")
            result = self._execute_tool(tool, tool_name, args, state)
            result["task_id"] = task_id
            result["purpose"] = purpose
            executions.append(result)

            stdout = result.get("stdout", "")
            stdout_preview = stdout[:500] + "..." if len(stdout) > 500 else stdout
            console.print(Panel(
                stdout_preview,
                title=f"{tool_name} Output",
                border_style="green" if result.get("success") else "red"
            ))
            if task_id:
                mark_task_status(state, task_id, "in_progress")
        
        state["seen_cmd_hashes"] = seen_hashes
        
        # Store execution results for Parsing node
        state["instruction_output"] = {
            "agent": "instruction",
            "text": response_text,
            "json": parsed,
            "executions": executions,
            "created_at": time.time(),
        }
        
        # Add runs to history
        for exec_result in executions:
            run: TaskRun = {
                "run_id": f"run_{int(time.time()*1000)}",
                "task_id": exec_result.get("task_id", ""),
                "commands": [f"{exec_result.get('tool', '')}({exec_result.get('args', {})})"],
                "success": exec_result.get("success", False),
                "stdout": exec_result.get("stdout", ""),
                "stderr": exec_result.get("stderr", ""),
                "started_at": time.time(),
                "finished_at": time.time(),
                "key_findings": [],
            }
            add_run(state, run)
        
        return state
    
    def _execute_tool(self, tool, tool_name: str, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return result."""
        result = {
            "tool": tool_name,
            "args": args,
            "stdout": "",
            "stderr": "",
            "success": False,
        }

        try:
            method = getattr(tool, tool_name, None)
            if method is None:
                result["stderr"] = f"Unknown tool: {tool_name}"
                return result

            output = method(**args)

            # Pwninit 특수 처리 - dict 반환
            if tool_name == "Pwninit" and isinstance(output, dict):
                result["stdout"] = output.get("output", "")
                result["success"] = output.get("success", False)

                # 패치된 바이너리로 경로 업데이트
                patched = output.get("patched_binary")
                if patched:
                    old_path = state.get("binary_path", "")
                    state["binary_path"] = patched
                    state["original_binary_path"] = old_path  # 원본 보관
                    tool.binary_path = patched  # Tool 인스턴스도 업데이트
                    result["stdout"] += f"\n\n[AUTO] binary_path updated: {patched}"
                    console.print(f"[green]Binary path updated to: {patched}[/green]")
            else:
                result["stdout"] = str(output) if output else ""
                result["success"] = not str(output).startswith("Error:")

        except Exception as e:
            result["stderr"] = str(e)

        return result
