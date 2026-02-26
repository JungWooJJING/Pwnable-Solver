"""
Parsing Agent - Parses execution results and extracts structured information.

Responsibilities:
- Parse raw output from tool executions
- Extract structured information for Analysis Document
- Identify vulnerabilities and key findings
- Deterministic parsing for checksec, ROPgadget, one_gadget (no LLM needed)
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent, parse_json_response
from Templates.prompt import build_messages
from LangGraph.state import mark_task_status, merge_analysis_updates, infer_readiness_from_key_findings

console = Console()


# =============================================================================
# Deterministic Parsers (no LLM needed)
# =============================================================================

def parse_checksec_output(raw: str) -> Dict[str, Any]:
    """
    Parse checksec output deterministically.

    Returns:
        {
            "done": True,
            "result": "NX enabled, No canary, No PIE, Partial RELRO",
            "nx": bool, "canary": bool, "pie": bool,
            "relro": "none" | "partial" | "full",
            "fortify": bool, "arch": str
        }
    """
    raw_lower = raw.lower()

    # NX
    nx = "nx enabled" in raw_lower and "nx disabled" not in raw_lower

    # Canary
    canary = "canary found" in raw_lower and "no canary" not in raw_lower

    # PIE
    pie = "pie enabled" in raw_lower and "no pie" not in raw_lower

    # RELRO
    if "full relro" in raw_lower:
        relro = "full"
    elif "partial relro" in raw_lower:
        relro = "partial"
    else:
        relro = "none"

    # Fortify
    fortify = "fortify" in raw_lower and "no fortify" not in raw_lower

    # Architecture
    arch = "unknown"
    arch_match = re.search(r'Arch:\s*(\S+)', raw, re.IGNORECASE)
    if arch_match:
        arch = arch_match.group(1)
    elif "amd64" in raw_lower or "x86-64" in raw_lower or "64-little" in raw_lower:
        arch = "amd64"
    elif "i386" in raw_lower or "x86" in raw_lower or "32-little" in raw_lower:
        arch = "i386"

    # Build summary string
    parts = []
    parts.append("NX enabled" if nx else "NX disabled")
    parts.append("Canary found" if canary else "No canary")
    parts.append("PIE enabled" if pie else "No PIE")
    parts.append(f"{relro.capitalize()} RELRO")
    result_str = ", ".join(parts)

    return {
        "done": True,
        "result": result_str,
        "nx": nx,
        "canary": canary,
        "pie": pie,
        "relro": relro,
        "fortify": fortify,
        "arch": arch,
    }


def parse_ropgadget_output(raw: str) -> List[Dict[str, str]]:
    """
    Parse ROPgadget output deterministically.

    Returns:
        [{"address": "0x401234", "instruction": "pop rdi ; ret"}, ...]
    """
    gadgets = []
    # ROPgadget format: 0x0000000000401234 : pop rdi ; ret
    pattern = re.compile(r'(0x[0-9a-fA-F]+)\s*:\s*(.+)')

    for line in raw.splitlines():
        line = line.strip()
        match = pattern.match(line)
        if match:
            addr = match.group(1)
            instruction = match.group(2).strip()
            # Skip very long gadgets (usually noise)
            if len(instruction) < 100:
                gadgets.append({
                    "address": addr,
                    "instruction": instruction,
                })

    return gadgets


def parse_one_gadget_output(raw: str) -> List[Dict[str, Any]]:
    """
    Parse one_gadget output deterministically.

    Returns:
        [{"offset": "0x4f3d5", "constraints": ["rax == NULL", "[rsp+0x70] == NULL"]}, ...]
    """
    results = []
    current_offset = None
    current_constraints = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            if current_offset:
                results.append({
                    "offset": current_offset,
                    "constraints": current_constraints,
                })
                current_offset = None
                current_constraints = []
            continue

        # Offset line: 0x4f3d5 execve("/bin/sh", rsp+0x40, environ)
        offset_match = re.match(r'(0x[0-9a-fA-F]+)\s+execve', line)
        if offset_match:
            if current_offset:
                results.append({
                    "offset": current_offset,
                    "constraints": current_constraints,
                })
            current_offset = offset_match.group(1)
            current_constraints = []
            continue

        # Constraint line: constraints:
        if line.lower().startswith("constraints:"):
            continue

        # Actual constraint: rax == NULL, [rsp+0x70] == NULL, etc.
        if current_offset and ("==" in line or "is" in line or "writable" in line):
            current_constraints.append(line)

    # Don't forget last entry
    if current_offset:
        results.append({
            "offset": current_offset,
            "constraints": current_constraints,
        })

    return results


# =============================================================================
# Parsing Agent
# =============================================================================

class ParsingAgent(BaseAgent):
    """
    Parsing Agent for PWN Solver.

    - Deterministic parsing for checksec, ROPgadget, one_gadget
    - LLM parsing for decompile, GDB, and other complex outputs
    """

    # Tools that can be parsed deterministically (no LLM needed)
    DETERMINISTIC_TOOLS = {"Checksec", "ROPgadget", "One_gadget"}

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

        self.clear_history()

        instruction_output = state.get("instruction_output", {})
        executions = instruction_output.get("executions", [])

        if not executions:
            console.print("[yellow]No executions to parse[/yellow]")
            state["parsing_had_no_executions"] = True
            return state

        # Split executions into deterministic and LLM-required
        deterministic_execs = []
        llm_execs = []

        for ex in executions:
            tool_name = ex.get("tool", "")
            if tool_name in self.DETERMINISTIC_TOOLS:
                deterministic_execs.append(ex)
            else:
                llm_execs.append(ex)

        all_key_findings = []

        # --- Phase 1: Deterministic parsing (no LLM) ---
        for ex in deterministic_execs:
            tool_name = ex.get("tool", "")
            stdout = ex.get("stdout", "")

            console.print(f"[green]Deterministic parse: {tool_name}[/green]")

            if tool_name == "Checksec":
                checksec_data = parse_checksec_output(stdout)
                merge_analysis_updates(state, {"checksec": checksec_data})
                all_key_findings.append(f"Checksec: {checksec_data['result']}")

                # Also get KB guide based on checksec
                try:
                    from Store.knowledge import get_checksec_guide
                    guide = get_checksec_guide(checksec_data)
                    if guide.get("recommended"):
                        top = guide["recommended"][0]
                        all_key_findings.append(
                            f"KB recommends: {top.get('name', '')} - {top.get('description', '')}"
                        )
                        # Set initial strategy suggestion
                        if not state.get("analysis", {}).get("strategy"):
                            merge_analysis_updates(state, {
                                "strategy": f"Recommended: {top.get('name', '')} ({top.get('description', '')})"
                            })
                except Exception:
                    pass

                console.print(Panel(checksec_data["result"], title="Checksec (Deterministic)", border_style="green"))

            elif tool_name == "ROPgadget":
                gadgets = parse_ropgadget_output(stdout)
                if gadgets:
                    merge_analysis_updates(state, {"gadgets": gadgets})
                    # Highlight important gadgets
                    important = [g for g in gadgets if any(
                        k in g["instruction"] for k in ["pop rdi", "pop rsi", "pop rdx", "ret"]
                    )]
                    for g in important[:5]:
                        all_key_findings.append(f"Gadget: {g['address']} : {g['instruction']}")
                    console.print(f"[green]Parsed {len(gadgets)} gadgets ({len(important)} important)[/green]")

            elif tool_name == "One_gadget":
                one_gadgets = parse_one_gadget_output(stdout)
                if one_gadgets:
                    # Store as formatted string for libc info
                    og_str = "\n".join(
                        f"{og['offset']}: {', '.join(og['constraints'])}"
                        for og in one_gadgets
                    )
                    merge_analysis_updates(state, {
                        "libc": {"one_gadgets": og_str}
                    })
                    all_key_findings.append(f"Found {len(one_gadgets)} one_gadget offsets")
                    console.print(f"[green]Parsed {len(one_gadgets)} one_gadgets[/green]")

        # --- Phase 2: LLM parsing for complex outputs ---
        parsed = {}
        if llm_execs:
            messages = build_messages(
                agent="parsing",
                state=state,
                include_initial=False,
                executions=llm_execs,
            )

            if messages:
                self.set_system_prompt(messages[0]["content"])
                for msg in messages[1:]:
                    if msg["role"] == "user":
                        self.add_user_message(msg["content"])

            console.print(f"[cyan]Calling LLM for {len(llm_execs)} complex outputs...[/cyan]")
            response_text = self.call_llm()
            parsed = parse_json_response(response_text)

            # Merge LLM analysis updates
            if "analysis_updates" in parsed:
                merge_analysis_updates(state, parsed["analysis_updates"])
                console.print("[green]Analysis document updated (LLM)[/green]")

            llm_findings = parsed.get("key_findings", [])
            all_key_findings.extend(llm_findings)
        else:
            console.print("[dim]All outputs parsed deterministically, no LLM call needed[/dim]")

        # Infer offset/leak from key_findings for readiness (LLM이 구조화하지 않은 경우 대비)
        # Include raw stdout from executions for DV extraction (canary/ret offsets)
        analysis = state.get("analysis", {})
        extras = []
        for ex in executions:
            stdout = ex.get("stdout", "")[:8000]
            if stdout:
                extras.append(stdout)
        infer_readiness_from_key_findings(
            analysis,
            all_key_findings + extras if extras else all_key_findings,
        )
        state["analysis"] = analysis

        # --- Store results ---
        state["parsing_output"] = {
            "agent": "parsing",
            "text": "",
            "json": {**parsed, "key_findings": all_key_findings},
            "created_at": time.time(),
        }

        if all_key_findings:
            console.print("[cyan]Key Findings:[/cyan]")
            for finding in all_key_findings:
                console.print(f"  • {finding}")

            runs = state.get("runs", [])
            if runs:
                runs[-1]["key_findings"] = all_key_findings

        # Mark tasks as done
        instruction_json = instruction_output.get("json", {})
        for task_id in instruction_json.get("selected_tasks", []):
            mark_task_status(state, task_id, "done")

        state["parsing_had_no_executions"] = False
        return state
