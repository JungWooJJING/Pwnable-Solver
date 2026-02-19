"""
LangGraph Workflow Definition for PWN Solver

Workflow:
    Plan → Instruction → Parsing → Feedback → (loop back to Plan or go to Stage Identify)

Staged Exploit Flow:
    Stage Identify → Stage Exploit → Stage Verify
      → (verified + 다음 단계 있음) → Stage Advance → Stage Exploit
      → (verified + 마지막 단계) → END
      → (실패 + 재시도 가능) → Stage Refine → Stage Verify
      → (실패 + 재시도 소진) → Plan (재분석)
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from LangGraph.state import SolverState
from LangGraph.node import (
    Plan_node,
    Instruction_node,
    Parsing_node,
    Feedback_node,
    # Legacy (kept for compatibility)
    Exploit_node,
    Verify_node,
    Crash_analysis_node,
    Refine_node,
    # Staged exploit nodes
    Stage_identify_node,
    Stage_exploit_node,
    Stage_verify_node,
    Stage_refine_node,
    Stage_advance_node,
)


def build_workflow() -> StateGraph:
    """Build and return the PWN Solver workflow graph (staged exploit)"""

    workflow = StateGraph(SolverState)

    # Analysis loop nodes (unchanged)
    workflow.add_node("plan", Plan_node)
    workflow.add_node("instruction", Instruction_node)
    workflow.add_node("parsing", Parsing_node)
    workflow.add_node("feedback", Feedback_node)

    # Staged exploit nodes (new)
    workflow.add_node("stage_identify", Stage_identify_node)
    workflow.add_node("stage_exploit", Stage_exploit_node)
    workflow.add_node("stage_verify", Stage_verify_node)
    workflow.add_node("stage_refine", Stage_refine_node)
    workflow.add_node("stage_advance", Stage_advance_node)

    # Entry point
    workflow.set_entry_point("plan")

    # Analysis loop edges (unchanged)
    workflow.add_edge("plan", "instruction")
    workflow.add_edge("instruction", "parsing")
    workflow.add_edge("parsing", "feedback")

    # feedback → (plan | stage_identify | end)
    workflow.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "plan": "plan",
            "stage_identify": "stage_identify",
            "end": END,
        }
    )

    # stage_identify → stage_exploit
    workflow.add_edge("stage_identify", "stage_exploit")

    # stage_exploit → stage_verify
    workflow.add_edge("stage_exploit", "stage_verify")

    # stage_verify → (stage_advance | stage_refine | end | plan)
    workflow.add_conditional_edges(
        "stage_verify",
        route_after_stage_verify,
        {
            "stage_advance": "stage_advance",
            "stage_refine": "stage_refine",
            "end": END,
            "plan": "plan",
        }
    )

    # stage_advance → stage_exploit (다음 단계 코드 생성)
    workflow.add_edge("stage_advance", "stage_exploit")

    # stage_refine → stage_verify (수정 후 재검증)
    workflow.add_edge("stage_refine", "stage_verify")

    return workflow


def route_after_feedback(state: SolverState) -> Literal["plan", "stage_identify", "end"]:
    """Determine next step after feedback node"""

    if state.get("flag_detected", False):
        return "end"

    if state.get("loop", True):
        return "plan"

    exploit_readiness = state.get("exploit_readiness", {})
    if exploit_readiness.get("recommend_exploit", False):
        return "stage_identify"

    return "end"


def route_after_stage_verify(
    state: SolverState,
) -> Literal["stage_advance", "stage_refine", "end", "plan"]:
    """Determine next step after stage verification."""
    from rich.console import Console
    console = Console()

    if state.get("flag_detected"):
        return "end"

    staged = state.get("staged_exploit", {})
    stages = staged.get("stages", [])
    current_idx = staged.get("current_stage_index", 0)

    if current_idx >= len(stages):
        return "end"

    current_stage = stages[current_idx]

    if current_stage.get("verified"):
        # Stage passed
        if staged.get("all_stages_verified") or state.get("exploit_verified"):
            console.print("[bold green]All stages verified — exploit success![/bold green]")
            return "end"
        elif current_idx + 1 < len(stages):
            console.print(f"[green]Stage {current_idx+1} passed → advancing to next stage[/green]")
            return "stage_advance"
        else:
            return "end"
    else:
        # Stage failed
        attempts = current_stage.get("refinement_attempts", 0)
        max_attempts = 3

        if attempts < max_attempts:
            console.print(f"[yellow]Stage {current_idx+1} failed (attempt {attempts+1}/{max_attempts}) → refining[/yellow]")
            return "stage_refine"
        else:
            console.print(f"[red]Stage {current_idx+1} exhausted retries → re-analyzing[/red]")
            # 실패 컨텍스트를 state에 기록해서 Plan이 인지하도록
            state["analysis_failure_reason"] = (
                f"Stage '{current_stage.get('stage_id', '?')}' failed after {max_attempts} refinement attempts. "
                f"Description: {current_stage.get('description', '')}. "
                f"Error: {current_stage.get('error', '')[:500]}"
            )
            state["exploit_failure_context"] = {
                "stage_id": current_stage.get("stage_id", ""),
                "stage_index": current_idx,
                "code": current_stage.get("code", "")[:2000],
                "error": current_stage.get("error", "")[:1000],
                "attempts": max_attempts,
            }
            return "plan"


def create_app():
    """Create compiled LangGraph app"""
    workflow = build_workflow()
    return workflow.compile()


# For direct execution
if __name__ == "__main__":
    from rich.console import Console
    from LangGraph.state import init_state

    console = Console()
    console.print("[bold green]Building PWN Solver Workflow...[/bold green]")

    app = create_app()
    console.print("[green]Workflow created successfully![/green]")

    console.print("\n[cyan]Workflow Structure:[/cyan]")
    console.print("  Plan → Instruction → Parsing → Feedback")
    console.print("    ↑                               ↓")
    console.print("    │               ┌───────────────┼───────────────┐")
    console.print("    │               ↓               ↓               ↓")
    console.print("    │             Plan        Stage Identify       END")
    console.print("    │           (loop)             ↓            (stop)")
    console.print("    │                        Stage Exploit")
    console.print("    │                              ↓")
    console.print("    │                        Stage Verify")
    console.print("    │                     ┌────┴────┬──────┐")
    console.print("    │                     ↓         ↓      ↓")
    console.print("    │                  Advance   Refine   END")
    console.print("    │                     ↓         ↓   (success)")
    console.print("    │              Stage Exploit  Stage Verify")
    console.print("    │              (next stage)  (retry)")
    console.print("    └──────────────────────────────────────┘")
