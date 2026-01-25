"""
LangGraph Workflow Definition for PWN Solver

Workflow:
    Plan → Instruction → Parsing → Feedback → (loop back to Plan or go to Exploit)
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from LangGraph.state import SolverState
from LangGraph.node import (
    Plan_node,
    Instruction_node,
    Parsing_node,
    Feedback_node,
    Exploit_node,
    should_continue,
)


def build_workflow() -> StateGraph:
    """Build and return the PWN Solver workflow graph"""

    # Create graph with SolverState
    workflow = StateGraph(SolverState)

    # Add nodes
    workflow.add_node("plan", Plan_node)
    workflow.add_node("instruction", Instruction_node)
    workflow.add_node("parsing", Parsing_node)
    workflow.add_node("feedback", Feedback_node)
    workflow.add_node("exploit", Exploit_node)

    # Set entry point
    workflow.set_entry_point("plan")

    # Add edges (linear flow within iteration)
    workflow.add_edge("plan", "instruction")
    workflow.add_edge("instruction", "parsing")
    workflow.add_edge("parsing", "feedback")

    # Conditional edge from feedback (loop or exploit or end)
    workflow.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "plan": "plan",       # Continue loop
            "exploit": "exploit", # Ready for exploitation
            "end": END,           # Terminate
        }
    )

    # Exploit goes to end
    workflow.add_edge("exploit", END)

    return workflow


def route_after_feedback(state: SolverState) -> Literal["plan", "exploit", "end"]:
    """Determine next step after feedback node"""

    # Check if flag was found
    if state.get("flag_detected", False):
        return "end"

    # Check if loop should continue
    if state.get("loop", True):
        return "plan"

    # Check if ready for exploitation
    exploit_readiness = state.get("exploit_readiness", {})
    if exploit_readiness.get("recommend_exploit", False):
        return "exploit"

    return "end"


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

    # Print graph structure
    console.print("\n[cyan]Workflow Structure:[/cyan]")
    console.print("  Plan → Instruction → Parsing → Feedback")
    console.print("                                    ↓")
    console.print("                    ┌───────────────┼───────────────┐")
    console.print("                    ↓               ↓               ↓")
    console.print("                  Plan          Exploit           END")
    console.print("                (loop)         (ready)          (stop)")
