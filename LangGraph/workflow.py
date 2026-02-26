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
    # Helpers
    _classify_failure_type,
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

    # stage_verify → (stage_advance | stage_refine | stage_exploit | end | plan)
    workflow.add_conditional_edges(
        "stage_verify",
        route_after_stage_verify,
        {
            "stage_advance": "stage_advance",
            "stage_refine": "stage_refine",
            "stage_exploit": "stage_exploit",  # Fresh regeneration (preserves verified stages)
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
        # Hard block: canary or PIE requires dynamic verification before exploitation
        analysis = state.get("analysis", {})
        result_str = str(analysis.get("checksec", {}).get("result", "")).lower()
        has_canary = "canary found" in result_str and "no canary" not in result_str
        has_pie = "pie enabled" in result_str and "no pie" not in result_str

        dv = analysis.get("dynamic_verification", {})
        vulns = analysis.get("vulnerabilities", [])
        vuln_types = {v.get("type", "").lower() for v in vulns}
        strategy_lower = str(analysis.get("strategy", "")).lower()
        # heap/vtable/type_confusion만 — 스택 미사용, dv/canary 불필요
        # oob, format_string 제외 (스택 기반 가능, canary 적용 가능)
        non_stack = (
            "use_after_free", "uaf", "heap_overflow", "double_free",
            "type_confusion", "type confusion", "vtable", "vtable_overwrite",
        )
        vuln_norm = {t.replace(" ", "_") for t in vuln_types}
        has_non_stack = (
            any(t in non_stack or t.replace("_", " ") in non_stack for t in vuln_norm)
            or "got overwrite" in strategy_lower or "vtable" in strategy_lower
            or "type confusion" in strategy_lower or "heap" in strategy_lower
        )
        has_win = analysis.get("win_function", False) or bool(analysis.get("win_function_addr"))
        # heap/vtable/type_confusion + win in binary: dv 불필요
        skip_dv_for_path = has_non_stack and has_win
        needs_stack_offsets = (
            any(t in ("buffer_overflow", "bof", "stack_overflow") for t in vuln_types)
            and not has_non_stack
        )
        if needs_stack_offsets:
            dv_complete = (
                dv.get("verified") is True
                and isinstance(dv.get("buf_offset_to_canary"), int)
                and isinstance(dv.get("buf_offset_to_ret"), int)
            )
        else:
            # 비-스택오버플로우: verified만 있으면 충분
            dv_complete = dv.get("verified") is True

        # If we came back from a failed stage exploit, don't block on dv again —
        # the LLM already understands the offsets from static/source analysis.
        came_from_stage_failure = bool(state.get("exploit_failure_context"))

        if (
            (has_canary or has_pie)
            and not dv_complete
            and not came_from_stage_failure
            and not skip_dv_for_path
        ):
            from rich.console import Console
            _console = Console()

            # Track how many times we've been forced back for GDB.
            # Bypass when: GDB_MAX_FORCE attempts, OR duplicate loop (no executions, already tried once)
            GDB_MAX_FORCE = 2
            gdb_forced = state.get("gdb_forced_count", 0)
            parsing_no_exec = state.get("parsing_had_no_executions", False)

            if gdb_forced >= GDB_MAX_FORCE or (parsing_no_exec and gdb_forced >= 1):
                reason = "duplicate tasks (no new executions)" if parsing_no_exec else f"{gdb_forced} GDB attempt(s)"
                _console.print(
                    f"[yellow]⚠ Dynamic verification incomplete ({reason}) — proceeding with static analysis[/yellow]"
                )
                return "stage_identify"

            # NOTE: state 변경은 여기서 하지 않음.
            # gdb_forced_count 증가는 Feedback_node에서 처리 (routing function의 state 변경은 LangGraph에서 반영 안 됨).
            _console.print(
                "[yellow]⚠ Canary/PIE detected but dynamic_verification incomplete "
                f"(verified={dv.get('verified')}, "
                f"canary_offset={dv.get('buf_offset_to_canary')}, "
                f"ret_offset={dv.get('buf_offset_to_ret')}) "
                f"— forcing GDB run before exploit (attempt {gdb_forced}/{GDB_MAX_FORCE})[/yellow]"
            )
            return "plan"

        return "stage_identify"

    return "end"


def route_after_stage_verify(
    state: SolverState,
) -> Literal["stage_advance", "stage_refine", "stage_exploit", "end", "plan"]:
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
        max_refine = 3
        max_regen  = 2  # fresh full regenerations before falling back to plan

        # --- Early escalation: same failure type repeated without improvement ---
        # If we've seen the same failure category >= 2 times, further refinement
        # won't help — escalate directly to re-analysis (plan).
        # Exception: CONNECTION_CLOSED often resolves with a single retry, so skip it.
        _current_ft = _classify_failure_type(current_stage.get("error", ""))
        _ft_history = current_stage.get("failure_type_history", [])
        _ESCALATE_TYPES = {"TIMED_OUT", "WRONG_VALUE", "WRONG_FORMAT", "EXCEPTION"}
        _TIMED_OUT_THRESHOLD = 1   # even 1 repeat → escalate (I/O protocol issue)
        _GENERAL_THRESHOLD   = 2   # 2 repeats of the same type → escalate

        if _current_ft in _ESCALATE_TYPES and _ft_history:
            # Count consecutive same-type entries from the END of history only
            _consecutive = 0
            for _t in reversed(_ft_history):
                if _t == _current_ft:
                    _consecutive += 1
                else:
                    break
            _threshold = _TIMED_OUT_THRESHOLD if _current_ft == "TIMED_OUT" else _GENERAL_THRESHOLD
            if _consecutive >= _threshold:
                console.print(
                    f"[red]Stage {current_idx+1} escalating: '{_current_ft}' "
                    f"failed {_consecutive + 1} times in a row (history={_ft_history}) "
                    f"— refinement ineffective, re-analyzing[/red]"
                )
                state["analysis_failure_reason"] = (
                    f"Stage '{current_stage.get('stage_id', '?')}' failed with "
                    f"'{_current_ft}' {_consecutive + 1} times in a row — "
                    f"refinement made no progress. Last error: "
                    f"{current_stage.get('error', '')[:400]}"
                )
                state["exploit_failure_context"] = {
                    "stage_id": current_stage.get("stage_id", ""),
                    "stage_index": current_idx,
                    "code": current_stage.get("code", "")[:2000],
                    "error": current_stage.get("error", "")[:1000],
                    "attempts": attempts,
                    "failure_type": _current_ft,
                    "failure_type_history": _ft_history,
                }
                return "plan"

        if attempts < max_refine:
            console.print(f"[yellow]Stage {current_idx+1} failed (attempt {attempts+1}/{max_refine}) → refining[/yellow]")
            return "stage_refine"
        else:
            # Refinement exhausted — try a fresh full regeneration first
            # so verified previous stages are NOT lost (no stage_identify reset).
            regen_count = current_stage.get("regen_count", 0)
            if regen_count < max_regen:
                console.print(
                    f"[yellow]Stage {current_idx+1} refinement exhausted "
                    f"(regen {regen_count+1}/{max_regen}) → fresh regeneration[/yellow]"
                )
                current_stage["refinement_attempts"] = 0
                current_stage["regen_count"] = regen_count + 1
                current_stage["code"] = ""
                current_stage["error"] = ""
                stages[current_idx] = current_stage
                state["staged_exploit"]["stages"] = stages
                return "stage_exploit"  # Regenerate without resetting verified stages
            else:
                console.print(f"[red]Stage {current_idx+1} exhausted all retries → re-analyzing[/red]")
                # 실패 컨텍스트를 state에 기록해서 Plan이 인지하도록
                state["analysis_failure_reason"] = (
                    f"Stage '{current_stage.get('stage_id', '?')}' failed after "
                    f"{max_refine} refinements × {max_regen} regenerations. "
                    f"Description: {current_stage.get('description', '')}. "
                    f"Error: {current_stage.get('error', '')[:500]}"
                )
                state["exploit_failure_context"] = {
                    "stage_id": current_stage.get("stage_id", ""),
                    "stage_index": current_idx,
                    "code": current_stage.get("code", "")[:2000],
                    "error": current_stage.get("error", "")[:1000],
                    "attempts": max_refine * max_regen,
                }

                # 이전 Stage 재검증:
                # Stage N이 모든 재시도를 소진했을 때 Stage N-1의 결과가
                # 잘못된 검증(false positive)을 통과했을 가능성을 고려.
                # Skip re-invalidation when failure is env/path or leak-parsing (not upstream false positive).
                stage_error = (current_stage.get("error") or "") + (state.get("exploit_failure_context", {}).get("error", "") or "")
                is_env_or_path_failure = (
                    "FileNotFoundError" in stage_error
                    or ("No such file" in stage_error and "libc" in stage_error.lower())
                )
                is_leak_or_parse_failure = (
                    "Leak parsing failed" in stage_error
                    or "unpack requires" in stage_error
                    or "STAGE_FAILED" in stage_error
                )
                if current_idx > 0 and not is_env_or_path_failure and not is_leak_or_parse_failure:
                    prev_stage = stages[current_idx - 1]
                    if prev_stage.get("verified"):
                        console.print(
                            f"[yellow]⚠ Re-invalidating Stage {current_idx} "
                            f"('{prev_stage.get('stage_id', '?')}') — "
                            f"downstream Stage {current_idx+1} exhausted all retries, "
                            f"upstream result may be a false positive[/yellow]"
                        )
                        prev_stage["verified"] = False
                        prev_stage["refinement_attempts"] = 0
                        prev_stage["regen_count"] = 0
                        prev_stage["error"] = (
                            f"Re-invalidated: Stage {current_idx+1} "
                            f"('{current_stage.get('stage_id', '?')}') exhausted all retries. "
                            f"This stage's output may have been a false positive "
                            f"(e.g. stack address mistaken for libc address)."
                        )
                        stages[current_idx - 1] = prev_stage
                        # current_stage도 초기화하여 이전 Stage 재성공 후 재시도 가능하게
                        current_stage["refinement_attempts"] = 0
                        current_stage["regen_count"] = 0
                        current_stage["verified"] = False
                        stages[current_idx] = current_stage
                        state["staged_exploit"]["stages"] = stages
                        state["staged_exploit"]["current_stage_index"] = current_idx - 1

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
