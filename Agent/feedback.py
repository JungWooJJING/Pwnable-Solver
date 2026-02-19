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

        # #4: 취약점 타입별 동적 threshold
        effective_threshold = self._compute_dynamic_threshold(analysis)
        console.print(f"[cyan]Dynamic Threshold: {effective_threshold:.2f} (base: {self.readiness_threshold})[/cyan]")

        # Recommend exploit if code score meets threshold
        recommend_exploit = readiness_score >= effective_threshold
        # Also allow if LLM recommends and score is close
        if llm_recommend and readiness_score >= (effective_threshold - 0.15):
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
            feedback_additions = []

            if missing:
                console.print(f"[yellow]  Still missing: {', '.join(missing)} — continuing[/yellow]")
                feedback_additions.append(f"[AUTO] Focus on resolving missing items: {', '.join(missing)}")

            # #3: 실패한 익스 정보를 Plan에 피드백
            failure_ctx = state.get("exploit_failure_context", {})
            if failure_ctx:
                stage_id = failure_ctx.get("stage_id", "unknown")
                error = failure_ctx.get("error", "")[:300]
                code_snippet = failure_ctx.get("code", "")[:500]
                feedback_additions.append(
                    f"\n[EXPLOIT FAILURE] Stage '{stage_id}' failed after {failure_ctx.get('attempts', '?')} attempts.\n"
                    f"Error: {error}\n"
                    f"Failing code snippet:\n```python\n{code_snippet}\n```\n"
                    f"→ Re-analyze: check offsets, I/O interaction pattern, leak values. "
                    f"If leak returns 0x0, the leak offset is wrong."
                )

            if feedback_additions:
                if "feedback_to_plan" not in parsed or not parsed["feedback_to_plan"]:
                    parsed["feedback_to_plan"] = ""
                parsed["feedback_to_plan"] += "\n\n" + "\n\n".join(feedback_additions)
                state["feedback_output"]["json"] = parsed

            state["loop"] = True

        return state

    def _compute_dynamic_threshold(self, analysis: dict) -> float:
        """#4: 취약점 타입에 따라 readiness threshold를 동적으로 계산."""
        vulns = analysis.get("vulnerabilities", [])
        has_win = analysis.get("win_function", False)
        checksec = analysis.get("checksec", {})
        result_str = str(checksec.get("result", "")).lower()
        nx = "nx enabled" in result_str
        pie = "pie enabled" in result_str and "no pie" not in result_str

        vuln_types = {v.get("type", "") for v in vulns}

        # win 함수 있으면 매우 낮은 임계값 (leak만 있으면 됨)
        if has_win and not pie:
            return 0.35
        if has_win and pie:
            return 0.40

        # fmt string: GOT overwrite 가능하면 libc leak 없어도 됨
        if "format_string" in vuln_types:
            return 0.40

        # shellcode (NX disabled): ROP/libc 불필요
        if not nx:
            return 0.35

        # heap: 분석이 복잡하므로 조금 더 요구
        if any(t in vuln_types for t in ("use_after_free", "heap_overflow", "double_free")):
            return 0.55

        # 기본 BOF+ROP 시나리오
        return self.readiness_threshold
