"""
Feedback Agent - Evaluates progress and provides feedback.

Responsibilities:
- Code-based readiness score computation (deterministic, no LLM)
- Decide whether to loop or go to Exploit
- Build deterministic feedback_to_plan from analysis state
"""

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

from Agent.base import BaseAgent
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
        """Run Feedback Agent — code-based score only (no LLM call)."""
        console.print(Panel("Feedback Agent", style="bold green"))

        # --- Step 1: Code-based readiness score ---
        analysis = state.get("analysis", {})
        runs = state.get("runs", [])
        code_score, completed, missing = compute_readiness(analysis, runs=runs)

        console.print(f"[cyan]Code-based Readiness: {code_score:.0%}[/cyan]")
        if completed:
            console.print(f"[green]  Completed: {', '.join(completed)}[/green]")
        if missing:
            console.print(f"[yellow]  Missing: {', '.join(missing)}[/yellow]")

        # --- Step 2: Threshold check ---
        effective_threshold = self._compute_dynamic_threshold(analysis)
        recommend_exploit = code_score >= effective_threshold

        state["exploit_readiness"] = {
            "score": code_score,
            "components": completed,
            "missing": missing,
            "recommend_exploit": recommend_exploit,
        }
        if "analysis" in state:
            state["analysis"]["readiness_score"] = code_score

        console.print(f"[cyan]Threshold: {effective_threshold:.2f} → Recommend Exploit: {recommend_exploit}[/cyan]")

        # --- Step 3: Build deterministic feedback_to_plan ---
        feedback_parts = []

        if missing:
            feedback_parts.append(f"Still missing: {', '.join(missing)}. Focus on resolving these.")

        # Exploit failure context → guide Plan to re-analyze
        failure_ctx = state.get("exploit_failure_context", {})
        if failure_ctx:
            stage_id = failure_ctx.get("stage_id", "unknown")
            error = failure_ctx.get("error", "")[:400]
            code_snippet = failure_ctx.get("code", "")[:600]
            feedback_parts.append(
                f"[EXPLOIT FAILURE] Stage '{stage_id}' failed after {failure_ctx.get('attempts', '?')} attempts.\n"
                f"Error: {error}\n"
                f"Failing code:\n```python\n{code_snippet}\n```\n"
                f"→ Re-analyze offsets, I/O pattern, and leak values. "
                f"If PIE leak returns 0x0 or an ASCII-like value, the pattern_len / offset is wrong."
            )

        # Detect unverified "potential" findings in key_findings and surface them as blockers.
        # These are often hallucinated or unconfirmed and should NOT drive exploitation.
        import re as _re_fb
        key_findings = analysis.get("key_findings", [])
        potential_findings = []
        for finding in key_findings:
            lower = str(finding).lower()
            if _re_fb.search(r'\b(?:potential(?:ly)?|possibly|may\s+be\s+able\s+to|might)\b', lower):
                potential_findings.append(str(finding))
        if potential_findings:
            console.print(f"[yellow]⚠ Unverified potential findings: {len(potential_findings)} item(s)[/yellow]")
            for pf in potential_findings[:3]:
                console.print(f"[yellow]  • {pf[:120]}[/yellow]")
            feedback_parts.append(
                f"[UNVERIFIED FINDINGS] The following findings are still 'potential'/'possible' "
                f"and have NOT been confirmed by dynamic analysis or decompilation:\n"
                + "\n".join(f"  • {pf[:200]}" for pf in potential_findings[:5])
                + "\n→ Do NOT use these as confirmed primitives. "
                f"Run GDB or decompile the relevant functions to confirm before proceeding."
            )

        # Key analysis hints for Plan
        dv = analysis.get("dynamic_verification", {})
        if dv.get("buf_offset_to_canary"):
            feedback_parts.append(
                f"Verified offsets: buf_offset_to_canary={dv.get('buf_offset_to_canary')}, "
                f"buf_offset_to_ret={dv.get('buf_offset_to_ret')}."
            )

        feedback_to_plan = "\n\n".join(feedback_parts) if feedback_parts else (
            "Analysis looks complete. Proceed with exploit generation."
        )

        parsed = {"feedback_to_plan": feedback_to_plan, "recommend_exploit": recommend_exploit}
        state["feedback_output"] = {
            "agent": "feedback",
            "text": feedback_to_plan,
            "json": parsed,
            "created_at": time.time(),
        }

        console.print(Panel(feedback_to_plan, title="Feedback to Plan", border_style="yellow"))

        # --- Step 4: Decision ---
        if recommend_exploit:
            console.print("[green]Ready for exploitation![/green]")
            state["loop"] = False
        else:
            if missing:
                console.print(f"[yellow]  Still missing: {', '.join(missing)} — continuing[/yellow]")
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
