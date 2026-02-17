import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rich.console import Console
from rich.panel import Panel

from LangGraph.state import (
    SolverState as State,
    init_analysis,
)

console = Console()


# =============================================================================
# Helper Functions
# =============================================================================

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _challenge_dir_for_state(state: State) -> Path:
    """
    Output directory:
      /home/wjddn0623/lab/new_solver/Challenge/<binary_stem>/
    """
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "Challenge"
    binary_path = state.get("binary_path", "") or ""
    name = Path(binary_path).stem if binary_path else "unknown"
    out_dir = root / f"{_slug(name)}_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# Agent Instances (lazy initialization)
# =============================================================================

_agents: Dict[str, Any] = {}


def _get_agent(agent_type: str, model: Optional[str] = None, provider: Optional[str] = None):
    """Get or create agent instance."""
    from Agent import (
        PlanAgent,
        InstructionAgent,
        ParsingAgent,
        FeedbackAgent,
        ExploitAgent,
    )

    agent_classes = {
        "plan": PlanAgent,
        "instruction": InstructionAgent,
        "parsing": ParsingAgent,
        "feedback": FeedbackAgent,
        "exploit": ExploitAgent,
    }

    # 캐싱 키 생성 - None 값 처리
    model_key = model or "default"
    provider_key = provider or "auto"
    key = f"{agent_type}_{model_key}_{provider_key}"

    if key not in _agents:
        cls = agent_classes.get(agent_type)
        if cls is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        _agents[key] = cls(model=model, provider=provider)

    return _agents[key]


def get_total_metrics() -> Dict[str, Any]:
    """Get combined metrics from all agents."""
    total = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cached_tokens": 0,
        "total_tokens": 0,
        "message_count": 0,
        "max_context_size": 0,
        "total_cost_usd": 0.0,
    }
    
    for agent in _agents.values():
        metrics = agent.get_metrics()
        total["total_input_tokens"] += metrics.get("total_input_tokens", 0)
        total["total_output_tokens"] += metrics.get("total_output_tokens", 0)
        total["total_cached_tokens"] += metrics.get("total_cached_tokens", 0)
        total["total_tokens"] += metrics.get("total_tokens", 0)
        total["message_count"] += metrics.get("message_count", 0)
        total["total_cost_usd"] += metrics.get("total_cost_usd", 0.0)
        if metrics.get("max_context_size", 0) > total["max_context_size"]:
            total["max_context_size"] = metrics["max_context_size"]
    
    return total


# =============================================================================
# Node Implementations
# =============================================================================

def Plan_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Plan Agent Node

    - First iteration: Auto-execute Phase 1 RECON (checksec, ghidra, ls)
    - Subsequent iterations: Analyze feedback and create/update tasks
    """
    console.print(Panel("Plan Node", style="bold magenta"))

    binary_path = state.get("binary_path", "")

    # First iteration initialization
    if not state.get("loop", False):
        console.print("[green]Initializing Plan Node...[/green]")

        # Get directory listing
        if binary_path:
            workdir = str(Path(binary_path).resolve().parent)
            ls_result = subprocess.run(
                ["ls", "-la"],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            state["directory_listing"] = ls_result.stdout.strip()
            state["cwd"] = workdir

            # Save to Challenge directory
            try:
                out_dir = _challenge_dir_for_state(state)
                (out_dir / "directory_listing.txt").write_text(
                    (ls_result.stdout or "") +
                    ("\n\n--- STDERR ---\n" + (ls_result.stderr or "") if ls_result.stderr else ""),
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                pass

        # Initialize analysis if needed
        if not state.get("analysis"):
            state["analysis"] = init_analysis()

        state["loop"] = True
        state["iteration_count"] = 1

        # --- Phase 1 RECON: Auto-execute checksec + ghidra ---
        if binary_path:
            _run_phase1_recon(state, binary_path)

    else:
        state["iteration_count"] = state.get("iteration_count", 0) + 1

    console.print(f"[dim]Iteration: {state['iteration_count']}[/dim]")

    # Run Plan Agent
    agent = _get_agent("plan", model=model, provider=provider)
    state = agent.run(state)

    return state


def _run_phase1_recon(state: State, binary_path: str) -> None:
    """
    Auto-execute Phase 1 RECON tools (no LLM needed).

    Runs checksec and ghidra_main, parses results deterministically,
    and populates the Analysis Document directly.
    """
    from Tool.tool import Tool
    from Agent.parsing import parse_checksec_output
    from LangGraph.state import merge_analysis_updates, add_run, TaskRun
    import time as _time

    console.print("[cyan]Phase 1 RECON: Auto-executing checksec + ghidra...[/cyan]")

    try:
        tool = Tool(binary_path=binary_path)
    except Exception as e:
        console.print(f"[red]Tool init failed: {e}[/red]")
        return

    # --- Checksec ---
    try:
        checksec_raw = tool.Checksec()
        checksec_data = parse_checksec_output(str(checksec_raw))
        merge_analysis_updates(state, {"checksec": checksec_data})
        console.print(Panel(checksec_data["result"], title="Phase 1: Checksec", border_style="green"))

        # Get KB recommendation
        try:
            from Store.knowledge import get_checksec_guide
            guide = get_checksec_guide(checksec_data)
            if guide.get("recommended"):
                top = guide["recommended"][0]
                strategy = f"Recommended: {top.get('name', '')} ({top.get('description', '')})"
                merge_analysis_updates(state, {"strategy": strategy})
                console.print(f"[green]KB Strategy: {strategy}[/green]")
        except Exception:
            pass

        add_run(state, {
            "run_id": f"recon_checksec_{int(_time.time()*1000)}",
            "task_id": "phase1_checksec",
            "commands": ["Checksec()"],
            "success": True,
            "stdout": str(checksec_raw),
            "stderr": "",
            "started_at": _time.time(),
            "finished_at": _time.time(),
            "key_findings": [f"Checksec: {checksec_data['result']}"],
        })
    except Exception as e:
        console.print(f"[yellow]Checksec failed: {e}[/yellow]")

    # --- Ghidra Main ---
    try:
        ghidra_raw = tool.Ghidra_main(main_only=True)
        if ghidra_raw and not str(ghidra_raw).startswith("Error:"):
            merge_analysis_updates(state, {
                "decompile": {
                    "done": True,
                    "functions": [{"name": "main", "address": "", "code": str(ghidra_raw)[:3000]}],
                }
            })
            preview = str(ghidra_raw)[:500]
            console.print(Panel(preview, title="Phase 1: Ghidra Main", border_style="green"))

            add_run(state, {
                "run_id": f"recon_ghidra_{int(_time.time()*1000)}",
                "task_id": "phase1_ghidra",
                "commands": ["Ghidra_main(main_only=True)"],
                "success": True,
                "stdout": str(ghidra_raw)[:5000],
                "stderr": "",
                "started_at": _time.time(),
                "finished_at": _time.time(),
                "key_findings": ["Main function decompiled"],
            })
        else:
            console.print(f"[yellow]Ghidra returned: {str(ghidra_raw)[:200]}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Ghidra failed: {e}[/yellow]")

    console.print("[green]Phase 1 RECON complete - Plan Agent starts at Phase 2[/green]")


def Instruction_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Instruction Agent Node

    - Select pending tasks based on priority
    - Convert tasks to tool calls
    - Execute tools and collect results
    """
    agent = _get_agent("instruction", model=model, provider=provider)
    return agent.run(state)


def Parsing_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Parsing Agent Node

    - Parse execution results from Instruction node
    - Extract structured information for Analysis Document
    - Identify vulnerabilities and key findings
    """
    agent = _get_agent("parsing", model=model, provider=provider)
    return agent.run(state)


def Feedback_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Feedback Agent Node

    - Evaluate progress toward exploitation
    - Calculate readiness score
    - Provide feedback to Plan agent
    - Decide whether to loop or go to Exploit
    """
    agent = _get_agent("feedback", model=model, provider=provider)
    return agent.run(state)


def Exploit_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Exploit Agent Node

    - Generate pwntools exploit code
    - Use all gathered analysis information
    """
    agent = _get_agent("exploit", model=model, provider=provider)
    return agent.run(state)


def Verify_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Verify Node - 익스플로잇 실행 및 검증

    - 생성된 exploit.py 실행
    - 플래그/쉘 획득 여부 확인
    - 실패 시 에러 저장
    """
    console.print(Panel("Verify Exploit", style="bold yellow"))

    import re
    import subprocess
    import os

    exploit_path = state.get("exploit_path", "")
    if not exploit_path or not Path(exploit_path).exists():
        console.print("[red]No exploit found to verify[/red]")
        state["exploit_error"] = "No exploit file found"
        return state

    state["exploit_attempts"] = state.get("exploit_attempts", 0) + 1
    attempt = state["exploit_attempts"]
    max_attempts = state.get("max_exploit_attempts", 3)

    console.print(f"[cyan]Running exploit (attempt {attempt}/{max_attempts})...[/cyan]")

    # 익스플로잇 실행
    try:
        env = os.environ.copy()
        # Docker가 설정되어 있으면 Docker로 테스트
        if state.get("docker_port"):
            env["TARGET_HOST"] = "localhost"
            env["TARGET_PORT"] = str(state.get("docker_port", 1337))

        result = subprocess.run(
            ["python3", exploit_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(exploit_path).parent),
            env=env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + ("\n--- STDERR ---\n" + stderr if stderr else "")

    except subprocess.TimeoutExpired:
        combined = "Error: Exploit timed out after 60 seconds"
        console.print(f"[red]{combined}[/red]")
        state["exploit_error"] = combined
        return state

    except Exception as e:
        combined = f"Error: {str(e)}"
        console.print(f"[red]{combined}[/red]")
        state["exploit_error"] = combined
        return state

    # 출력 표시
    preview = combined[:1000] + "..." if len(combined) > 1000 else combined
    console.print(Panel(preview, title="Exploit Output", border_style="cyan"))

    # 플래그 감지
    flag_format = state.get("challenge", {}).get("flag_format", "FLAG{...}")
    flag_patterns = [
        r"flag\{[^}]+\}",
        r"FLAG\{[^}]+\}",
        r"ctf\{[^}]+\}",
        r"CTF\{[^}]+\}",
    ]
    # 커스텀 플래그 포맷 추가
    if flag_format and "{" in flag_format:
        prefix = flag_format.split("{")[0]
        if prefix:
            flag_patterns.insert(0, rf"{re.escape(prefix)}\{{[^}}]+\}}")

    for pattern in flag_patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            flag = match.group()
            console.print(f"[bold green]FLAG FOUND: {flag}[/bold green]")
            state["flag_detected"] = True
            state["detected_flag"] = flag
            state["exploit_verified"] = True
            if flag not in state.get("all_detected_flags", []):
                state.setdefault("all_detected_flags", []).append(flag)
            return state

    # 쉘 획득 여부 확인: exploit 스크립트가 쉘에서 'id' 명령을 실행하고
    # 그 결과(uid=)를 출력했는지 확인. 단순 문자열 매칭은 pwntools debug 로그에서
    # false positive가 발생하므로, "SHELL_VERIFIED" 마커 또는 실제 uid= 패턴만 신뢰.
    shell_verified_marker = "SHELL_VERIFIED"
    uid_pattern = re.search(r"uid=\d+", combined)
    if shell_verified_marker in combined or uid_pattern:
        console.print("[green]Shell access detected (verified via 'id' command)![/green]")
        state["exploit_verified"] = True
        return state

    # 실패 - 에러 저장 + 히스토리 누적
    console.print(f"[yellow]Exploit attempt {attempt} failed[/yellow]")
    state["exploit_error"] = combined
    state["exploit_verified"] = False

    # 실패 히스토리 누적 (exploit agent가 같은 실수 반복하지 않도록)
    exploit_code = ""
    if exploit_path and Path(exploit_path).exists():
        try:
            exploit_code = Path(exploit_path).read_text(encoding="utf-8")
        except Exception:
            pass

    failure_record = {
        "attempt": attempt,
        "error_summary": combined[:1000],
        "exploit_snippet": exploit_code[:500] if exploit_code else "",
    }
    state.setdefault("exploit_failure_history", []).append(failure_record)

    return state


def Refine_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Refine Node - 실패한 익스플로잇 수정

    - ExploitRefinerAgent로 에러 분석
    - 수정된 익스플로잇 생성
    """
    console.print(Panel("Refining Exploit", style="bold orange1"))

    from Agent.exploit import ExploitRefinerAgent

    exploit_path = state.get("exploit_path", "")
    exploit_error = state.get("exploit_error", "")

    if not exploit_path or not Path(exploit_path).exists():
        console.print("[red]No exploit to refine[/red]")
        return state

    # 현재 익스플로잇 코드 읽기
    current_exploit = Path(exploit_path).read_text(encoding="utf-8")

    # Refiner 실행
    refiner = ExploitRefinerAgent(model=model, provider=provider)
    state = refiner.run(state, error_output=exploit_error, current_exploit=current_exploit)

    return state


def Crash_analysis_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Crash Analysis Node - 익스플로잇 실패 시 Core dump 및 GDB 분석

    - Core dump 파일 찾기 및 분석
    - GDB로 크래시 원인 분석
    - 실패 원인을 state에 저장하여 Plan에서 활용
    """
    console.print(Panel("Crash Analysis", style="bold red"))

    import re
    import os

    binary_path = state.get("binary_path", "")
    exploit_error = state.get("exploit_error", "")
    exploit_path = state.get("exploit_path", "")

    if not binary_path:
        console.print("[red]No binary path set[/red]")
        return state

    workdir = Path(binary_path).resolve().parent
    challenge_dir = _challenge_dir_for_state(state)

    crash_analysis = {
        "performed": True,
        "core_found": False,
        "crash_reason": "",
        "registers": {},
        "stack_state": "",
        "exploit_issues": [],
        "recommendations": [],
    }

    # 1. Core dump 파일 찾기
    core_files = list(workdir.glob("core*")) + list(Path("/tmp").glob("core*"))
    core_file = None

    if core_files:
        # 가장 최근 core 파일 선택
        core_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        core_file = core_files[0]
        crash_analysis["core_found"] = True
        console.print(f"[green]Found core dump: {core_file}[/green]")

    # 2. GDB 분석 실행
    gdb_analysis = ""

    if core_file and core_file.exists():
        # Core dump 분석
        gdb_commands = [
            "set pagination off",
            f"file {binary_path}",
            f"core-file {core_file}",
            "info registers",
            "bt",  # backtrace
            "x/20gx $rsp",  # stack
            "x/10i $rip",   # instructions at crash
        ]

        cmd = ["gdb", "-q", "-nx", "-batch"]
        for c in gdb_commands:
            cmd += ["-ex", c]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            gdb_analysis = result.stdout or ""
            if result.stderr:
                gdb_analysis += "\n--- STDERR ---\n" + result.stderr
        except Exception as e:
            gdb_analysis = f"GDB analysis failed: {e}"

        console.print(Panel(gdb_analysis[:2000], title="GDB Core Analysis", border_style="red"))
    else:
        # Core가 없으면 exploit 에러 메시지 분석
        console.print("[yellow]No core dump found. Analyzing exploit error...[/yellow]")
        gdb_analysis = f"No core dump available.\n\nExploit error output:\n{exploit_error}"

        # Docker 모드: 컨테이너 로그 수집으로 보완
        if state.get("docker_port"):
            slug = Path(binary_path).stem.lower()
            slug = re.sub(r"[^a-z0-9._-]+", "_", slug).strip("_") or "unknown"
            container_name = f"pwn_{slug}_container"
            try:
                docker_result = subprocess.run(
                    ["docker", "logs", "--tail", "50", container_name],
                    capture_output=True, text=True, timeout=10,
                )
                docker_logs = (docker_result.stdout or "") + (docker_result.stderr or "")
                if docker_logs.strip():
                    gdb_analysis += f"\n\n--- Docker Container Logs ({container_name}) ---\n{docker_logs}"
                    console.print(Panel(docker_logs[:1000], title="Docker Logs", border_style="yellow"))
            except Exception:
                pass

    crash_analysis["gdb_output"] = gdb_analysis

    # 3. 크래시 원인 분석 (패턴 매칭)
    issues = []
    recommendations = []

    # 레지스터 값 파싱
    rip_match = re.search(r'rip\s+0x([0-9a-fA-F]+)', gdb_analysis)
    rsp_match = re.search(r'rsp\s+0x([0-9a-fA-F]+)', gdb_analysis)
    rbp_match = re.search(r'rbp\s+0x([0-9a-fA-F]+)', gdb_analysis)

    if rip_match:
        rip = rip_match.group(1)
        crash_analysis["registers"]["rip"] = f"0x{rip}"

        # RIP 분석
        rip_int = int(rip, 16)

        # 잘못된 주소로 점프
        if rip_int < 0x400000 or rip_int > 0x7fffffffffff:
            issues.append(f"RIP points to invalid address: 0x{rip}")
            recommendations.append("Check offset calculation - RIP is corrupted")

        # cyclic 패턴 감지 (0x616161XX 형태)
        if 0x61616161 <= (rip_int & 0xffffffff) <= 0x6161617a:
            issues.append("RIP contains cyclic pattern - offset might be wrong")
            # 패턴에서 오프셋 계산 시도
            try:
                from pwn import cyclic_find
                offset = cyclic_find(rip_int & 0xffffffff, n=4)
                if offset != -1:
                    recommendations.append(f"Cyclic pattern detected. Calculated offset: {offset}")
            except:
                recommendations.append("Install pwntools to auto-calculate offset from cyclic pattern")

    if rsp_match:
        crash_analysis["registers"]["rsp"] = f"0x{rsp_match.group(1)}"

    if rbp_match:
        crash_analysis["registers"]["rbp"] = f"0x{rbp_match.group(1)}"

    # 에러 메시지 분석
    error_lower = exploit_error.lower()

    if "segmentation fault" in error_lower or "sigsegv" in error_lower:
        issues.append("Segmentation fault - likely wrong address or offset")

    if "stack smashing" in error_lower or "stack_chk_fail" in error_lower:
        issues.append("Stack canary detected - need to leak and preserve canary")
        recommendations.append("Find format string or other leak to get canary value")

    if "timeout" in error_lower:
        issues.append("Exploit timed out - might be stuck or wrong interaction")
        recommendations.append("Check recv/send sequence matches binary behavior")

    if "eof" in error_lower or "got eof" in error_lower:
        issues.append("Connection closed unexpectedly")
        recommendations.append("Binary crashed or rejected input - check payload")

    if "broken pipe" in error_lower:
        issues.append("Broken pipe - binary crashed before sending data")
        recommendations.append("Exploit crashes too early - verify offset and addresses")

    # Alignment 문제 감지
    if "movaps" in gdb_analysis.lower() or "alignment" in error_lower:
        issues.append("Stack alignment issue (SIGSEGV on movaps)")
        recommendations.append("Add extra 'ret' gadget before function call for 16-byte alignment")

    # 기본 권장사항
    if not recommendations:
        recommendations.append("Verify offset with cyclic pattern test")
        recommendations.append("Check if addresses are correct (PIE? ASLR?)")
        recommendations.append("Ensure stack is 16-byte aligned before calls")

    crash_analysis["exploit_issues"] = issues
    crash_analysis["recommendations"] = recommendations
    crash_analysis["crash_reason"] = "; ".join(issues) if issues else "Unknown crash reason"

    # 4. 결과 표시
    if issues:
        console.print("[bold red]Detected Issues:[/bold red]")
        for issue in issues:
            console.print(f"  • {issue}")

    if recommendations:
        console.print("[bold yellow]Recommendations:[/bold yellow]")
        for rec in recommendations:
            console.print(f"  → {rec}")

    # 5. LLM으로 crash 분류 (minor fix vs strategy change)
    from Agent.base import BaseAgent, parse_json_response

    class CrashClassifier(BaseAgent):
        def __init__(self, **kwargs):
            super().__init__(name="CrashClassifier", **kwargs)
        def run(self, state):
            return state

    classifier = CrashClassifier(model=model, provider=provider)
    classifier.set_system_prompt(
        "You classify exploit crash reports. Output STRICT JSON only.\n"
        '{"fix_type": "minor" or "strategy", "reasoning": "brief explanation"}\n\n'
        "minor = offset wrong, alignment issue, wrong address, null byte, timeout, EOF, broken pipe, "
        "small calculation error. These can be fixed by adjusting values.\n"
        "strategy = wrong exploitation technique, missing leak, wrong target function, "
        "fundamentally wrong approach. Needs full re-planning."
    )
    classifier.add_user_message(
        f"Issues: {'; '.join(issues)}\n"
        f"Recommendations: {'; '.join(recommendations)}\n"
        f"Registers: {crash_analysis['registers']}\n"
        f"Error excerpt: {exploit_error[:500]}"
    )
    try:
        classify_response = classifier.call_llm()
        classify_parsed = parse_json_response(classify_response)
        fix_type = classify_parsed.get("fix_type", "minor")
        console.print(f"[cyan]LLM Crash Classification: {fix_type}[/cyan]")
        console.print(f"[dim]Reasoning: {classify_parsed.get('reasoning', '')}[/dim]")
    except Exception:
        fix_type = "minor"  # fallback to minor

    crash_analysis["fix_type"] = fix_type

    # 6. State에 저장
    state["crash_analysis"] = crash_analysis

    # Feedback에 전달할 정보 업데이트
    state["exploit_failure_reason"] = crash_analysis["crash_reason"]

    # 분석 문서에 실패 정보 추가
    if "analysis" not in state:
        state["analysis"] = init_analysis()

    state["analysis"]["last_exploit_failure"] = {
        "crash_reason": crash_analysis["crash_reason"],
        "issues": issues,
        "recommendations": recommendations,
        "registers": crash_analysis["registers"],
    }

    # 결과 파일 저장
    try:
        analysis_text = f"""=== CRASH ANALYSIS ===

Core dump found: {crash_analysis['core_found']}
Crash reason: {crash_analysis['crash_reason']}

Registers:
{crash_analysis.get('registers', {})}

Issues:
{chr(10).join('- ' + i for i in issues)}

Recommendations:
{chr(10).join('- ' + r for r in recommendations)}

=== GDB Output ===
{gdb_analysis}

=== Exploit Error ===
{exploit_error}
"""
        (challenge_dir / "crash_analysis.txt").write_text(analysis_text, encoding="utf-8")
    except Exception:
        pass

    # Core 파일 정리 (선택적)
    # if core_file and core_file.exists():
    #     core_file.unlink()

    return state


# =============================================================================
# Routing Functions (for LangGraph)
# =============================================================================

def should_continue(state: State) -> str:
    """Determine next node based on state."""
    if state.get("flag_detected"):
        return "end"

    if not state.get("loop", True):
        exploit_readiness = state.get("exploit_readiness", {})
        if exploit_readiness.get("recommend_exploit", False):
            return "exploit"
        return "end"

    return "plan"


def route_after_verify(state: State) -> str:
    """Verify 후 라우팅 결정 - Crash Analysis로 이동."""
    # 성공
    if state.get("flag_detected") or state.get("exploit_verified"):
        return "end"

    # 최대 시도 횟수 초과
    attempts = state.get("exploit_attempts", 0)
    max_attempts = state.get("max_exploit_attempts", 3)

    if attempts >= max_attempts:
        console.print(f"[red]Max exploit attempts ({max_attempts}) reached[/red]")
        return "end"

    # 실패 - crash analysis로 이동 후 plan으로 돌아감
    console.print("[yellow]Exploit failed - going to crash analysis...[/yellow]")
    return "crash_analysis"


# =============================================================================
# Staged Exploit Nodes (단계별 익스플로잇)
# =============================================================================

def Stage_identify_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Stage Identify Node - 분석 결과 기반으로 exploit 단계 식별.
    analysis loop 종료 후 한 번 실행.
    """
    console.print(Panel("Stage Identify", style="bold magenta"))

    from Agent.exploit import StageIdentifierAgent

    agent = StageIdentifierAgent(model=model, provider=provider)
    state = agent.run(state)
    return state


def Stage_exploit_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Stage Exploit Node - 현재 단계의 exploit 코드 생성.
    이전 verified 단계 코드를 포함한 완전한 스크립트.
    """
    staged = state.get("staged_exploit", {})
    stages = staged.get("stages", [])
    current_idx = staged.get("current_stage_index", 0)

    if current_idx >= len(stages):
        console.print("[red]All stages complete or no stages defined[/red]")
        return state

    console.print(Panel(
        f"Stage Exploit ({current_idx+1}/{len(stages)})",
        style="bold red"
    ))

    from Agent.exploit import StageExploitAgent

    agent = StageExploitAgent(model=model, provider=provider)
    state = agent.run(state)

    console.print(f"[green]Completed: Stage {current_idx+1} code generation[/green]")
    return state


def Stage_verify_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Stage Verify Node - 현재 단계의 스크립트를 실행하고 검증.

    중간 단계: verification_marker in stdout → verified
    최종 단계: SHELL_VERIFIED or uid=\\d+ → verified + exploit_verified
    """
    import re
    import os

    console.print(Panel("Stage Verify", style="bold yellow"))

    staged = state.get("staged_exploit", {})
    stages = staged.get("stages", [])
    current_idx = staged.get("current_stage_index", 0)

    if current_idx >= len(stages):
        console.print("[red]No stage to verify[/red]")
        return state

    current_stage = stages[current_idx]
    exploit_path = state.get("exploit_path", "")

    if not exploit_path or not Path(exploit_path).exists():
        console.print("[red]No exploit file to run[/red]")
        current_stage["verified"] = False
        current_stage["error"] = "No exploit file found"
        stages[current_idx] = current_stage
        state["staged_exploit"]["stages"] = stages
        return state

    console.print(f"[cyan]Running Stage {current_idx+1}: {current_stage['stage_id']}...[/cyan]")

    # Run the script
    try:
        env = os.environ.copy()
        # Docker 모드: TARGET_HOST/TARGET_PORT 환경변수 주입
        if state.get("docker_port"):
            env["TARGET_HOST"] = "localhost"
            env["TARGET_PORT"] = str(state.get("docker_port", 1337))
        result = subprocess.run(
            ["python3", exploit_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(exploit_path).parent),
            env=env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + ("\n--- STDERR ---\n" + stderr if stderr else "")
    except subprocess.TimeoutExpired:
        combined = "TIMEOUT: Exploit timed out after 60 seconds"
        console.print(f"[red]{combined}[/red]")
        current_stage["verified"] = False
        current_stage["error"] = combined
        stages[current_idx] = current_stage
        state["staged_exploit"]["stages"] = stages
        return state
    except Exception as e:
        combined = f"ERROR: {str(e)}"
        console.print(f"[red]{combined}[/red]")
        current_stage["verified"] = False
        current_stage["error"] = combined
        stages[current_idx] = current_stage
        state["staged_exploit"]["stages"] = stages
        return state

    # Display output
    preview = combined[:1500] + "..." if len(combined) > 1500 else combined
    console.print(Panel(preview, title=f"Stage {current_idx+1} Output", border_style="cyan"))

    current_stage["output"] = combined

    # Check for flag in any stage
    flag_patterns = [r"flag\{[^}]+\}", r"FLAG\{[^}]+\}", r"ctf\{[^}]+\}", r"CTF\{[^}]+\}"]
    flag_format = state.get("challenge", {}).get("flag_format", "")
    if flag_format and "{" in flag_format:
        prefix = flag_format.split("{")[0]
        if prefix:
            flag_patterns.insert(0, rf"{re.escape(prefix)}\{{[^}}]+\}}")

    for pattern in flag_patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            flag = match.group()
            console.print(f"[bold green]FLAG FOUND: {flag}[/bold green]")
            state["flag_detected"] = True
            state["detected_flag"] = flag
            state["exploit_verified"] = True
            current_stage["verified"] = True
            state["staged_exploit"]["all_stages_verified"] = True
            if flag not in state.get("all_detected_flags", []):
                state.setdefault("all_detected_flags", []).append(flag)
            stages[current_idx] = current_stage
            state["staged_exploit"]["stages"] = stages
            return state

    # Verify based on marker
    marker = current_stage.get("verification_marker", "")

    if marker == "SHELL_VERIFIED":
        # Final stage: check for actual shell verification
        uid_match = re.search(r"uid=\d+", combined)
        if "SHELL_VERIFIED" in combined or uid_match:
            console.print(f"[bold green]Stage {current_idx+1} VERIFIED (shell obtained!)[/bold green]")
            current_stage["verified"] = True
            state["exploit_verified"] = True
            state["staged_exploit"]["all_stages_verified"] = True
        else:
            console.print(f"[red]Stage {current_idx+1} FAILED (no shell)[/red]")
            current_stage["verified"] = False
            current_stage["error"] = combined
    else:
        # Intermediate stage: check for marker string
        # Also check for explicit STAGE_FAILED marker (code validated and found invalid result)
        has_marker = marker and marker in combined
        has_stage_failed = "STAGE_FAILED" in combined

        if has_stage_failed:
            # Code explicitly reported failure (e.g., leaked value validation failed)
            console.print(f"[red]Stage {current_idx+1} FAILED (code reported STAGE_FAILED)[/red]")
            current_stage["verified"] = False
            current_stage["error"] = combined
        elif has_marker:
            # Additional heuristic: for leak stages, check if a valid address is near the marker
            is_leak_stage = current_stage.get("stage_id", "").lower() in ("leak", "canary_leak")
            if is_leak_stage:
                # Extract the marker line and nearby context
                marker_line = ""
                for line in combined.split("\n"):
                    if marker in line:
                        marker_line = line
                        break
                # Check if marker line contains a hex value that looks valid (not 0x0 or very small)
                addr_match = re.search(r"0x([0-9a-fA-F]+)", marker_line)
                if addr_match:
                    addr_val = int(addr_match.group(1), 16)
                    if addr_val < 0x1000:
                        console.print(f"[red]Stage {current_idx+1} FAILED (leaked value 0x{addr_val:x} is too small — likely invalid)[/red]")
                        current_stage["verified"] = False
                        current_stage["error"] = f"Leaked value 0x{addr_val:x} is invalid (too small)\n" + combined
                    else:
                        console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found, addr=0x{addr_val:x})[/bold green]")
                        current_stage["verified"] = True
                else:
                    # Marker found but no hex value — still accept but warn
                    console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found)[/bold green]")
                    current_stage["verified"] = True
            else:
                console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found)[/bold green]")
                current_stage["verified"] = True
        else:
            console.print(f"[red]Stage {current_idx+1} FAILED (marker '{marker}' not found)[/red]")
            current_stage["verified"] = False
            current_stage["error"] = combined

    stages[current_idx] = current_stage
    state["staged_exploit"]["stages"] = stages

    return state


def Stage_refine_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Stage Refine Node - 실패한 단계의 코드만 수정.
    """
    console.print(Panel("Stage Refine", style="bold orange1"))

    staged = state.get("staged_exploit", {})
    stages = staged.get("stages", [])
    current_idx = staged.get("current_stage_index", 0)

    if current_idx >= len(stages):
        return state

    current_stage = stages[current_idx]
    current_stage["refinement_attempts"] = current_stage.get("refinement_attempts", 0) + 1

    from Agent.exploit import StageRefinerAgent

    refiner = StageRefinerAgent(model=model, provider=provider)
    state = refiner.run(
        state=state,
        stage=current_stage,
        error_output=current_stage.get("error", ""),
    )

    return state


def Stage_advance_node(
    state: State,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> State:
    """
    Stage Advance Node - 다음 단계로 이동 (current_stage_index += 1).
    """
    staged = state.get("staged_exploit", {})
    old_idx = staged.get("current_stage_index", 0)
    staged["current_stage_index"] = old_idx + 1
    state["staged_exploit"] = staged

    stages = staged.get("stages", [])
    new_idx = staged["current_stage_index"]

    if new_idx < len(stages):
        next_stage = stages[new_idx]
        console.print(f"[cyan]Advancing to Stage {new_idx+1}: {next_stage['stage_id']} — {next_stage['description']}[/cyan]")
    else:
        console.print("[green]All stages complete![/green]")

    return state


# =============================================================================
# Backward Compatibility (for existing code)
# =============================================================================

def call_llm(messages, model: str = "gpt-4o") -> str:
    """Legacy function - use Agent.base.get_llm_client() instead."""
    from Agent.base import get_llm_client
    client = get_llm_client()
    response, _ = client.chat(messages=messages, model=model)
    return response


def parse_json_response(text: str) -> Dict[str, Any]:
    """Legacy function - use Agent.base.parse_json_response() instead."""
    from Agent.base import parse_json_response as _parse
    return _parse(text)
