import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

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
    out_dir = root / _slug(name)
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

    - First iteration: Initialize with directory listing
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
    else:
        state["iteration_count"] = state.get("iteration_count", 0) + 1

    console.print(f"[dim]Iteration: {state['iteration_count']}[/dim]")

    # Run Plan Agent
    agent = _get_agent("plan", model=model, provider=provider)
    state = agent.run(state)

    return state


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

    # 쉘 획득 여부 확인 (interactive shell indicators)
    shell_indicators = ["$ ", "# ", "sh-", "bash", "/bin/sh", "id=", "uid="]
    if any(indicator in combined.lower() for indicator in shell_indicators):
        console.print("[green]Shell access detected![/green]")
        state["exploit_verified"] = True
        return state

    # 실패 - 에러 저장
    console.print(f"[yellow]Exploit attempt {attempt} failed[/yellow]")
    state["exploit_error"] = combined
    state["exploit_verified"] = False

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

    # 5. State에 저장
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
