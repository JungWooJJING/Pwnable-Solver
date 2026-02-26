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
from Agent.base import LLMRetryExhausted

console = Console()


# =============================================================================
# Helper Functions
# =============================================================================

def _trim_output_for_context(
    output: str,
    max_chars: int = 8000,
    head_chars: int = 500,
) -> str:
    """
    긴 실행 출력을 LLM context에 맞게 trim.
    앞부분(설정/초기화 정보) + 뒷부분(실제 실패 지점)을 유지하고 중간을 생략.

    Args:
        output: 원본 출력 문자열
        max_chars: 최대 허용 길이
        head_chars: 앞에서 보존할 글자 수 (초기 컨텍스트용)

    Returns:
        Trimmed output string
    """
    if len(output) <= max_chars:
        return output

    tail_chars = max_chars - head_chars
    omitted = len(output) - max_chars
    head = output[:head_chars]
    tail = output[-tail_chars:]
    return f"{head}\n\n[... {omitted} chars omitted ...]\n\n{tail}"

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


def _binary_for_dynamic_analysis(state: State) -> str:
    """
    동적 분석(GDB, 코어 덤프, 바이너리 프로브)에 쓸 바이너리 경로.
    pwninit 패치가 있으면 패치 바이너리, 없으면 원본. 범용.
    """
    return (state.get("patched_binary") or state.get("binary_path") or "").strip()


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

    # --- Guard: restore binary_path from original if LLM drifted it ---
    orig_binary = state.get("original_binary_path", "")
    if orig_binary and state.get("binary_path", "") != orig_binary:
        console.print(f"[yellow]⚠ binary_path drifted → restoring to original: {orig_binary}[/yellow]")
        state["binary_path"] = orig_binary

    binary_path = state.get("binary_path", "")

    # First iteration initialization — use recon_done to guard RECON,
    # NOT loop, because Feedback sets loop=False when ready for exploit
    # and the dynamic_verification gate can route back to Plan with loop=False.
    if not state.get("recon_done", False):
        console.print("[green]Initializing Plan Node...[/green]")

        # Lock the binary_path so LLMs cannot change it later
        if binary_path and not state.get("original_binary_path"):
            state["original_binary_path"] = binary_path

        # Get directory listing (filter core dumps to prevent LLM confusion)
        if binary_path:
            workdir = str(Path(binary_path).resolve().parent)
            ls_result = subprocess.run(
                ["ls", "-la"],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            raw_listing = ls_result.stdout or ""
            # Filter out core dump files and pwntools _patched copies so the LLM
            # doesn't mistake them for real binaries (core.* / *_patched / *_ghidra dirs)
            filtered_lines = [
                line for line in raw_listing.splitlines()
                if not any(
                    part in line
                    for part in ("_patched", "_ghidra", " core.", "/core.")
                ) or line.startswith("total")
            ]
            state["directory_listing"] = "\n".join(filtered_lines)
            state["cwd"] = workdir

            # Save full (unfiltered) listing to disk for debugging
            try:
                out_dir = _challenge_dir_for_state(state)
                (out_dir / "directory_listing.txt").write_text(
                    raw_listing +
                    ("\n\n--- STDERR ---\n" + (ls_result.stderr or "") if ls_result.stderr else ""),
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                pass

        # Initialize analysis if needed
        if not state.get("analysis"):
            state["analysis"] = init_analysis()

        state["recon_done"] = True
        state["loop"] = True
        state["iteration_count"] = 1

        # --- Phase 1 RECON: Auto-execute checksec + ghidra ---
        if binary_path:
            _run_phase1_recon(state, binary_path)

    else:
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        # Ensure loop stays True during re-analysis after stage failures
        if not state.get("loop"):
            state["loop"] = True

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
                    "functions": [{"name": "main", "address": "", "code": str(ghidra_raw)[:8000]}],
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

            # Shellcode size limit: read(0, buf, N) 패턴에서 N 추출
            try:
                import re as _re_sc
                _rsc = _re_sc.compile(
                    r'\bread\s*\(\s*(?:0|0x0)\s*,\s*[^,)]+,\s*(0x[0-9a-fA-F]+|\d+)\s*\)'
                )
                _sc_matches = _rsc.findall(str(ghidra_raw))
                _sc_sizes = []
                for _m in _sc_matches:
                    try:
                        _s = int(_m, 16) if _m.startswith("0x") else int(_m)
                        if 8 <= _s <= 4096:  # 합리적인 shellcode 크기 범위
                            _sc_sizes.append(_s)
                    except ValueError:
                        pass
                if _sc_sizes:
                    _sc_limit = min(_sc_sizes)
                    _facts = state.get("facts", {})
                    _facts["shellcode_size_limit"] = _sc_limit
                    state["facts"] = _facts
                    console.print(
                        f"[green]Shellcode size limit detected: {_sc_limit} (0x{_sc_limit:x}) bytes[/green]"
                    )
            except Exception:
                pass
        else:
            console.print(f"[yellow]Ghidra returned: {str(ghidra_raw)[:200]}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Ghidra failed: {e}[/yellow]")

    # --- Symbol Scan: win function + all function addresses ---
    try:
        import subprocess as _sp
        import re as _re

        # Try nm first (works on non-stripped), fallback to readelf -s
        sym_result = _sp.run(
            ["nm", "--defined-only", binary_path],
            capture_output=True, text=True, timeout=10
        )
        if sym_result.returncode != 0 or not sym_result.stdout.strip():
            sym_result = _sp.run(
                ["readelf", "-s", binary_path],
                capture_output=True, text=True, timeout=10
            )

        sym_raw = sym_result.stdout
        sym_lower = sym_raw.lower()

        # Parse function symbols: "addr T funcname" or readelf format
        # nm format:  0000000000401156 T win
        # readelf format: ... FUNC ... win
        func_map: dict = {}
        for line in sym_raw.splitlines():
            # nm format
            nm_match = _re.match(r'([0-9a-fA-F]+)\s+[Tt]\s+(\S+)', line)
            if nm_match:
                addr_hex, fname = nm_match.group(1), nm_match.group(2)
                func_map[fname.lower()] = int(addr_hex, 16)
                continue
            # readelf format: "  Num: addr size FUNC ... name"
            re_match = _re.search(r':\s+([0-9a-fA-F]+)\s+\d+\s+FUNC\s+\S+\s+\S+\s+\S+\s+(\S+)', line)
            if re_match:
                addr_hex, fname = re_match.group(1), re_match.group(2)
                if int(addr_hex, 16) > 0:
                    func_map[fname.lower()] = int(addr_hex, 16)

        WIN_KEYWORDS = ["win", "flag", "shell", "backdoor", "get_shell", "spawn_shell"]
        found_win = any(kw in sym_lower for kw in WIN_KEYWORDS)

        if found_win:
            # Find the first matching win function name and its address
            matched_names = [kw for kw in WIN_KEYWORDS if kw in sym_lower]
            win_addr = 0
            win_name = ""
            for kw in WIN_KEYWORDS:
                for fname, addr in func_map.items():
                    if kw in fname:
                        win_addr = addr
                        win_name = fname
                        break
                if win_addr:
                    break

            win_addr_str = f"0x{win_addr:x}" if win_addr else "(address unknown)"
            console.print(f"[bold green]Win function detected: {matched_names} → {win_name} @ {win_addr_str}[/bold green]")

            # Store win function address via merge so all fields are consistent
            merge_analysis_updates(state, {
                "win_function": True,
                "win_function_name": win_name,
                "win_function_addr": win_addr_str,
            })

            # Re-run KB strategy with win_function=True so the right strategy is set
            try:
                from Store.knowledge import get_checksec_guide
                checksec_data = state.get("analysis", {}).get("checksec", {})
                guide = get_checksec_guide(checksec_data, win_function=True)
                if guide.get("recommended"):
                    top = guide["recommended"][0]
                    strategy = f"Recommended: {top.get('name', '')} ({top.get('description', '')})"
                    merge_analysis_updates(state, {"strategy": strategy})
                    console.print(f"[bold green]KB Strategy updated (win function): {strategy}[/bold green]")
            except Exception:
                pass
        else:
            console.print("[cyan]No win function found in symbols[/cyan]")

        # Log all found functions for Plan agent context
        if func_map:
            fn_summary = ", ".join(f"{n}@0x{a:x}" for n, a in list(func_map.items())[:20])
            console.print(f"[dim]Functions found: {fn_summary}[/dim]")
            # Store function list as a known fact
            facts = state.get("facts", {})
            facts["function_symbols"] = {n: f"0x{a:x}" for n, a in func_map.items()}
            state["facts"] = facts

    except Exception as e:
        console.print(f"[yellow]Symbol scan failed: {e}[/yellow]")

    # --- Ghidra: Critical Security Functions Auto-Decompile ---
    # sandbox, init, setup 등 보안 관련 함수를 자동으로 decompile하여 LLM에 전달.
    # 바이너리 심볼 스캔에서 발견된 함수 중 critical 키워드와 일치하는 것만 처리.
    try:
        CRITICAL_FUNC_KEYWORDS = [
            "sandbox", "init", "setup", "initialize",
            "filter", "validate", "check", "verify",
            "seccomp", "prctl",
        ]
        function_symbols = state.get("facts", {}).get("function_symbols", {})

        # 이미 decompile된 함수 이름 집합 (중복 방지)
        already_decompiled = {
            f["name"].lower()
            for f in state.get("analysis", {}).get("decompile", {}).get("functions", [])
        }

        # func_map에서 critical 키워드를 포함하는 함수 탐지
        critical_funcs_found = []
        for fname in function_symbols:
            fname_lower = fname.lower()
            if fname_lower in already_decompiled:
                continue
            for kw in CRITICAL_FUNC_KEYWORDS:
                if kw in fname_lower:
                    critical_funcs_found.append(fname)
                    break

        if critical_funcs_found:
            console.print(f"[cyan]Critical security functions detected: {critical_funcs_found} → decompiling...[/cyan]")
            decompiled_funcs = []
            for fname in critical_funcs_found[:5]:  # 최대 5개 제한 (성능)
                try:
                    raw = tool.Ghidra_decompile_function(function_name=fname)
                    if raw and not str(raw).startswith("Error:"):
                        decompiled_funcs.append({
                            "name": fname,
                            "address": function_symbols.get(fname, ""),
                            "code": str(raw)[:4000],
                        })
                        console.print(f"[green]Decompiled critical: {fname}[/green]")
                    else:
                        console.print(f"[yellow]Critical decompile returned: {str(raw)[:100]}[/yellow]")
                except Exception as _e:
                    console.print(f"[yellow]Critical decompile error ({fname}): {_e}[/yellow]")

            if decompiled_funcs:
                merge_analysis_updates(state, {
                    "decompile": {
                        "done": True,
                        "functions": decompiled_funcs,
                    }
                })
                names_str = ", ".join(f["name"] for f in decompiled_funcs)
                console.print(f"[bold green]Phase 1: Critical functions decompiled: {names_str}[/bold green]")
                add_run(state, {
                    "run_id": f"recon_ghidra_critical_{int(_time.time()*1000)}",
                    "task_id": "phase1_ghidra_critical",
                    "commands": [f"Ghidra_decompile_function({f['name']})" for f in decompiled_funcs],
                    "success": True,
                    "stdout": "\n\n".join(f"=== {f['name']} ===\n{f['code'][:500]}" for f in decompiled_funcs),
                    "stderr": "",
                    "started_at": _time.time(),
                    "finished_at": _time.time(),
                    "key_findings": [f"Critical functions decompiled: {names_str}"],
                })
    except Exception as e:
        console.print(f"[dim]Critical function decompile skipped: {e}[/dim]")

    # --- Source Code Discovery ---
    # Read any .c source files in the binary's directory and store in state so all
    # agents (Plan, Instruction, Stage Exploit) can use the ground-truth I/O protocol.
    try:
        import glob as _g
        workdir = str(Path(binary_path).resolve().parent)
        for c_file in _g.glob(str(Path(workdir) / "*.c")):
            source_code = Path(c_file).read_text(encoding="utf-8", errors="replace")[:5000]
            facts = state.get("facts", {})
            facts["source_code"] = source_code
            facts["source_code_file"] = str(c_file)
            state["facts"] = facts
            console.print(f"[green]Source code found: {c_file}[/green]")
            # Also store in analysis so LLM agents can access it easily
            merge_analysis_updates(state, {"source_code_available": True, "source_code_path": str(c_file)})
            break
    except Exception:
        pass

    # --- ROPgadget: 핵심 ROP 가젯 자동 추출 ---
    # NX가 활성화된 바이너리에서 pop rdi / ret 주소를 확정적으로 추출하여
    # LLM이 매번 다른 주소를 추측하는 진동 현상을 방지.
    try:
        checksec_data = state.get("analysis", {}).get("checksec", {})
        if checksec_data.get("nx", False):
            rop_result = _sp.run(
                ["ROPgadget", "--binary", binary_path, "--rop"],
                capture_output=True, text=True, timeout=30,
            )
            if rop_result.returncode == 0 and rop_result.stdout:
                from Agent.parsing import parse_ropgadget_output
                gadgets = parse_ropgadget_output(rop_result.stdout)
                rop_map: dict = {}
                for g in gadgets:
                    instr = g.get("instruction", "").strip()
                    addr  = g.get("address", "")
                    # 핵심 가젯만 추출 (범용)
                    if instr in ("pop rdi ; ret", "pop rdi; ret"):
                        rop_map["pop_rdi"] = addr
                    elif instr in ("pop rsi ; ret", "pop rsi; ret"):
                        rop_map["pop_rsi"] = addr
                    elif instr in ("pop rdx ; ret", "pop rdx; ret"):
                        rop_map["pop_rdx"] = addr
                    elif instr == "ret":
                        rop_map.setdefault("ret", addr)  # 스택 정렬용
                if rop_map:
                    facts = state.get("facts", {})
                    facts["rop_gadgets"] = rop_map
                    state["facts"] = facts
                    summary = ", ".join(f"{k}={v}" for k, v in rop_map.items())
                    console.print(f"[bold green]ROPgadget: {summary}[/bold green]")
                    add_run(state, {
                        "run_id": f"recon_ropgadget_{int(_time.time()*1000)}",
                        "task_id": "phase1_ropgadget",
                        "commands": ["ROPgadget --rop"],
                        "success": True,
                        "stdout": rop_result.stdout[:3000],
                        "stderr": "",
                        "started_at": _time.time(),
                        "finished_at": _time.time(),
                        "key_findings": [f"ROP gadgets: {summary}"],
                    })
    except Exception as e:
        console.print(f"[dim]ROPgadget scan skipped: {e}[/dim]")

    # --- pwninit: libc 파일이 있으면 바이너리 패치 ---
    try:
        import glob as _glob
        workdir = str(Path(binary_path).resolve().parent)
        patched_path = Path(workdir) / (Path(binary_path).stem + "_patched")

        # libc 파일 탐지: libc*.so* 패턴 (libc.so.6, libc-2.23.so, libc.so 등)
        libc_candidates = _glob.glob(str(Path(workdir) / "libc*.so*"))
        libc_file = Path(libc_candidates[0]) if libc_candidates else None

        if patched_path.exists():
            # 이미 패치된 바이너리가 있으면 그냥 등록
            state["patched_binary"] = str(patched_path)
            console.print(f"[green]Existing patched binary found: {patched_path}[/green]")
        elif libc_file:
            console.print(f"[cyan]libc detected ({libc_file.name}) → running pwninit...[/cyan]")
            pwninit_result = _sp.run(
                ["pwninit", "--bin", binary_path, "--libc", str(libc_file)],
                capture_output=True, text=True, timeout=30, cwd=workdir,
            )
            if pwninit_result.returncode == 0 and patched_path.exists():
                state["patched_binary"] = str(patched_path)
                console.print(f"[bold green]pwninit success → patched binary: {patched_path}[/bold green]")
                add_run(state, {
                    "run_id": f"recon_pwninit_{int(_time.time()*1000)}",
                    "task_id": "phase1_pwninit",
                    "commands": [f"pwninit --bin {binary_path} --libc {libc_file.name}"],
                    "success": True,
                    "stdout": pwninit_result.stdout[:2000],
                    "stderr": pwninit_result.stderr[:500],
                    "started_at": _time.time(),
                    "finished_at": _time.time(),
                    "key_findings": [f"Patched binary created: {patched_path.name} (libc: {libc_file.name})"],
                })
            else:
                console.print(f"[yellow]pwninit failed (rc={pwninit_result.returncode}): {pwninit_result.stderr[:200]}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]pwninit skipped: {e}[/yellow]")

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
    result = agent.run(state)

    # ── GDB 강제 카운트 증가 (routing function에서는 state 변경이 반영 안 됨) ──
    # route_after_feedback()가 "plan"으로 돌려보낼 GDB 강제 조건을 미리 검사해서
    # gdb_forced_count를 여기서 증가시킴. routing function은 읽기 전용으로 유지.
    exploit_readiness = result.get("exploit_readiness", {})
    if (
        not result.get("loop", True)
        and not result.get("flag_detected", False)
        and exploit_readiness.get("recommend_exploit", False)
    ):
        analysis = result.get("analysis", {})
        result_str = str(analysis.get("checksec", {}).get("result", "")).lower()
        has_canary = "canary found" in result_str and "no canary" not in result_str
        has_pie = "pie enabled" in result_str and "no pie" not in result_str

        if has_canary or has_pie:
            dv = analysis.get("dynamic_verification", {})
            vulns = analysis.get("vulnerabilities", [])
            vuln_types = {v.get("type", "").lower() for v in vulns}
            strategy_lower = str(analysis.get("strategy", "")).lower()
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
                dv_complete = dv.get("verified") is True

            came_from_stage_failure = bool(result.get("exploit_failure_context"))

            if not dv_complete and not came_from_stage_failure and not skip_dv_for_path:
                # GDB 강제 조건 충족 → 카운트 증가 (routing function이 읽을 수 있도록)
                result["gdb_forced_count"] = result.get("gdb_forced_count", 0) + 1

    return result


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
    analysis_binary = _binary_for_dynamic_analysis(state) or binary_path
    exploit_error = state.get("exploit_error", "")
    exploit_path = state.get("exploit_path", "")

    if not analysis_binary:
        console.print("[red]No binary path set[/red]")
        return state

    workdir = Path(analysis_binary).resolve().parent
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
        # Core dump 분석 (동적 분석용 바이너리 사용 — 패치 바이너리로 크래시했으면 일치시킴)
        gdb_commands = [
            "set pagination off",
            f"file {analysis_binary}",
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
            slug = Path(analysis_binary).stem.lower()
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

    # --- Deterministic PIE stack scan (before LLM) ---
    # For PIE + win-function binaries, run GDB at break main and scan the stack
    # for binary-range addresses. This gives us the exact buf_offset of the PIE
    # pointer so we can compute pattern_len deterministically.
    analysis = state.get("analysis", {})
    _result_str = str(analysis.get("checksec", {}).get("result", "")).lower()
    _has_pie = "pie enabled" in _result_str and "no pie" not in _result_str
    _has_win = analysis.get("win_function", False)

    if _has_pie and _has_win and not analysis.get("pie_stack_offsets"):
        analysis_binary = _binary_for_dynamic_analysis(state) or state.get("binary_path", "")
        if analysis_binary:
            console.print("[cyan]PIE stack scan: scanning stack at break main for binary-range addresses...[/cyan]")
            try:
                _offsets = _find_pie_stack_offsets(state, analysis_binary)
                if _offsets:
                    console.print(
                        f"[bold green]PIE stack scan found {len(_offsets)} address(es): "
                        + ", ".join(f"buf+{o['buf_offset']}=0x{o['addr']:x}" for o in _offsets)
                        + "[/bold green]"
                    )
                    state.setdefault("analysis", {})["pie_stack_offsets"] = _offsets
                else:
                    console.print("[yellow]PIE stack scan: no binary-range addresses found in stack[/yellow]")
            except Exception as _e:
                console.print(f"[yellow]PIE stack scan failed: {_e}[/yellow]")

    from Agent.exploit import StageIdentifierAgent

    agent = StageIdentifierAgent(model=model, provider=provider)
    try:
        state = agent.run(state)
    except LLMRetryExhausted as e:
        console.print(f"[red]Stage identify API exhausted: {e}. Returning state unchanged.[/red]")
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
    try:
        state = agent.run(state)
        console.print(f"[green]Completed: Stage {current_idx+1} code generation[/green]")
    except LLMRetryExhausted as e:
        console.print(f"[red]Stage exploit API exhausted: {e}. Returning state unchanged.[/red]")
    return state


def _looks_like_ascii_text(val: int, width: int = 8) -> bool:
    """
    Returns True if val's bytes are mostly printable ASCII (likely reading prompt text
    instead of a real binary value such as a stack canary or address).

    Real canary: first byte \\x00, remaining 7 bytes random (rarely all printable).
    Real address: contains many high-byte values (0x7f... etc.), rarely all printable.
    """
    try:
        raw = val.to_bytes(width, "little")
    except (OverflowError, ValueError):
        return False
    printable = sum(1 for b in raw if 0x20 <= b <= 0x7e or b in (0x09, 0x0a, 0x0d))
    return printable >= width - 1  # ≥ 7/8 bytes printable → almost certainly text


def _extract_buf_rbp_offset(text: str) -> int:
    """
    Parse GDB disassembly output for [rbp - 0x...] references to find buf's
    offset from RBP. Returns the largest such offset found (the deepest local = buf).
    Falls back to parsing 'sub rsp, 0x...' for the frame size.
    """
    import re as _re
    ansi = _re.compile(r'\x1b\[[0-9;]*m')
    clean = ansi.sub('', text)

    max_offset = 0
    # e.g. [rbp - 0x3f0] or [rbp-0x3f0]
    for m in _re.finditer(r'\[rbp\s*-\s*0x([0-9a-fA-F]+)\]', clean):
        val = int(m.group(1), 16)
        if val > max_offset:
            max_offset = val

    if max_offset == 0:
        # fallback: sub rsp, 0xNNN gives total frame size ≈ buf_rbp_offset
        for m in _re.finditer(r'sub\s+(?:rsp|esp),\s*0x([0-9a-fA-F]+)', clean):
            val = int(m.group(1), 16)
            if val > max_offset:
                max_offset = val

    return max_offset


def _find_pie_stack_offsets(state: dict, binary_path: str) -> list:
    """
    Run GDB at 'break main' and scan the stack for PIE-range addresses.

    At break main (after prologue: push rbp; mov rbp,rsp), RSP == RBP.
    'x/30gx $rsp' therefore shows [RBP+0], [RBP+8], [RBP+16], ...
    Any address in the binary's mapped range that appears at RBP+N
    is at buf_offset = buf_rbp_offset + N from buf start.

    Returns list of dicts: {"buf_offset": int, "addr": int, "rva": int}.
    """
    import re as _re
    import math as _math

    try:
        from Tool.tool import Tool as _Tool
        tool = _Tool(binary_path=binary_path)
    except Exception:
        return []

    gdb_output = tool.Pwndbg(commands=[
        "break main",
        "run",
        "p/x $rsp",
        "disassemble main",
        "x/30gx $rsp",
        "vmmap",
    ], timeout=40)

    if not gdb_output or gdb_output.startswith("Error"):
        return []

    ansi = _re.compile(r'\x1b\[[0-9;]*m')
    clean = ansi.sub('', gdb_output)

    # --- Parse vmmap for binary address range ---
    binary_name = Path(binary_path).name
    binary_ranges: list = []
    # pwndbg vmmap: "    0x5555...  0x5556...  r-xp  ...  binary_name"
    vmmap_re = _re.compile(
        r'(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)\s+\S+\s+\S+\s+\S+\s+\S*'
        + _re.escape(binary_name),
        _re.IGNORECASE,
    )
    for m in vmmap_re.finditer(clean):
        binary_ranges.append((int(m.group(1), 16), int(m.group(2), 16)))

    if not binary_ranges:
        return []

    binary_start = min(r[0] for r in binary_ranges)
    binary_end   = max(r[1] for r in binary_ranges)

    # --- Extract buf_rbp_offset ---
    # First try existing runs (already have disassemble main output)
    buf_rbp_offset = 0
    for run in state.get("runs", []):
        off = _extract_buf_rbp_offset(run.get("stdout", ""))
        if off > buf_rbp_offset:
            buf_rbp_offset = off
    # Also try the fresh GDB output from this session
    off = _extract_buf_rbp_offset(clean)
    if off > buf_rbp_offset:
        buf_rbp_offset = off
    if buf_rbp_offset == 0:
        buf_rbp_offset = 1008  # reasonable fallback for typical 1000-byte buffers

    # --- Parse x/30gx $rsp output ---
    # Format: "0x7fff1234:  0xAAAA  0xBBBB"
    # Each line shows 2 qwords (16 bytes). Track running index.
    results: list = []
    index = 0
    scan_re = _re.compile(r'0x[0-9a-fA-F]+:\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)')
    for m in scan_re.finditer(clean):
        for val_str in [m.group(1), m.group(2)]:
            val = int(val_str, 16)
            if binary_start <= val < binary_end:
                buf_offset = buf_rbp_offset + index * 8
                rva = val - binary_start
                # Skip duplicates (same buf_offset)
                if not any(r["buf_offset"] == buf_offset for r in results):
                    results.append({
                        "buf_offset": buf_offset,
                        "addr": val,
                        "rva": rva,
                        "stack_index": index,
                        "binary_start": binary_start,
                    })
            index += 1

    return results


def _extract_last_hex_dump(combined: str) -> str:
    """
    Extract the last pwnlib hexdump block from exploit output.
    Returns a compact summary of the received bytes for Stage Refine context.
    Pwnlib format:
      [DEBUG] Received 0x41f bytes:
          00000000  41 41 41 ...  │AAAA│...│
          *
          00000410  f8 5f ...     │····│
          0000041f
    """
    import re as _re
    # Match "[DEBUG] Received N bytes:" followed by hex lines
    pattern = _re.compile(
        r"\[DEBUG\] (Received|Sent) (0x[0-9a-fA-F]+|\d+) bytes:\n((?:[ \t]+[^\n]+\n)*)",
        _re.MULTILINE,
    )
    matches = list(pattern.finditer(combined))
    if not matches:
        return ""

    # Get the last 3 Received blocks (most recent interactions)
    received = [m for m in matches if m.group(1) == "Received"]
    if not received:
        return ""

    blocks = received[-3:]  # last 3 received blocks
    result_parts = []
    for m in blocks:
        size_str = m.group(2)
        hex_lines = m.group(3)
        result_parts.append(f"[DEBUG] Received {size_str} bytes:\n{hex_lines.rstrip()}")

    summary = "\n\n".join(result_parts)
    return summary


def _find_hardcoded_pie_base(exploit_path: str) -> str:
    """
    Scan exploit code for hardcoded PIE base addresses when PIE/ASLR is enabled.
    Returns a warning string if found, empty string otherwise.
    """
    import re as _re
    try:
        code = Path(exploit_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    # 0x555555554000: most common ASLR-off debug PIE base
    # Also catch `pie_base = 0x<static>`, `binary_base = 0x<static>`
    checks = [
        (r'0x555555554000', "0x555555554000 is the ASLR-off debug PIE base"),
        (r'(?:pie_base|binary_base|base_addr)\s*=\s*(0x[0-9a-f]{6,12})\b', "hardcoded base"),
    ]
    for pat, label in checks:
        m = _re.search(pat, code, _re.IGNORECASE)
        if not m:
            continue
        # Skip if immediately followed by arithmetic (might be offset computation)
        tail = code[m.end():m.end() + 5].strip()
        if tail and tail[0] in "+-*/":
            continue
        matched_val = m.group(1) if m.lastindex else m.group(0)
        return (
            f"\n\n[!] HARDCODED PIE BASE DETECTED ({label} = {matched_val}). "
            f"With ASLR enabled the base changes every run — it MUST be computed from a "
            f"dynamically leaked return address or binary address, not a fixed constant. "
            f"The stage will likely fail unless PIE/ASLR is verified to be disabled."
        )
    return ""


def _analyze_core_if_exists(binary_path: str, search_dirs: list, since: float) -> str:
    """
    Core dump 파일이 있으면 GDB로 분석하여 crash 정보를 반환.
    없으면 빈 문자열 반환 (caller가 fallback 처리).

    Args:
        binary_path: 바이너리 경로
        search_dirs: core 파일 탐색 디렉토리 목록
        since: 이 시각 이후 생성된 core 파일만 대상 (epoch seconds)
    """
    import glob, time as _t

    core_file = None
    for d in search_dirs:
        candidates = list(Path(d).glob("core*")) + list(Path(d).glob("vgcore*"))
        for c in candidates:
            try:
                if c.stat().st_mtime >= since - 1:  # 1초 여유
                    core_file = c
                    break
            except OSError:
                continue
        if core_file:
            break

    if not core_file:
        return ""

    console.print(f"[green]Core dump found: {core_file} — running GDB analysis[/green]")

    # gdb binary core 형식으로 positional arg 전달 → pwndbg auto-context 정상 동작
    gdb_cmds = [
        "set pagination off",
        "info registers",           # 전체 레지스터 (rip/rsp/rbp/rdi/rsi/rdx 등)
        "x/10i $rip",               # 크래시 지점 인스트럭션
        "x/20gx $rsp",              # 스택 상위 20개
        "bt",                       # 백트레이스
        "info proc mappings",       # 메모리 레이아웃 (ASLR/PIE 확인용)
    ]
    cmd = (
        ["gdb", "-q", "-batch", binary_path, str(core_file)]
        + [arg for c in gdb_cmds for arg in ["-ex", c]]
    )
    analysis_result = ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        result = out if out else err
        analysis_result = f"\n\n=== GDB CRASH ANALYSIS (core: {core_file.name}) ===\n{result}\n=== END GDB ANALYSIS ==="
    except Exception as e:
        analysis_result = f"\n[Core dump found at {core_file} but GDB analysis failed: {e}]"

    # Move core file to a 'cores' subdirectory so it doesn't pollute the binary dir
    # and prevent Plan agent from mistaking it for a binary on the next iteration.
    try:
        cores_dir = core_file.parent / "cores"
        cores_dir.mkdir(exist_ok=True)
        core_file.rename(cores_dir / core_file.name)
        console.print(f"[dim]Core file moved to {cores_dir / core_file.name}[/dim]")
    except Exception:
        pass  # Non-critical — analysis already captured

    return analysis_result


def _classify_failure_type(error_text: str) -> str:
    """Classify the type of stage failure from error output text.

    Categories:
        TIMED_OUT         — exploit hung (bad recvuntil, infinite loop)
        CONNECTION_CLOSED — remote EOF / pipe broken / server closed
        EXCEPTION         — Python exception / import error / file not found
        WRONG_VALUE       — leaked value failed range or sanity check
        WRONG_FORMAT      — marker found but required fields missing
        STAGE_FAILED_CODE — exploit code explicitly printed STAGE_FAILED
        NO_OUTPUT         — expected marker absent, minimal output
        UNKNOWN           — cannot determine
    """
    if not error_text:
        return "NO_OUTPUT"

    err = error_text[:3000]

    if "TIMEOUT" in err or "timed out after" in err:
        return "TIMED_OUT"

    if any(kw in err for kw in (
        "EOFError", "BrokenPipeError", "Connection reset by peer",
        "EOF occurred in violation", "closed the connection",
        "[Errno 104]", "[Errno 111]",
    )):
        return "CONNECTION_CLOSED"

    if "STAGE_FAILED" in err:
        return "STAGE_FAILED_CODE"

    if any(kw in err for kw in (
        "contains no 'pie_leak=0x",
        "no hex values detected in the marker line",
        "was found but the line contains no",
        "marker was found but no hex values",
        "missing required",
    )):
        return "WRONG_FORMAT"

    if any(kw in err for kw in (
        "not page-aligned",
        "non-zero LSB",
        "look like printable ASCII text",
        "looks like printable ASCII text",
        "look like ASCII text",
        "looks like a canonical pointer",
        "is in STACK range",
        "STACK address",
        "is in binary/heap range",
        "binary/heap range",
        "out of expected PIE range",
        "out of valid user-space",
        "is below 0x10000000000",
        "PIE pointer was NOT reached",
        "is zero or near-zero",
        "invalid (too small)",
        "printable bytes",
        "reads prompt text",
        "No exploit file found",
    )):
        return "WRONG_VALUE"

    if "Traceback" in err or err.lstrip().startswith("ERROR:"):
        return "EXCEPTION"

    if "not found" in err.lower() or ("marker" in err.lower() and "not" in err.lower()):
        return "NO_OUTPUT"

    return "UNKNOWN"


def _set_failure_type(stage: dict) -> None:
    """Set failure_type without appending to history.

    Called from Stage_verify_node so the Refiner template can show
    failure_type.  History is maintained separately by Stage_refine_node via
    _record_failure_type() to avoid double-counting.
    """
    if stage.get("verified"):
        return
    stage["failure_type"] = _classify_failure_type(stage.get("error", ""))


def _record_failure_type(stage: dict) -> None:
    """Classify and APPEND failure_type to stage history.

    Called from Stage_refine_node (once per refinement cycle) so that
    route_after_stage_verify can detect repeated same-type failures and
    escalate early.  No-op for verified stages.
    """
    if stage.get("verified"):
        return
    ft = _classify_failure_type(stage.get("error", ""))
    stage.setdefault("failure_type_history", []).append(ft)
    stage["failure_type"] = ft


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
        _set_failure_type(current_stage)
        stages[current_idx] = current_stage
        state["staged_exploit"]["stages"] = stages
        return state

    binary_path = state.get("binary_path", "")
    analysis_binary = _binary_for_dynamic_analysis(state) or binary_path
    _core_search_dirs = [str(Path(exploit_path).parent), "/tmp"]
    if analysis_binary or binary_path:
        _core_search_dirs.insert(0, str(Path(analysis_binary or binary_path).parent))
    _run_start = 0.0

    # Pre-run: detect hardcoded PIE base when PIE is enabled
    analysis = state.get("analysis", {})
    result_str = str(analysis.get("checksec", {}).get("result", "")).lower()
    _has_pie = "pie enabled" in result_str and "no pie" not in result_str
    _pie_warn = ""
    if _has_pie:
        _pie_warn = _find_hardcoded_pie_base(exploit_path)
        if _pie_warn:
            console.print(f"[yellow]⚠ {_pie_warn.strip()}[/yellow]")

    # p.interactive() 자동 제거 — 자동화 환경에서 무한 블로킹을 유발함.
    # 프롬프트에 "NEVER use p.interactive()"가 있음에도 LLM이 삽입하는 경우가 있어
    # 실행 직전에 코드 레벨에서 제거하여 60초 TIMEOUT을 원천 차단.
    try:
        _code_text = Path(exploit_path).read_text(encoding="utf-8", errors="replace")
        if "p.interactive()" in _code_text:
            console.print("[yellow]⚠ p.interactive() detected — auto-removing to prevent subprocess hang[/yellow]")
            _fixed = _code_text.replace(
                "p.interactive()",
                "pass  # p.interactive() removed: non-interactive automation environment",
            )
            Path(exploit_path).write_text(_fixed, encoding="utf-8")
    except Exception:
        pass

    console.print(f"[cyan]Running Stage {current_idx+1}: {current_stage['stage_id']}...[/cyan]")

    # Run the script
    try:
        import resource, time as _time
        env = os.environ.copy()
        # Docker 모드: TARGET_HOST/TARGET_PORT 환경변수 주입
        if state.get("docker_port"):
            env["TARGET_HOST"] = "localhost"
            env["TARGET_PORT"] = str(state.get("docker_port", 1337))

        def _enable_core():
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            except Exception:
                pass

        _run_start = _time.time()
        result = subprocess.run(
            ["python3", exploit_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(exploit_path).parent),
            env=env,
            preexec_fn=_enable_core,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + ("\n--- STDERR ---\n" + stderr if stderr else "")
    except subprocess.TimeoutExpired:
        combined = "TIMEOUT: Exploit timed out after 60 seconds"
        console.print(f"[red]{combined}[/red]")

        # Probe the binary to capture its actual I/O output so the refiner can
        # identify wrong recvuntil() strings (the most common cause of timeout).
        if analysis_binary and Path(analysis_binary).exists():
            try:
                import pty, os as _os, select as _sel, time as _t
                # Run binary briefly, send a newline, capture first ~500 bytes of output
                master_fd, slave_fd = pty.openpty()
                probe_proc = subprocess.Popen(
                    [analysis_binary],
                    stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
                    close_fds=True,
                )
                _os.close(slave_fd)
                _t.sleep(0.3)
                probe_output = b""
                while True:
                    r, _, _ = _sel.select([master_fd], [], [], 0.2)
                    if not r:
                        break
                    chunk = _os.read(master_fd, 256)
                    if not chunk:
                        break
                    probe_output += chunk
                    if len(probe_output) >= 512:
                        break
                probe_proc.kill()
                probe_proc.wait(timeout=2)
                _os.close(master_fd)
                if probe_output:
                    combined += (
                        f"\n\n[BINARY PROBE] Actual binary output (first interaction):\n"
                        f"{repr(probe_output)}\n\n"
                        "HINT: recvuntil() strings in the exploit MUST EXACTLY match the above output.\n"
                        "Common cause of timeout: waiting for a string that does not appear "
                        "(e.g. 'Result: ' when the binary only outputs the buffer directly)."
                    )
                    console.print(f"[yellow]Binary probe output: {repr(probe_output[:200])}[/yellow]")
            except Exception as probe_err:
                console.print(f"[dim]Binary probe failed: {probe_err}[/dim]")

        current_stage["verified"] = False
        current_stage["error"] = combined
        current_stage["last_hex_dump"] = _extract_last_hex_dump(combined)
        # Opportunistic crash analysis (동적 분석용 바이너리로 코어 분석 — 범용)
        _crash_info = _analyze_core_if_exists(analysis_binary, _core_search_dirs, _run_start)
        if _crash_info:
            current_stage["error"] += "\n\n=== CORE DUMP ANALYSIS ===\n" + _crash_info
        _set_failure_type(current_stage)
        stages[current_idx] = current_stage
        state["staged_exploit"]["stages"] = stages
        return state
    except Exception as e:
        combined = f"ERROR: {str(e)}"
        console.print(f"[red]{combined}[/red]")
        current_stage["verified"] = False
        current_stage["error"] = combined
        # Opportunistic crash analysis (동적 분석용 바이너리로 코어 분석 — 범용)
        _crash_info = _analyze_core_if_exists(analysis_binary, _core_search_dirs, _run_start)
        if _crash_info:
            current_stage["error"] += "\n\n=== CORE DUMP ANALYSIS ===\n" + _crash_info
        _set_failure_type(current_stage)
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
            current_stage["error"] = (
                "=== SHELL_FAILED (no shell obtained — check exploit flow) ===\n"
                + combined
            )
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
            current_stage["last_hex_dump"] = _extract_last_hex_dump(combined)
        elif has_marker:
            # Additional heuristic: for leak stages, check if leaked values are valid
            stage_id_lower = current_stage.get("stage_id", "").lower()
            is_leak_stage = "leak" in stage_id_lower or "canary" in stage_id_lower

            if is_leak_stage:
                # Extract the marker line
                marker_line = ""
                for line in combined.split("\n"):
                    if marker in line:
                        marker_line = line
                        break

                # For canary_pie_leak: validate both canary AND pie_leak
                if "canary_pie" in stage_id_lower or "canary" in stage_id_lower or "pie" in stage_id_lower:
                    # --- Canary validation ---
                    canary_match = re.search(r"canary=0x([0-9a-fA-F]+)", marker_line)
                    if canary_match:
                        canary_val = int(canary_match.group(1), 16)
                        if _looks_like_ascii_text(canary_val):
                            console.print(f"[red]Stage {current_idx+1} FAILED (canary=0x{canary_val:x} looks like ASCII text — I/O desync, reading prompt instead of canary)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"canary=0x{canary_val:x} bytes look like printable ASCII text "
                                f"(little-endian: {list(canary_val.to_bytes(8, 'little'))}). "
                                f"This means recvn()/recv() is reading the prompt string, not the canary. "
                                f"Fix: consume all output up to the NEXT prompt with recvuntil(), "
                                f"then slice bytes at the correct offset.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state
                        elif (canary_val & 0xff) != 0:
                            console.print(f"[red]Stage {current_idx+1} FAILED (canary=0x{canary_val:x} LSB is not \\x00 — stack canary always ends with null byte)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"canary=0x{canary_val:x} has non-zero LSB (0x{canary_val & 0xff:02x}). "
                                f"Real stack canary always has \\x00 as its first (lowest) byte. "
                                f"Check that u64() is applied correctly and the right bytes are being extracted.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state
                        elif canary_val >= 0x7FF000000000:
                            # x86-64 stack is in high half (e.g. 0x7ffe..., 0x7ffd...). Reject as false positive.
                            console.print(
                                f"[red]Stage {current_idx+1} FAILED "
                                f"(canary=0x{canary_val:x} is STACK address >= 0x7ff000000000, not canary)[/red]"
                            )
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"canary=0x{canary_val:x} is in STACK range (>= 0x7ff000000000). "
                                f"You are reading the wrong offset (e.g. saved RBP). Use the verified canary offset "
                                f"from analysis (e.g. buf_offset_to_canary), not a larger index.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state
                        elif (canary_val >> 48) == 0 and canary_val > 0x1000:
                            # x86-64 canonical user-space addresses have bits 48-63 = 0x0000
                            # (e.g. 0x00007fff... stack, 0x00007f... libc, 0x00007b... WSL2 libc).
                            # A real stack canary is a random 64-bit value — the probability that
                            # its top 2 bytes are both 0x00 AND the value is > 0x1000 is ~1/65536,
                            # which makes this check a reliable pointer-address detector.
                            console.print(
                                f"[red]Stage {current_idx+1} FAILED "
                                f"(canary=0x{canary_val:x} looks like a canonical pointer — "
                                f"upper 16 bits are 0x0000, typical of stack/libc addresses. "
                                f"Real canaries have random upper bytes.)[/red]"
                            )
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"canary=0x{canary_val:x} has upper 16 bits = 0x0000, matching the pattern "
                                f"of x86-64 canonical user-space addresses (stack: 0x00007fff..., "
                                f"libc: 0x00007f... or lower on WSL2/Docker). "
                                f"This is almost certainly a stack or library address, NOT the canary. "
                                f"The exploit is reading from the wrong index — check the canary offset "
                                f"from the buffer start (e.g. Ghidra: canary at RBP-0x10, buffer at RBP-0x1F "
                                f"→ canary_index = 0x1F - 0x10 = 15, not 23).\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state

                    # --- Raw return address validation (before masking) ---
                    raw_ret_match = re.search(r"raw_ret=0x([0-9a-fA-F]+)", marker_line)
                    if raw_ret_match:
                        raw_ret_val = int(raw_ret_match.group(1), 16)
                        raw_bytes = raw_ret_val.to_bytes(8, "little")
                        printable = sum(1 for b in raw_bytes if 0x20 <= b <= 0x7e)
                        if printable >= 6:
                            console.print(f"[red]Stage {current_idx+1} FAILED (raw_ret=0x{raw_ret_val:x} has {printable}/8 printable bytes — reading prompt text, not return address)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"raw_ret=0x{raw_ret_val:x} bytes: {list(raw_bytes[:6])} — {printable}/8 printable. "
                                f"This is prompt text, not a real return address. "
                                f"printf stopped at a null byte before reaching the return address (likely in saved RBP). "
                                f"The return address was never in the received output.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state

                    # --- libc_base validation (reject stack address or invalid range) ---
                    libc_base_match = re.search(r"libc_base=0x([0-9a-fA-F]+)", marker_line)
                    if libc_base_match and "libc" in stage_id_lower:
                        libc_base_val = int(libc_base_match.group(1), 16)
                        # x86-64: libc typically 0x7f...; stack 0x7ff... (e.g. 0x7ffd...). Page-aligned.
                        if (libc_base_val & 0xFFF) != 0:
                            console.print(
                                f"[red]Stage {current_idx+1} FAILED "
                                f"(libc_base=0x{libc_base_val:x} is not page-aligned — invalid)[/red]"
                            )
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"libc_base=0x{libc_base_val:x} is not page-aligned (expected & 0xfff == 0). "
                                f"Likely wrong leak or wrong offset.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state
                        if libc_base_val >= 0x7FF000000000:
                            console.print(
                                f"[red]Stage {current_idx+1} FAILED "
                                f"(libc_base=0x{libc_base_val:x} is STACK range, not libc — false positive)[/red]"
                            )
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"libc_base=0x{libc_base_val:x} is in STACK range (>= 0x7ff000000000), not a valid libc base. "
                                f"Do not treat this as STAGE1_OK. Check leak offset and that the leaked value is from libc.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state
                        # Lower bound: 0x10000000000 (1TB) — generous enough for WSL2/Docker/VM
                        # environments where libc can be mapped at 0x72d.., 0x73e.., 0x7b.. etc.
                        # Upper bound already handled by stack-range check above.
                        if libc_base_val < 0x10000000000:
                            console.print(
                                f"[red]Stage {current_idx+1} FAILED "
                                f"(libc_base=0x{libc_base_val:x} out of valid user-space shared-lib range "
                                f"[0x10000000000, 0x7ff000000000))[/red]"
                            )
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"libc_base=0x{libc_base_val:x} is below 0x10000000000 — likely a binary, heap, "
                                f"or wrong offset. Valid x86-64 shared-lib range: [0x10000000000, 0x7ff000000000). "
                                f"Check leak calculation.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state

                    # --- PIE base validation ---
                    # pie_base= (preferred) or pie_leak= (raw leaked addr) from marker
                    pie_base_match = re.search(r"pie_base=0x([0-9a-fA-F]+)", marker_line)
                    pie_leak_match = re.search(r"pie_leak=0x([0-9a-fA-F]+)", marker_line)
                    pie_match = pie_base_match or pie_leak_match
                    if pie_match:
                        pie_val = int(pie_match.group(1), 16)
                        # raw leak addr → compute page-aligned base for validation
                        pie_base_val = pie_val & ~0xfff
                        if pie_val < 0x1000:
                            # pwntools debug 출력에서 수신 바이트 크기 파싱
                            recv_sizes = [
                                int(m, 16)
                                for m in re.findall(r"\[DEBUG\] Received (0x[0-9a-fA-F]+) bytes", combined)
                            ]
                            recv_hint = ""
                            if recv_sizes:
                                last_recv = recv_sizes[-1]
                                # recv에는 printf output + '\n'(1) + "Pattern: "(9) 등 overhead 포함
                                # printf가 출력한 실제 바이트 수는 recv - overhead
                                # overhead 최소 추정: \n(1) + next_prompt(~9)
                                est_printed = max(0, last_recv - 10)
                                recv_hint = (
                                    f"\n\nPIE POINTER NOT REACHED — DIAGNOSIS:\n"
                                    f"  Last server response: {last_recv} bytes (0x{last_recv:x}).\n"
                                    f"  Estimated printf output before null-stop: ~{est_printed} bytes.\n"
                                    f"  → printf hit a null byte at ~offset {est_printed} and stopped there.\n"
                                    f"  → The PIE pointer is at a HIGHER offset on the stack.\n"
                                    f"  → You need to write MORE non-null bytes to overwrite null bytes in the canary, saved RBP, and saved RIP.\n"
                                    f"\n"
                                    f"  CRITICAL — HOW MANY BYTES TO SEND:\n"
                                    f"  - If pattern is read with `read(STDIN_FILENO, ...)` or `read(0, ...)` (check source!):\n"
                                    f"    → NO +1 from scanf '\\n'. The '\\n' ungetted by scanf is in stdio's push-back buffer,\n"
                                    f"      which is INVISIBLE to the raw `read()` syscall.\n"
                                    f"    → effective_len = EXACTLY the bytes you send. Send 43 to get effective=43.\n"
                                    f"    → formula: ceil(target_len / pattern_len) * pattern_len = total_written\n"
                                    f"    → choose pattern_len so ceil(target_len / pattern_len) * pattern_len covers the PIE pointer offset\n"
                                    f"  - If pattern is read with `fgets()` or `scanf()` (stdio-based):\n"
                                    f"    → '\\n' from previous scanf IS in stdio buffer → effective = sent + 1\n"
                                    f"    → Send 42 to get effective=43.\n"
                                )
                            last_recv_str = f" | last recv={recv_sizes[-1]}B" if recv_sizes else ""
                            console.print(f"[red]Stage {current_idx+1} FAILED (pie_leak=0x{pie_val:x} — output truncated short{last_recv_str})[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"pie_leak=0x{pie_val:x} is zero or near-zero — the PIE pointer was NOT reached in the output.\n"
                                + recv_hint + "\n"
                            ) + combined
                        elif _looks_like_ascii_text(pie_val):
                            console.print(f"[red]Stage {current_idx+1} FAILED (pie_leak=0x{pie_val:x} looks like ASCII text — I/O desync)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"pie_leak=0x{pie_val:x} bytes look like printable ASCII text — I/O desync.\n"
                            ) + combined
                        elif pie_base_val < 0x100000000000:
                            # pie base가 너무 낮으면 의심 (64bit PIE는 보통 0x55.../0x56...)
                            console.print(f"[red]Stage {current_idx+1} FAILED (pie_base=0x{pie_base_val:x} out of expected PIE range)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"Computed PIE base 0x{pie_base_val:x} is below expected 64-bit PIE range (>= 0x100000000000). "
                                f"Check the leak offset.\n"
                            ) + combined
                        else:
                            console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found, pie_base=0x{pie_base_val:x})[/bold green]")
                            current_stage["verified"] = True
                    else:
                        # canary/pie 스테이지인데 마커에 pie_leak= 필드가 없으면 FAIL
                        # → LLM이 print("STAGE1_OK") 처럼 값 없이 마커만 출력한 경우
                        if "pie" in stage_id_lower or "canary_pie" in stage_id_lower:
                            console.print(f"[red]Stage {current_idx+1} FAILED (marker '{marker}' found but missing 'pie_leak=0x...' field — exploit must include validated PIE base in the marker)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = (
                                f"Stage marker '{marker}' was found but the line contains no 'pie_leak=0x...' value. "
                                f"The exploit code must print the marker WITH the leaked values for validation:\n"
                                f"  print(f\"{marker} canary=0x{{canary:016x}} pie_leak=0x{{pie_base:x}}\")\n"
                                f"Do NOT print the marker unconditionally — only print it after confirming the leak succeeded.\n"
                            ) + combined
                            stages[current_idx] = current_stage
                            state["staged_exploit"]["stages"] = stages
                            return state
                        # 일반 leak 스테이지: 마커 라인의 첫 번째 hex 값으로 폴백
                        addr_match = re.search(r"0x([0-9a-fA-F]+)", marker_line)
                        if addr_match:
                            addr_val = int(addr_match.group(1), 16)
                            if addr_val < 0x1000:
                                console.print(f"[red]Stage {current_idx+1} FAILED (leaked value 0x{addr_val:x} is too small)[/red]")
                                current_stage["verified"] = False
                                current_stage["error"] = f"Leaked value 0x{addr_val:x} is invalid (too small)\n" + combined
                            else:
                                console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found, addr=0x{addr_val:x})[/bold green]")
                                current_stage["verified"] = True
                        else:
                            # hex 값도 없음 → canary가 포함된 스테이지면 FAIL
                            if "canary" in stage_id_lower:
                                console.print(f"[red]Stage {current_idx+1} FAILED (marker '{marker}' found but no hex values — missing canary/pie output)[/red]")
                                current_stage["verified"] = False
                                current_stage["error"] = (
                                    f"Stage marker '{marker}' found but no hex values detected in the marker line.\n"
                                    f"For canary/PIE leak stages, the marker MUST include the leaked values:\n"
                                    f"  print(f\"{marker} canary=0x{{canary:016x}} pie_leak=0x{{pie_base:x}}\")\n"
                                ) + combined
                                stages[current_idx] = current_stage
                                state["staged_exploit"]["stages"] = stages
                                return state
                            else:
                                console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found)[/bold green]")
                                current_stage["verified"] = True
                else:
                    # Generic leak stage: check first hex value
                    addr_match = re.search(r"0x([0-9a-fA-F]+)", marker_line)
                    if addr_match:
                        addr_val = int(addr_match.group(1), 16)
                        is_libc_stage = "libc" in stage_id_lower or "leak" in stage_id_lower

                        if addr_val < 0x1000:
                            console.print(f"[red]Stage {current_idx+1} FAILED (leaked value 0x{addr_val:x} is too small — likely invalid)[/red]")
                            current_stage["verified"] = False
                            current_stage["error"] = f"Leaked value 0x{addr_val:x} is invalid (too small)\n" + combined
                        elif is_libc_stage:
                            # libc/leak stage: 주소 범위 검증
                            # x86-64 user space: 0 ~ 0x7fffffffffff (47-bit)
                            # shared libs (mmap):  0x10000000000 ~ 0x7feff fffffff
                            #   WSL2/Docker/VM 환경에서는 0x70..., 0x7b..., 0x72d... 등 다양
                            # x86-32: libc는 0xf6000000 ~ 0xffffffff
                            # 스택:   0x7ff000000000 이상 (x86-64) — always top of user space
                            # binary/heap: typically < 0x10000000000 (1TB) for x86-64
                            is_64bit_libc = 0x10000000000 <= addr_val < 0x7ff000000000
                            is_32bit_libc = 0xf6000000 <= addr_val <= 0xffffffff
                            is_stack    = addr_val >= 0x7ff000000000
                            is_binary   = addr_val < 0x10000000000  # <1TB: binary/heap 영역

                            if is_64bit_libc or is_32bit_libc:
                                console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found, addr=0x{addr_val:x})[/bold green]")
                                current_stage["verified"] = True
                            elif is_stack:
                                # 스택 주소: 0x7ff000000000 이상
                                # libc stage에서 스택 주소가 나오면 ROP 체인이 실행되지 않은 것
                                console.print(
                                    f"[red]Stage {current_idx+1} FAILED "
                                    f"(0x{addr_val:x} is a STACK address ≥ 0x7ff000000000, not libc — "
                                    f"ROP chain did not execute. Check: "
                                    f"(1) p.clean() before Stage 2, "
                                    f"(2) buf_offset_to_ret, "
                                    f"(3) ROP gadget addresses)[/red]"
                                )
                                current_stage["verified"] = False
                                current_stage["error"] = (
                                    f"Leaked address 0x{addr_val:x} is in STACK range (>= 0x7ff000000000), "
                                    f"not a valid libc address.\n"
                                    f"Root causes to check:\n"
                                    f"  1. Stage 1 output was NOT flushed before Stage 2 — add p.clean(timeout=0.5)\n"
                                    f"  2. buf_offset_to_ret may be wrong — ROP chain overwrote wrong location\n"
                                    f"  3. ROP gadgets (pop_rdi, ret) may have incorrect addresses\n"
                                ) + combined
                            else:
                                # 바이너리/힙 영역 (<1TB) — libc stage에서는 invalid
                                console.print(
                                    f"[red]Stage {current_idx+1} FAILED "
                                    f"(0x{addr_val:x} is in binary/heap range (<0x10000000000), not libc — "
                                    f"valid 64-bit shared-lib range: [0x10000000000, 0x7ff000000000). "
                                    f"Check leak offset and puts() null-termination.)[/red]"
                                )
                                current_stage["verified"] = False
                                current_stage["error"] = (
                                    f"Leaked address 0x{addr_val:x} is in binary/heap range (< 0x10000000000).\n"
                                    f"Valid x86-64 shared-lib range: [0x10000000000, 0x7ff000000000).\n"
                                    f"NOTE: On WSL2/Docker/VM, libc may be at 0x7bc..., 0x73e..., 0x72d... "
                                    f"which all fall within the valid range. If the value is very small "
                                    f"(e.g. 0x4015d0), it is a binary .text address, not libc.\n"
                                    f"Root causes to check:\n"
                                    f"  1. Wrong stack offset — leaking binary .text / saved RBP instead of ret addr\n"
                                    f"  2. puts() null-termination — padding null bytes stop the leak early\n"
                                    f"  3. Wrong GOT entry — leaked non-libc function pointer\n"
                                    f"  4. Incorrect libc base calculation offset\n"
                                ) + combined
                        else:
                            # Non-libc/leak stage: addr >= 0x1000 이면 통과
                            console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found, addr=0x{addr_val:x})[/bold green]")
                            current_stage["verified"] = True
                    else:
                        console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found)[/bold green]")
                        current_stage["verified"] = True
            else:
                console.print(f"[bold green]Stage {current_idx+1} VERIFIED ({marker} found)[/bold green]")
                current_stage["verified"] = True
        else:
            console.print(f"[red]Stage {current_idx+1} FAILED (marker '{marker}' not found)[/red]")
            current_stage["verified"] = False
            current_stage["error"] = combined
            current_stage["last_hex_dump"] = _extract_last_hex_dump(combined)

    stages[current_idx] = current_stage
    state["staged_exploit"]["stages"] = stages

    # Persist verified stage code for reuse (e.g. canary_leak) when the same stage_id appears in a later strategy
    if current_stage.get("verified") and current_stage.get("code") and current_stage.get("stage_id"):
        state.setdefault("verified_stage_code_by_id", {})[current_stage["stage_id"]] = current_stage["code"]

    # Opportunistic crash analysis + pre-run warnings for any non-early-return failure path (동적 분석용 바이너리 사용 — 범용)
    if not current_stage.get("verified"):
        _set_failure_type(current_stage)  # classify before crash suffix
        _crash_info = _analyze_core_if_exists(analysis_binary, _core_search_dirs, _run_start)
        suffix = ""
        if _pie_warn:
            suffix += _pie_warn
        if _crash_info:
            suffix += "\n\n=== CORE DUMP ANALYSIS ===\n" + _crash_info
        if suffix:
            current_stage["error"] = current_stage.get("error", "") + suffix
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

    # Record failure type for escalation tracking (catches inline early-return cases
    # that Stage_verify_node didn't classify, e.g. canary/libc range check failures).
    _record_failure_type(current_stage)

    from Agent.exploit import StageRefinerAgent

    refiner = StageRefinerAgent(model=model, provider=provider)
    error_output = _trim_output_for_context(current_stage.get("error", ""))
    try:
        state = refiner.run(
            state=state,
            stage=current_stage,
            error_output=error_output,
        )
    except LLMRetryExhausted as e:
        console.print(f"[red]Stage refine API exhausted: {e}. Returning state unchanged.[/red]")

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
