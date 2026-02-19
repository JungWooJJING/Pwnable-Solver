from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
import time

AgentName = Literal["plan", "instruction", "parsing", "feedback", "exploit"]

TaskStatus = Literal["pending", "in_progress", "done", "blocked", "cancelled"]
TaskOutputStore = Literal["state", "log", "artifacts"]


class TaskProduce(TypedDict, total=False):
    kind: str  # e.g. "directory_listing", "binary_metadata", "decompile_notes"
    store: TaskOutputStore  # "state" | "log" | "artifacts"
    key: str  # state/knowledge/artifacts dict에 저장할 때 키(선택)
    path: str  # artifacts 파일 경로(선택)


class Task(TypedDict, total=False):
    id: str
    title: str          # 사람이 보는 제목: "디렉토리 정보 확인: 내부 파일 확인"
    objective: str      # 왜 하는지/성공 조건
    actions_hint: List[str]  # Instruction이 실제 명령으로 바꾸기 위한 힌트
    depends_on: List[str]    # 선행 task id
    produces: List[TaskProduce]
    priority: float     # 0.0~1.0 (높을수록 먼저)
    status: TaskStatus
    track_id: str       # 멀티 트랙일 때 연결(선택)
    notes: str          # Plan/Feedback 코멘트
    created_at: float
    updated_at: float


class TaskRun(TypedDict, total=False):
    run_id: str
    task_id: str
    commands: List[str]
    success: bool
    stdout: str
    stderr: str
    started_at: float
    finished_at: float
    log_ref: str             
    artifacts: List[str]    
    key_findings: List[str] 


# =========================
# 지식/가설/블로커
# =========================

class Hypothesis(TypedDict, total=False):
    id: str
    claim: str
    confidence: float  # 0.0~1.0
    evidence_refs: List[str]  # log/artifacts 참조


class Blocker(TypedDict, total=False):
    question: str
    severity: float  # 0.0~1.0
    resolved: bool
    resolution: str


# =========================
# Agent 출력(원문/파싱 결과)
# =========================

class AgentOutput(TypedDict, total=False):
    agent: AgentName
    text: str                   # 원문(짧게 유지 권장)
    json: Dict[str, Any]         # 파싱된 결과(있으면)
    created_at: float


# =========================
# Staged Exploit (단계별 익스플로잇)
# =========================

class ExploitStage(TypedDict, total=False):
    stage_id: str               # "leak", "write", "trigger"
    stage_index: int            # 0, 1, 2
    description: str            # "Leak libc via puts@GOT"
    code: str                   # 전체 실행 가능 스크립트
    verified: bool              # 이 단계 검증 완료?
    verification_marker: str    # "STAGE1_OK" or "SHELL_VERIFIED"
    output: str                 # 마지막 실행 stdout
    error: str                  # 실패 시 에러
    refinement_attempts: int    # 수정 시도 횟수


class StagedExploitPlan(TypedDict, total=False):
    stages: List[ExploitStage]
    current_stage_index: int
    all_stages_verified: bool


# =========================
# 전체 State (SSOT)
# =========================

class AnalysisChecksec(TypedDict, total=False):
    done: bool
    result: str


class AnalysisFunction(TypedDict, total=False):
    name: str
    address: str
    code: str


class AnalysisDecompile(TypedDict, total=False):
    done: bool
    functions: List[AnalysisFunction]


class AnalysisDisasm(TypedDict, total=False):
    done: bool
    result: str


class AnalysisIOPattern(TypedDict, total=False):
    prompt: str            # exact string binary outputs before input (e.g. "Pattern: ", "Size: ")
    input_method: str      # "scanf", "read", "gets", "fgets", "recv", etc.
    format: str            # format string if scanf ("%d", "%s", etc.)
    max_len: int           # max input length if known


class AnalysisVuln(TypedDict, total=False):
    type: str                      # buffer_overflow, format_string, use_after_free, etc.
    function: str
    location: str
    description: str
    code_snippet: str
    exploit_constraints: List[str] # binary-specific constraints (e.g. ["target_len <= 1000", "pattern len <= 80"])
    exploit_mechanism: str         # how overflow actually works (e.g. "standard linear overflow" or "pattern repeat overshoot")


class AnalysisLibc(TypedDict, total=False):
    detected: bool
    path: str
    version: str
    one_gadgets: str
    offsets: Dict[str, str]        # symbol -> offset


class AnalysisGadget(TypedDict, total=False):
    instruction: str
    address: str


class AnalysisDynamicVerification(TypedDict, total=False):
    """
    Runtime values extracted by the LLM from Pwndbg/GDB output.
    Populated by the Parsing Agent after a Pwndbg task is run.
    """
    verified: bool
    # Stack layout (all values are LLM-extracted from GDB output)
    buf_offset_to_canary: int        # bytes from buf[0] to canary (e.g. 984)
    buf_offset_to_saved_rbp: int     # bytes from buf[0] to saved rbp  (e.g. 1000)
    buf_offset_to_ret: int           # bytes from buf[0] to return addr (e.g. 1008)
    canary_offset_from_rbp: int      # negative offset, e.g. -16
    # Local variables (LLM fills from 'info locals' / 'info frame' output)
    local_vars: List[Dict[str, Any]] # [{name, address, size, offset_from_rbp}, ...]
    # Memory layout
    binary_base: str                 # hex, from vmmap (e.g. "0x555555554000")
    stack_base: str                  # hex, approximate rsp at time of capture
    rbp_value: str                   # hex (e.g. "0x7fffffffe0f0")
    # Raw output for reference
    raw_gdb_output: str


class Analysis(TypedDict, total=False):
    """Progressive Analysis Document - filled incrementally"""
    checksec: AnalysisChecksec
    decompile: AnalysisDecompile
    disasm: AnalysisDisasm
    vulnerabilities: List[AnalysisVuln]
    strategy: str                  # Exploitation strategy description
    libc: AnalysisLibc
    gadgets: List[AnalysisGadget]
    io_patterns: List[AnalysisIOPattern]  # Binary I/O prompts/inputs extracted from decompiled code
    win_function: bool             # win/flag/backdoor function exists in binary
    win_function_name: str         # name of the win function (e.g. "win", "get_flag")
    win_function_addr: str         # address of the win function (e.g. "0x401156")
    heap_pointer_type: str         # "single", "array", "linked_list", "unknown"
    dynamic_verification: AnalysisDynamicVerification  # LLM-extracted runtime values from Pwndbg
    leak_primitive: bool
    control_hijack: bool
    payload_ready: bool
    readiness_score: float         # 0.0 ~ 1.0
    last_exploit_failure: Dict[str, Any]  # Set by crash analysis node


class SolverState(TypedDict, total=False):
    challenge: Dict[str, Any]      # 문제 메타(카테고리, 플래그 포맷 등)
    cwd: str
    constraints: List[str]
    env: Dict[str, Any]

    # --- 타겟/분석 ---
    binary_path: str
    directory_listing: str         # ls output of working directory
    target_info: Dict[str, Any]    # 추가 타겟 정보
    protections: Dict[str, Any]    # checksec 등
    mitigations: List[str]

    # --- Analysis Document (점진적으로 채워지는 분석 문서) ---
    analysis: Analysis

    # --- 확정/요약/추측 (legacy, 호환용) ---
    facts: Dict[str, Any]          # "검증된 사실"만
    knowledge: Dict[str, Any]      # 정찰/구조 파악 "요약"(과도한 원문 금지)
    hypotheses: List[Hypothesis]   # 가설(증거 부족)
    blockers: List[Blocker]        # 모르는 것(다음 계획을 만드는 트리거)

    # --- 작업 큐/실행 이력 ---
    loop: bool
    tasks: List[Task]              # Plan이 쌓는 큐
    runs: List[TaskRun]            # Instruction 실행 이력
    seen_cmd_hashes: List[str]     # 중복 실행 방지(선택)

    # --- Agent 단계별 결과 ---
    plan_output: AgentOutput
    instruction_output: AgentOutput
    parsing_output: AgentOutput
    feedback_output: AgentOutput
    exploit_output: AgentOutput

    # --- 루프 제어 ---
    iteration_count: int
    workflow_step_count: int

    # --- 성공 판정 ---
    exploit_readiness: Dict[str, Any]   # {score:0~1, components:{...}, recommend_exploit: bool}
    flag_detected: bool
    detected_flag: str
    all_detected_flags: List[str]

    # --- Exploit 검증 ---
    exploit_attempts: int               # 현재 시도 횟수
    max_exploit_attempts: int           # 최대 시도 횟수 (기본: 3)
    exploit_verified: bool              # 익스플로잇 성공 여부
    exploit_error: str                  # 마지막 에러 출력
    exploit_path: str                   # 저장된 exploit.py 경로

    # --- Staged Exploit (단계별 익스플로잇) ---
    staged_exploit: StagedExploitPlan

    # --- 실패 컨텍스트 (Plan 재진입 시 사용) ---
    analysis_failure_reason: str        # "stage_failed", "exploit_failed" 등
    exploit_failure_context: Dict[str, Any]  # {stage_id, code, error, attempts}

    # --- Docker ---
    docker_port: int                    # --docker 모드일 때 포트 (0이면 비활성)


# =========================
# State 초기화
# =========================

def init_analysis() -> Analysis:
    """Initialize empty Analysis Document"""
    return {
        "checksec": {"done": False, "result": ""},
        "decompile": {"done": False, "functions": []},
        "disasm": {"done": False, "result": ""},
        "vulnerabilities": [],
        "strategy": "",
        "libc": {
            "detected": False,
            "path": "",
            "version": "",
            "one_gadgets": "",
            "offsets": {}
        },
        "gadgets": [],
        "io_patterns": [],
        "win_function": False,
        "win_function_name": "",
        "win_function_addr": "",
        "heap_pointer_type": "unknown",
        "dynamic_verification": {"verified": False, "local_vars": [], "raw_gdb_output": ""},
        "leak_primitive": False,
        "control_hijack": False,
        "payload_ready": False,
        "readiness_score": 0.0,
        "last_exploit_failure": {},
    }


def init_state(**overrides: Any) -> SolverState:
    """초기 State 생성(필드가 비어 있어도 안전하도록 기본값 제공)."""
    now = time.time()
    state: SolverState = {
        "challenge": {},
        "cwd": "",
        "constraints": [],
        "env": {},

        "binary_path": "",
        "directory_listing": "",
        "target_info": {},
        "protections": {},
        "mitigations": [],

        # Analysis Document (progressive)
        "analysis": init_analysis(),

        # Legacy (for compatibility)
        "facts": {},
        "knowledge": {},
        "hypotheses": [],
        "blockers": [],

        "loop": False,
        "tasks": [],
        "runs": [],
        "seen_cmd_hashes": [],

        "plan_output": {"agent": "plan", "text": "", "json": {}, "created_at": now},
        "instruction_output": {"agent": "instruction", "text": "", "json": {}, "created_at": now},
        "parsing_output": {"agent": "parsing", "text": "", "json": {}, "created_at": now},
        "feedback_output": {"agent": "feedback", "text": "", "json": {}, "created_at": now},
        "exploit_output": {"agent": "exploit", "text": "", "json": {}, "created_at": now},

        "iteration_count": 0,
        "workflow_step_count": 0,

        "exploit_readiness": {"score": 0.0, "components": {}, "recommend_exploit": False},
        "flag_detected": False,
        "detected_flag": "",
        "all_detected_flags": [],

        # Exploit 검증
        "exploit_attempts": 0,
        "max_exploit_attempts": 3,
        "exploit_verified": False,
        "exploit_error": "",
        "exploit_path": "",

        # Staged Exploit
        "staged_exploit": {
            "stages": [],
            "current_stage_index": 0,
            "all_stages_verified": False,
        },

        # 실패 컨텍스트
        "analysis_failure_reason": "",
        "exploit_failure_context": {},
    }
    state.update(overrides)
    return state


# =========================
# Agent별 "view" (필요한 정보만 전달)
# =========================

def get_state_for_plan(state: SolverState) -> Dict[str, Any]:
    """Plan Agent가 읽을 최소 상태: facts/knowledge/blockers/tasks 요약 중심."""
    return {
        "challenge": state.get("challenge", {}),
        "cwd": state.get("cwd", ""),
        "constraints": state.get("constraints", []),
        "binary_path": state.get("binary_path", ""),
        "target_info": state.get("target_info", {}),
        "protections": state.get("protections", {}),
        "mitigations": state.get("mitigations", []),
        "facts": state.get("facts", {}),
        "knowledge": state.get("knowledge", {}),
        "hypotheses": state.get("hypotheses", []),
        "blockers": state.get("blockers", []),
        "tasks": _summarize_tasks(state.get("tasks", []), limit=20),
        "runs": state.get("runs", [])[-5:],  # 최근 5개만
        "exploit_readiness": state.get("exploit_readiness", {}),
        "analysis_failure_reason": state.get("analysis_failure_reason", ""),
        "exploit_failure_context": state.get("exploit_failure_context", {}),
    }


def get_state_for_instruction(state: SolverState) -> Dict[str, Any]:
    """
    Instruction Agent가 읽을 최소 상태.
    - Plan이 만든 tasks 전체(또는 요약) + 중복 방지 정보 + 타겟/제약
    """
    return {
        "constraints": state.get("constraints", []),
        "cwd": state.get("cwd", ""),
        "binary_path": state.get("binary_path", ""),
        "target_info": state.get("target_info", {}),
        "protections": state.get("protections", {}),
        "mitigations": state.get("mitigations", []),
        "facts": state.get("facts", {}),
        "knowledge": state.get("knowledge", {}),
        "blockers": state.get("blockers", []),
        "tasks": state.get("tasks", []),
        "seen_cmd_hashes": state.get("seen_cmd_hashes", []),
        "runs": state.get("runs", []),
    }


def get_state_for_parsing(state: SolverState) -> Dict[str, Any]:
    """Parsing Agent: 방금 실행된 출력 + 최소 타겟 정보 + 기존 facts."""
    return {
        "execution": state.get("instruction_output", {}),
        "binary_path": state.get("binary_path", ""),
        "protections": state.get("protections", {}),
        "facts": state.get("facts", {}),
        "knowledge": state.get("knowledge", {}),
    }


def get_state_for_feedback(state: SolverState) -> Dict[str, Any]:
    """Feedback Agent: 계획/실행/파싱/가설/블로커를 종합해 readiness 업데이트."""
    return {
        "plan_output": state.get("plan_output", {}),
        "instruction_output": state.get("instruction_output", {}),
        "parsing_output": state.get("parsing_output", {}),
        "facts": state.get("facts", {}),
        "knowledge": state.get("knowledge", {}),
        "hypotheses": state.get("hypotheses", []),
        "blockers": state.get("blockers", []),
        "tasks": _summarize_tasks(state.get("tasks", []), limit=50),
        "runs": state.get("runs", [])[-10:],
        "exploit_readiness": state.get("exploit_readiness", {}),
    }


def get_state_for_exploit(state: SolverState) -> Dict[str, Any]:
    """Exploit Agent: exploit 실행에 필요한 것만 전달(과도한 노이즈 제거)."""
    return {
        "binary_path": state.get("binary_path", ""),
        "protections": state.get("protections", {}),
        "mitigations": state.get("mitigations", []),
        "facts": state.get("facts", {}),
        "knowledge": state.get("knowledge", {}),
        "exploit_readiness": state.get("exploit_readiness", {}),
        "constraints": state.get("constraints", []),
    }


# =========================
# Task 선택/머지 유틸
# =========================

def _task_status(task: Dict[str, Any]) -> str:
    return str(task.get("status", "pending"))


def _task_priority(task: Dict[str, Any]) -> float:
    try:
        return float(task.get("priority", 0.5))
    except Exception:
        return 0.5


def _summarize_tasks(tasks: Any, limit: int = 20) -> List[Dict[str, Any]]:
    if not isinstance(tasks, list):
        return []
    out: List[Dict[str, Any]] = []
    for t in tasks[: max(0, int(limit))]:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "id": t.get("id"),
                "title": t.get("title"),
                "objective": t.get("objective"),
                "status": t.get("status", "pending"),
                "priority": t.get("priority", 0.5),
                "depends_on": t.get("depends_on", []),
                "track_id": t.get("track_id", ""),
            }
        )
    return out


def select_next_tasks(state: SolverState, n: int = 3, track_id: str = "") -> List[Task]:
    """
    Instruction Agent가 실행할 후보 task를 고르는 간단한 선택기.
    - pending만 대상
    - depends_on이 done/cancelled인 것만 선택
    - priority 내림차순
    """
    tasks = state.get("tasks", []) or []
    if not isinstance(tasks, list):
        return []

    completed_ids: set = set()
    for t in tasks:
        if isinstance(t, dict) and _task_status(t) in ("done", "cancelled"):
            tid = t.get("id")
            if isinstance(tid, str) and tid:
                completed_ids.add(tid)

    candidates: List[Task] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        if track_id and t.get("track_id") not in ("", track_id):
            continue
        if _task_status(t) != "pending":
            continue
        deps = t.get("depends_on", []) or []
        if isinstance(deps, list) and all((d in completed_ids) for d in deps if isinstance(d, str)):
            candidates.append(t)  # type: ignore[typeddict-item]

    candidates.sort(key=_task_priority, reverse=True)
    return candidates[: max(0, int(n))]


def upsert_tasks(state: SolverState, new_tasks: List[Task]) -> None:
    """
    Plan Agent가 만든 task들을 state.tasks에 머지.
    - 같은 id가 있으면 업데이트(제목/목표/힌트/우선순위 등 갱신)
    - status는 기본적으로 기존 값을 유지(Plan이 임의로 done으로 덮어쓰지 않게)
    """
    tasks = state.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
        state["tasks"] = tasks

    index: Dict[str, int] = {}
    for i, t in enumerate(tasks):
        if isinstance(t, dict) and isinstance(t.get("id"), str) and t["id"]:
            index[t["id"]] = i

    now = time.time()
    for nt in new_tasks:
        tid = nt.get("id", "")
        if not tid:
            continue
        if tid in index:
            cur = tasks[index[tid]]
            if not isinstance(cur, dict):
                continue
            # status는 유지(단, 새 task가 blocked/cancelled면 반영 가능)
            cur_status = cur.get("status", "pending")
            nt_status = nt.get("status", "pending")
            merged: Dict[str, Any] = dict(cur)
            merged.update(nt)
            if cur_status in ("done", "cancelled"):
                merged["status"] = cur_status
            else:
                # 진행중인 작업을 plan이 되돌리지 않게 보호
                if cur_status == "in_progress" and nt_status == "pending":
                    merged["status"] = "in_progress"
                else:
                    merged["status"] = cur_status if cur_status else nt_status
            merged["updated_at"] = now
            tasks[index[tid]] = merged
        else:
            created = dict(nt)
            created.setdefault("status", "pending")
            created.setdefault("priority", 0.5)
            created.setdefault("depends_on", [])
            created.setdefault("actions_hint", [])
            created.setdefault("produces", [])
            created.setdefault("track_id", "")
            created.setdefault("notes", "")
            created.setdefault("created_at", now)
            created.setdefault("updated_at", now)
            tasks.append(created)  # type: ignore[arg-type]


def add_run(state: SolverState, run: TaskRun) -> None:
    runs = state.get("runs")
    if not isinstance(runs, list):
        runs = []
        state["runs"] = runs
    runs.append(run)


def mark_task_status(state: SolverState, task_id: str, status: TaskStatus) -> bool:
    tasks = state.get("tasks", [])
    if not isinstance(tasks, list):
        return False
    for i, t in enumerate(tasks):
        if isinstance(t, dict) and t.get("id") == task_id:
            t = dict(t)
            t["status"] = status
            t["updated_at"] = time.time()
            tasks[i] = t  # type: ignore[list-item]
            return True
    return False


# =========================
# Analysis Document 업데이트
# =========================

def merge_analysis_updates(state: SolverState, updates: Dict[str, Any]) -> None:
    """
    Analysis Document에 새로운 정보 머지.
    Plan/Parsing Agent에서 공통으로 사용.
    """
    analysis = state.get("analysis", init_analysis())

    # Checksec
    if "checksec" in updates:
        analysis["checksec"].update(updates["checksec"])

    # Decompile
    if "decompile" in updates:
        if updates["decompile"].get("done"):
            analysis["decompile"]["done"] = True

        new_funcs = updates["decompile"].get("functions", [])
        existing_names = {f["name"] for f in analysis["decompile"].get("functions", [])}

        for func in new_funcs:
            if func.get("name") not in existing_names:
                analysis["decompile"]["functions"].append(func)

    # Disasm
    if "disasm" in updates:
        analysis["disasm"].update(updates["disasm"])

    # Vulnerabilities (append, don't overwrite)
    if "vulnerabilities" in updates:
        existing_vulns = {(v.get("type"), v.get("function")) for v in analysis.get("vulnerabilities", [])}

        for vuln in updates["vulnerabilities"]:
            key = (vuln.get("type"), vuln.get("function"))
            if key not in existing_vulns:
                analysis["vulnerabilities"].append(vuln)

    # Strategy
    if "strategy" in updates and updates["strategy"]:
        analysis["strategy"] = updates["strategy"]

    # Libc
    if "libc" in updates:
        if isinstance(updates["libc"], dict):
            analysis["libc"].update(updates["libc"])
        else:
            analysis["libc"]["version"] = str(updates["libc"])

    # Gadgets (append)
    if "gadgets" in updates:
        existing_addrs = {g.get("address") for g in analysis.get("gadgets", [])}

        for gadget in updates["gadgets"]:
            if gadget.get("address") not in existing_addrs:
                analysis["gadgets"].append(gadget)

    # Heap pointer type
    if "heap_pointer_type" in updates:
        analysis["heap_pointer_type"] = str(updates["heap_pointer_type"])

    # I/O patterns (append new ones, deduplicate by prompt string)
    if "io_patterns" in updates:
        existing_prompts = {p.get("prompt", "") for p in analysis.get("io_patterns", [])}
        for pattern in updates["io_patterns"]:
            if pattern.get("prompt", "") not in existing_prompts:
                analysis.setdefault("io_patterns", []).append(pattern)
                existing_prompts.add(pattern.get("prompt", ""))

    # Boolean flags
    for flag in ["win_function", "leak_primitive", "control_hijack", "payload_ready"]:
        if flag in updates:
            analysis[flag] = updates[flag]

    # Win function metadata
    for field in ["win_function_name", "win_function_addr"]:
        if field in updates:
            analysis[field] = str(updates[field])

    # Dynamic verification (LLM-extracted from Pwndbg output — deep merge)
    if "dynamic_verification" in updates:
        dv = analysis.setdefault("dynamic_verification", {"verified": False, "local_vars": [], "raw_gdb_output": ""})
        dv.update(updates["dynamic_verification"])

    # Last exploit failure (overwrite)
    if "last_exploit_failure" in updates:
        analysis["last_exploit_failure"] = updates["last_exploit_failure"]

    state["analysis"] = analysis


# =========================
# Code-based Readiness Score
# =========================

def compute_readiness(analysis: Analysis, runs: Optional[List] = None) -> Tuple[float, List[str], List[str]]:
    """
    Compute exploit readiness score deterministically from Analysis Document.

    Returns:
        (score, completed_components, missing_components)

    Scoring:
        Tier 1 (+0.25): checksec done, vuln found, vuln type known
        Tier 2 (+0.25): offset/detail known, control primitive identified
        Tier 3 (+0.25): gadgets (if NX), leak method (if needed), libc (if needed)
        Tier 4 (+0.15): dynamic verification (Pwndbg executed)
        Tier 5 (+0.1): strategy clear
    """
    completed: List[str] = []
    missing: List[str] = []
    score = 0.0

    checksec = analysis.get("checksec", {})
    vulns = analysis.get("vulnerabilities", [])
    gadgets = analysis.get("gadgets", [])
    libc = analysis.get("libc", {})
    strategy = analysis.get("strategy", "")

    nx = checksec.get("nx", False) if isinstance(checksec.get("result"), str) else False
    # Try to parse nx from result string if not set as bool
    result_str = str(checksec.get("result", "")).lower()
    if "nx enabled" in result_str or "nx:true" in result_str.replace(" ", ""):
        nx = True

    has_canary = False
    if "canary found" in result_str and "no canary" not in result_str:
        has_canary = True

    pie = checksec.get("pie", False)
    if "pie enabled" in result_str and "no pie" not in result_str:
        pie = True

    # --- Tier 1: Basic Requirements (+0.25) ---
    tier1_items = 3
    tier1_done = 0

    if checksec.get("done"):
        completed.append("protections_analyzed")
        tier1_done += 1
    else:
        missing.append("protections_analyzed")

    if vulns:
        completed.append("vulnerability_found")
        tier1_done += 1
    else:
        missing.append("vulnerability_found")

    if any(v.get("type") for v in vulns):
        completed.append("vulnerability_type_known")
        tier1_done += 1
    else:
        missing.append("vulnerability_type_known")

    score += 0.25 * (tier1_done / tier1_items)

    # --- Tier 2: Exploitation Details (+0.25) ---
    tier2_items = 2
    tier2_done = 0

    # Check for offset/buffer_size/fmt_offset
    has_detail = False
    for v in vulns:
        if v.get("estimated_offset") or v.get("buffer_size") or v.get("offset"):
            has_detail = True
            break
        if v.get("fmt_offset"):
            has_detail = True
            break

    if has_detail:
        completed.append("offset_or_detail_known")
        tier2_done += 1
    else:
        missing.append("offset_or_detail_known")

    # Control primitive
    has_primitive = analysis.get("control_hijack", False)
    if not has_primitive:
        for v in vulns:
            if v.get("exploit_primitive") or v.get("type") in ("buffer_overflow", "format_string", "use_after_free"):
                has_primitive = True
                break

    if has_primitive:
        completed.append("control_primitive_identified")
        tier2_done += 1
    else:
        missing.append("control_primitive_identified")

    score += 0.25 * (tier2_done / tier2_items)

    # --- Tier 3: ROP/Libc Requirements (+0.25) ---
    tier3_items = 0
    tier3_done = 0

    has_win = analysis.get("win_function", False)

    if has_win and not pie:
        # Simple ret2win (no PIE): fixed address → no leak, no gadgets, no libc needed
        score += 0.25
        completed.append("win_function_no_pie")
    elif has_win and pie:
        # ret2win with PIE: need binary base leak, but NOT libc or gadgets
        tier3_items = 1
        if analysis.get("leak_primitive"):
            completed.append("pie_base_leak_found")
            tier3_done += 1
        else:
            missing.append("pie_base_leak_for_win")
        score += 0.25 * (tier3_done / tier3_items)
    else:
        # Standard path: no win function
        # Gadgets needed if NX
        if nx:
            tier3_items += 1
            if gadgets:
                completed.append("rop_gadgets_found")
                tier3_done += 1
            else:
                missing.append("rop_gadgets_found")

        # Leak + libc needed if NX
        if nx:
            tier3_items += 1
            if analysis.get("leak_primitive"):
                completed.append("leak_method_found")
                tier3_done += 1
            else:
                missing.append("leak_method_found")

            tier3_items += 1
            if libc.get("detected"):
                completed.append("libc_available")
                tier3_done += 1
            else:
                missing.append("libc_available")

        if tier3_items > 0:
            score += 0.25 * (tier3_done / tier3_items)
        elif checksec.get("done"):
            # Checksec done and NX disabled → shellcode, no ROP needed
            score += 0.25
            completed.append("no_rop_needed")
        else:
            # Checksec not done yet → can't determine tier 3
            missing.append("tier3_unknown_until_checksec")

    # --- Tier 4: Dynamic Verification (+0.15) ---
    # Primary check: LLM has extracted and stored verified runtime values
    dv = analysis.get("dynamic_verification", {})
    has_dynamic = bool(dv.get("verified"))
    # Fallback: any Pwndbg/GDB run recorded (partial credit)
    if not has_dynamic and runs:
        for run in runs:
            cmds = run.get("commands", [])
            cmd_str = " ".join(str(c) for c in cmds).lower()
            if "pwndbg" in cmd_str or "gdb" in cmd_str:
                has_dynamic = True
                break

    if has_dynamic:
        completed.append("dynamic_verification_done")
        score += 0.15
    else:
        missing.append("dynamic_verification_done")

    # --- Tier 5: Strategy (+0.1) ---
    if strategy:
        completed.append("exploit_strategy_clear")
        score += 0.1
    else:
        missing.append("exploit_strategy_clear")

    return round(min(score, 1.0), 2), completed, missing

