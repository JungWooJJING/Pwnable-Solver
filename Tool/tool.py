import subprocess
import os
import importlib
import shutil
import time
import re

from rich.console import Console
from typing import Optional, Any
from pathlib import Path


console = Console()

_GHIDRA_STARTED: bool = False
_GHIDRA_START_ERROR: str = ""


def _ensure_ghidra_started() -> tuple[bool, str]:
    global _GHIDRA_STARTED, _GHIDRA_START_ERROR
    if _GHIDRA_STARTED:
        return True, ""

    ghidra_dir = os.environ.get("GHIDRA_INSTALL_DIR")
    if not ghidra_dir or not os.path.isdir(ghidra_dir):
        return False, "GHIDRA_INSTALL_DIR environment variable is not set or path does not exist"

    try:
        pyghidra = importlib.import_module("pyghidra")
        pyghidra.start(install_dir=ghidra_dir)
        _GHIDRA_STARTED = True
        _GHIDRA_START_ERROR = ""
        return True, ""
    except Exception as e:
        _GHIDRA_STARTED = False
        _GHIDRA_START_ERROR = str(e)
        return False, f"Ghidra start 실패: {_GHIDRA_START_ERROR}"

"""
Checksec, Ghidra_main, Ghidra_decompile_function, pwninit, ROPgadget, Pwndbg, one_gadget
"""
class Tool:
    def __init__(self, binary_path: Optional[str] = None):
        self.binary_path = binary_path
        self._check_binary_exists()
        self._ensure_challenge_dir()

    def _check_binary_exists(self):
        if self.binary_path and not Path(self.binary_path).exists():
            error_message = f"Binary not found: {self.binary_path}"
            console.print(error_message, style="bold red")
            raise FileNotFoundError(error_message)

    def _project_root(self) -> Path:
        # /.../new_solver/Tool/tool.py -> /.../new_solver
        return Path(__file__).resolve().parents[1]

    def _challenge_root(self) -> Path:
        # 결과물은 항상 여기에 저장
        return self._project_root() / "Challenge"

    def _slug(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return "unknown"
        s = s.lower()
        s = re.sub(r"[^a-z0-9._-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unknown"

    def _challenge_dir(self) -> Path:
        name = "unknown"
        if self.binary_path:
            try:
                name = Path(self.binary_path).stem
            except Exception:
                name = "unknown"
        return self._challenge_root() / self._slug(name)

    def _ensure_challenge_dir(self) -> None:
        try:
            self._challenge_dir().mkdir(parents=True, exist_ok=True)
        except Exception:
            # 디렉토리 생성 실패는 치명적이지 않게 처리(명령은 계속 실행 가능)
            pass

    def _write_artifact(self, stem: str, content: str, suffix: str = ".txt") -> Optional[str]:
        """
        Challenge 디렉토리에 결과 저장.
        - 저장 실패 시 None 반환(동작은 계속)
        """
        try:
            self._ensure_challenge_dir()
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe_stem = self._slug(stem) or "output"
            out_path = self._challenge_dir() / f"{safe_stem}_{ts}{suffix}"
            out_path.write_text(content, encoding="utf-8", errors="replace")
            return str(out_path)
        except Exception:
            return None

    def _truncate_for_llm(self, text: str, limit: int = 20000) -> str:
        t = text or ""
        if len(t) <= limit:
            return t
        return t[:limit] + f"\n\n[TRUNCATED] total_chars={len(t)} (full output saved under Challenge/)\n"

    def _run_and_capture(
        self,
        cmd: list[str],
        cwd: Optional[str] = None,
        timeout: int = 60,
    ) -> tuple[str, str, int]:
        try:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            return (result.stdout or ""), (result.stderr or ""), int(result.returncode or 0)
        except FileNotFoundError:
            # cmd[0] 자체가 없을 때
            return "", f"Error: command not found: {cmd[0]}", 127
        except subprocess.TimeoutExpired:
            return "", f"Error: timeout ({timeout}s)", 124
        except Exception as e:
            return "", f"Error: {e}", 1

    def _run_tool_and_save(
        self,
        tool_name: str,
        cmd: list[str],
        out_stem: str,
        timeout: int = 60,
        cwd: Optional[str] = None,
        llm_preview_limit: int = 20000,
    ) -> str:
        """
        - stdout/stderr를 캡처
        - Challenge/ 아래에 전체 결과 저장
        - LLM에는 요약/일부만 반환(너무 큰 출력 방지)
        """
        stdout, stderr, rc = self._run_and_capture(cmd, cwd=cwd, timeout=timeout)
        combined = ""
        if stdout:
            combined += stdout
        if stderr:
            if combined:
                combined += "\n\n--- STDERR ---\n"
            combined += stderr

        saved_path = self._write_artifact(out_stem, combined, suffix=".txt")
        preview = self._truncate_for_llm(combined, limit=llm_preview_limit)

        if rc != 0 and (combined.strip() == "" or combined.startswith("Error:")):
            return combined.strip() or f"Error: {tool_name} failed (rc={rc})"

        # 반환값에는 파일 경로를 함께 제공(파싱/디버깅에 도움)
        if saved_path:
            return f"[artifact] {saved_path}\n\n{preview}".strip()
        return preview.strip()
        
    def Checksec(self) -> str:
        cmd = ["checksec", self.binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        combined = out if out else err
        # 결과는 Challenge/ 아래에 저장(반환값은 기존과 동일하게 유지)
        self._write_artifact("checksec", combined, suffix=".txt")
        return out if out else err

    def Ghidra_main(self, main_only: bool = True) -> str:
        if not self.binary_path:
            return "Error: binary_path is not set"

        ok, err = _ensure_ghidra_started()
        if not ok:
            return f"Error: {err}"

        pyghidra = importlib.import_module("pyghidra")
        try:
            flatapi = importlib.import_module("ghidra.app.decompiler.flatapi")
            FlatDecompilerAPI = getattr(flatapi, "FlatDecompilerAPI")
        except Exception as e:
            return f"Error: Failed to load FlatDecompilerAPI: {e}"

        preferred = ["main", "entry", "entry_point", "_start"]
        result = ""

        with pyghidra.open_program(self.binary_path) as flat:
            program = flat.getCurrentProgram()
            fm = program.getFunctionManager()
            decomp = FlatDecompilerAPI(flat)

            try:
                funcs = list(fm.getFunctions(True))
                entry_funcs = []

                # 이름 우선순위대로 검색
                for name in preferred:
                    for f in funcs:
                        if f.getName() == name:
                            entry_funcs.append(f)

                if not entry_funcs and funcs:
                    entry_funcs = [funcs[0]]

                if not entry_funcs:
                    return "Error: No entry function found"

                funcs_to_process = [entry_funcs[0]] if main_only else entry_funcs

                for func in funcs_to_process:
                    name = func.getName()
                    try:
                        c_code = decomp.decompile(func, 30)
                        entry = func.getEntryPoint()
                        result += f"=== MATCH: {name} {entry} ===\n"
                        result += "--- Decompiled Code ---\n"
                        result += f"{c_code}\n"
                    except Exception as e:
                        result += f"[!] Failed {name}: {e}\n"
            finally:
                decomp.dispose()

        self._write_artifact("ghidra_main", result, suffix=".txt")
        return result

    def Ghidra_decompile_function(
        self,
        function_name: Optional[str] = None,
        function_address: Optional[str] = None,
    ) -> str:
        if not self.binary_path:
            return "Error: binary_path is not set"

        ok, err = _ensure_ghidra_started()
        if not ok:
            return f"Error: {err}"

        pyghidra = importlib.import_module("pyghidra")
        try:
            flatapi = importlib.import_module("ghidra.app.decompiler.flatapi")
            FlatDecompilerAPI = getattr(flatapi, "FlatDecompilerAPI")
        except Exception as e:
            return f"Error: Failed to load FlatDecompilerAPI: {e}"

        def _addr_to_func(fm, addr_str: str):
            s = (addr_str or "").strip().lower().replace("0x", "")
            if not s:
                return None
            try:
                addr_int = int(s, 16)
            except ValueError:
                return None
            addr_space = fm.getProgram().getAddressFactory().getDefaultAddressSpace()
            address = addr_space.getAddress(addr_int)
            return fm.getFunctionAt(address) or fm.getFunctionContaining(address)

        if function_address:
            with pyghidra.open_program(self.binary_path) as flat:
                program = flat.getCurrentProgram()
                fm = program.getFunctionManager()
                decomp = FlatDecompilerAPI(flat)
                try:
                    func = _addr_to_func(fm, function_address)
                    if not func:
                        return "Error: Function not found"
                    c_code = decomp.decompile(func, 30)
                    out = str(c_code)
                    self._write_artifact("ghidra_decompile_function", out, suffix=".txt")
                    return out
                finally:
                    decomp.dispose()

        if function_name:
            with pyghidra.open_program(self.binary_path) as flat:
                program = flat.getCurrentProgram()
                fm = program.getFunctionManager()
                decomp = FlatDecompilerAPI(flat)
                try:
                    func = None
                    if function_name.strip().lower().startswith("0x"):
                        func = _addr_to_func(fm, function_name)
                    if not func:
                        for f in fm.getFunctions(True):
                            if f.getName() == function_name:
                                func = f
                                break
                    if not func:
                        fn = function_name.lower()
                        for f in fm.getFunctions(True):
                            if f.getName().lower() == fn:
                                func = f
                                break
                    if not func:
                        return "Error: Function not found"
                    c_code = decomp.decompile(func, 30)
                    out = str(c_code)
                    self._write_artifact("ghidra_decompile_function", out, suffix=".txt")
                    return out
                finally:
                    decomp.dispose()

        return "Error: function_name or function_address is required"
    
    def Pwninit(self) -> str:
        if not self.binary_path:
            return "Error: binary_path is not set"
        
        workdir = str(Path(self.binary_path).resolve().parent)
        cmd = ["pwninit"]
        try:
            result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        except FileNotFoundError:
            return "Error: pwninit not found. Install via: cargo install pwninit (or download a release binary)"

        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        return out if out else err
    
    def ROPgadget(self, query: Optional[str] = None) -> str:
        if not self.binary_path:
            return "Error: binary_path is not set"
        
        cmd = ["ROPgadget", "--binary", self.binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        text = out if out else err
        self._write_artifact("ropgadget", text, suffix=".txt")

        if query:
            q = query.lower()
            return "\n".join([ln for ln in text.splitlines() if q in ln.lower()])
        return text
    
    def Pwndbg(self, commands: Optional[list[str]] = None, timeout: int = 30) -> str:
        if not self.binary_path:
            return "Error: binary_path is not set"

        cmds = list(commands or [])
        cmd = ["gdb", "-q", "-nx", "-batch", "-ex", "set pagination off", "-ex", "set confirm off"]
        for c in cmds:
            cmd += ["-ex", c]
        cmd += ["--args", self.binary_path]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            return "Error: gdb not found"
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        combined = out if out else err
        self._write_artifact("pwndbg", combined, suffix=".txt")
        return combined

    def One_gadget(self) -> str:
        if not self.binary_path:
            return "Error: binary_path is not set"
        
        cmd = ["one_gadget", self.binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        combined = out if out else err
        self._write_artifact("one_gadget", combined, suffix=".txt")
        return combined

    # =============================================================================
    # 범용 Bash 명령 실행기
    # =============================================================================

    def Run(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
        save: bool = True,
    ) -> str:
        """
        임의의 bash 명령어 실행.

        사용 예:
            tool.Run("strings ./vuln")
            tool.Run("objdump -d -M intel ./vuln | head -100")
            tool.Run("python3 solve.py", cwd="/tmp")
            tool.Run("file ./vuln && readelf -h ./vuln")

        Args:
            cmd: 실행할 bash 명령어 (shell=True로 실행됨)
            cwd: 작업 디렉토리 (기본: binary_path의 부모 디렉토리)
            timeout: 타임아웃(초)
            save: True면 Challenge/ 아래에 결과 저장

        Returns:
            stdout + stderr 결합 결과 (LLM 전달용으로 truncate됨)
        """
        if not cmd or not cmd.strip():
            return "Error: empty command"

        # 작업 디렉토리 결정
        workdir = cwd
        if not workdir and self.binary_path:
            workdir = str(Path(self.binary_path).resolve().parent)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            rc = result.returncode
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

        # stdout + stderr 합치기
        combined = stdout
        if stderr:
            if combined:
                combined += "\n\n--- STDERR ---\n"
            combined += stderr

        # 결과 저장
        if save:
            # 명령어에서 파일명용 stem 추출 (첫 단어)
            stem = cmd.split()[0].split("/")[-1] if cmd.split() else "cmd"
            saved_path = self._write_artifact(stem, combined, suffix=".txt")
            if saved_path:
                preview = self._truncate_for_llm(combined, limit=20000)
                return f"[artifact] {saved_path}\n\n{preview}".strip()

        return self._truncate_for_llm(combined, limit=20000).strip()