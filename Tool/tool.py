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
    
    def Pwninit(self) -> dict:
        """
        Run pwninit and return patched binary info.

        Returns:
            dict: {
                "output": str,  # pwninit 출력
                "patched_binary": str | None,  # 패치된 바이너리 경로
                "success": bool
            }
        """
        if not self.binary_path:
            return {"output": "Error: binary_path is not set", "patched_binary": None, "success": False}

        workdir = Path(self.binary_path).resolve().parent
        binary_stem = Path(self.binary_path).stem
        binary_name = Path(self.binary_path).name

        cmd = ["pwninit"]
        try:
            result = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
        except FileNotFoundError:
            return {
                "output": "Error: pwninit not found. Install via: cargo install pwninit",
                "patched_binary": None,
                "success": False
            }

        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        output = out if out else err

        # 패치된 바이너리 찾기
        patched_binary = None
        possible_names = [
            f"{binary_stem}_patched",
            f"{binary_name}_patched",
            f"./{binary_stem}_patched",
        ]

        for name in possible_names:
            patched_path = workdir / name
            if patched_path.exists():
                patched_binary = str(patched_path)
                break

        # 출력에서 패치 바이너리 경로 파싱
        if not patched_binary and "copying" in output.lower():
            for line in output.split("\n"):
                if "copying" in line.lower() and "_patched" in line:
                    # "copying ./mc_thread to ./mc_thread_patched"
                    parts = line.split()
                    for p in parts:
                        if "_patched" in p:
                            candidate = workdir / p.lstrip("./")
                            if candidate.exists():
                                patched_binary = str(candidate)
                                break

        self._write_artifact("pwninit", output, suffix=".txt")

        return {
            "output": output,
            "patched_binary": patched_binary,
            "success": result.returncode == 0 or patched_binary is not None
        }
    
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
    # Docker 관련 도구 - 익스플로잇 테스트 환경
    # =============================================================================

    def Docker_setup(
        self,
        port: int = 1337,
        ubuntu_version: str = "22.04",
        timeout: int = 300,
    ) -> str:
        """
        익스플로잇 테스트용 도커 환경 자동 세팅.

        pwninit으로 로컬에서 안 될 때, 실제 문제 환경과 동일한 도커에서 테스트.

        - Dockerfile이 있으면 그걸로 빌드
        - 없으면 기본 Ubuntu + 제공된 libc로 환경 구성
        - 자동으로 바이너리 복사하고 socat으로 서비스 실행

        Args:
            port: 포트 번호 (기본: 1337)
            ubuntu_version: Ubuntu 버전 (기본: 22.04)
            timeout: 빌드 타임아웃(초)

        Returns:
            성공/실패 메시지 및 접속 정보
        """
        if not self.binary_path:
            return "Error: binary_path is not set"

        workdir = Path(self.binary_path).resolve().parent
        binary_name = Path(self.binary_path).name
        slug = self._slug(Path(self.binary_path).stem)
        image_name = f"pwn_{slug}"
        container_name = f"pwn_{slug}_container"

        # Dockerfile 존재 확인
        dockerfile_path = workdir / "Dockerfile"

        if dockerfile_path.exists():
            # 기존 Dockerfile 사용
            console.print(f"[cyan]Using existing Dockerfile: {dockerfile_path}[/cyan]")
        else:
            # 자동 Dockerfile 생성
            console.print(f"[cyan]Generating Dockerfile for Ubuntu {ubuntu_version}...[/cyan]")

            # libc 파일 찾기
            libc_files = list(workdir.glob("libc*.so*")) + list(workdir.glob("libc-*.so*"))
            ld_files = list(workdir.glob("ld-*.so*")) + list(workdir.glob("ld-linux*.so*"))

            copy_commands = [f"COPY {binary_name} /challenge/{binary_name}"]
            for f in libc_files + ld_files:
                copy_commands.append(f"COPY {f.name} /challenge/{f.name}")

            dockerfile_content = f"""FROM ubuntu:{ubuntu_version}

RUN apt-get update && apt-get install -y \\
    socat \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /challenge

{chr(10).join(copy_commands)}

RUN chmod +x /challenge/{binary_name}

EXPOSE {port}

CMD ["socat", "TCP-LISTEN:{port},reuseaddr,fork", "EXEC:/challenge/{binary_name}"]
"""
            dockerfile_path.write_text(dockerfile_content)
            console.print(f"[green]Generated Dockerfile at {dockerfile_path}[/green]")

        # 기존 컨테이너 정리
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        # 이미지 빌드
        console.print(f"[cyan]Building Docker image: {image_name}...[/cyan]")
        build_cmd = ["docker", "build", "-t", image_name, str(workdir)]
        stdout, stderr, rc = self._run_and_capture(build_cmd, timeout=timeout)

        if rc != 0:
            error_msg = stderr if stderr else stdout
            return f"Error: Docker build failed\n{error_msg}"

        # 컨테이너 실행
        console.print(f"[cyan]Starting container on port {port}...[/cyan]")
        run_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:{port}",
            image_name
        ]
        stdout, stderr, rc = self._run_and_capture(run_cmd, timeout=30)

        if rc != 0:
            error_msg = stderr if stderr else stdout
            return f"Error: Docker run failed\n{error_msg}"

        container_id = stdout.strip()[:12]

        result = f"""[SUCCESS] Docker environment ready!

Container: {container_name} (ID: {container_id})
Image: {image_name}
Port: {port}

Connect with:
  nc localhost {port}

Or in pwntools:
  p = remote('localhost', {port})

To stop:
  docker stop {container_name}
"""
        self._write_artifact("docker_setup", result, suffix=".txt")
        return result

    def Docker_exploit_test(
        self,
        exploit_path: Optional[str] = None,
        port: int = 1337,
        timeout: int = 60,
    ) -> str:
        """
        도커 환경에서 익스플로잇 테스트 실행.

        Args:
            exploit_path: exploit.py 경로 (기본: Challenge/<binary>/exploit.py)
            port: 타겟 포트
            timeout: 타임아웃(초)
        """
        if not self.binary_path:
            return "Error: binary_path is not set"

        # exploit.py 경로 결정
        if not exploit_path:
            challenge_dir = self._challenge_dir()
            exploit_path = str(challenge_dir / "exploit.py")

        if not Path(exploit_path).exists():
            return f"Error: Exploit not found at {exploit_path}"

        # 익스플로잇 실행 (remote 타겟을 localhost:{port}로)
        console.print(f"[cyan]Running exploit against localhost:{port}...[/cyan]")

        cmd = f"python3 {exploit_path}"
        env = os.environ.copy()
        env["TARGET_HOST"] = "localhost"
        env["TARGET_PORT"] = str(port)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(exploit_path).parent),
                env=env,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
        except subprocess.TimeoutExpired:
            return f"Error: Exploit timed out after {timeout}s"

        combined = stdout
        if stderr:
            combined += f"\n\n--- STDERR ---\n{stderr}"

        self._write_artifact("docker_exploit_test", combined, suffix=".txt")

        # 플래그 감지
        flag_patterns = [r"flag\{[^}]+\}", r"FLAG\{[^}]+\}", r"ctf\{[^}]+\}"]
        for pattern in flag_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return f"[SUCCESS] FLAG FOUND: {match.group()}\n\n{self._truncate_for_llm(combined, 5000)}"

        return self._truncate_for_llm(combined, 10000)

    def Docker_stop(
        self,
        container_name: Optional[str] = None,
        remove: bool = True,
    ) -> str:
        """
        도커 컨테이너 중지 및 삭제.

        Args:
            container_name: 컨테이너 이름 (기본: binary 이름 기반)
            remove: 중지 후 삭제 여부
        """
        if not self.binary_path:
            return "Error: binary_path is not set"

        if not container_name:
            slug = self._slug(Path(self.binary_path).stem)
            container_name = f"pwn_{slug}_container"

        subprocess.run(["docker", "stop", container_name], capture_output=True)

        if remove:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return f"[SUCCESS] Container '{container_name}' stopped and removed"

        return f"[SUCCESS] Container '{container_name}' stopped"

    def Docker_status(self) -> str:
        """현재 실행중인 pwn 관련 컨테이너 상태 확인."""
        cmd = ["docker", "ps", "-a", "--filter", "name=pwn_", "--format",
               "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"]

        stdout, stderr, rc = self._run_and_capture(cmd, timeout=10)
        combined = stdout if stdout else stderr

        if not combined.strip():
            return "No pwn containers found"
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