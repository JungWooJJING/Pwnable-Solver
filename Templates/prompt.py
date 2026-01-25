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
env = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
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

**Parameters:**
- `function_name` (str, optional): Function name to decompile
- `function_address` (str, optional): Function address (hex string like "0x401234")

**Note**: Provide either `function_name` OR `function_address`, not both.

### Example Output
```c
void vuln(void) {
    char buf[64];
    gets(buf);  // VULNERABLE: no bounds checking
    return;
}
```

### Vulnerability Patterns to Identify

| Pattern | Vulnerability | Exploitation |
|---------|--------------|--------------|
| `gets(buf)` | Buffer Overflow | No size limit - overflow to RIP |
| `scanf("%s", buf)` | Buffer Overflow | No size limit |
| `strcpy(dst, src)` | Buffer Overflow | No size check |
| `sprintf(buf, fmt, ...)` | Buffer Overflow | Output may exceed buffer |
| `printf(user_input)` | Format String | Leak/write via %n, %p |
| `read(0, buf, 0x100)` | Buffer Overflow | If 0x100 > sizeof(buf) |
| `free(ptr); *ptr` | Use-After-Free | Dangling pointer access |
| `malloc` without check | Heap Overflow | Check for adjacent allocations |

### When to Use
- After finding interesting function names from main decompilation
- When you need to understand specific vulnerable code
- To calculate buffer sizes and offsets for exploitation

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
        "function": "vuln",
        "location": "line 3",
        "description": "64-byte buffer with gets() - no bounds checking",
        "code_snippet": "char buf[64]; gets(buf);"
      }
    ]
  }
}
```
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
#### {{ func.name }} ({{ func.address }})
```c
{{ func.code }}
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
#### {{ vuln.type }} in `{{ vuln.function }}`
- **Location:** {{ vuln.location }}
- **Description:** {{ vuln.description }}
{% if vuln.code_snippet %}
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
| {{ g.instruction }} | {{ g.address }} |
{% endfor %}
{% else %}
[NOT YET]
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
2. Identify what information is missing ([NOT YET] sections)
3. Create 2-4 prioritized tasks to fill those gaps
4. Update the Analysis Document with any new findings

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Brief explanation of your analysis and decisions",
  "tasks": [
    {
      "id": "task_001",
      "title": "Check binary protections",
      "objective": "Run checksec to identify protections",
      "tool_hint": "Checksec",
      "priority": 0.9
    }
  ],
  "analysis_updates": {
    "checksec": {"done": true, "result": "..."},
    "vulnerabilities": [{"type": "buffer_overflow", ...}]
  },
  "next_focus": "What to prioritize next"
}
```

## Task Priority Guidelines
- **0.9-1.0**: Critical path (checksec, main decompile)
- **0.7-0.8**: Important (vuln function analysis, libc identification)
- **0.5-0.6**: Useful (gadget search, one_gadget)
- **0.3-0.4**: Optional (additional function analysis)

## Analysis Document Sections to Fill
1. **Binary Information**: checksec, decompile, disasm
2. **Vulnerability Analysis**: identified vulns, exploitation strategy
3. **Libc / Libraries**: detection, one_gadgets, offsets
4. **ROP / Gadgets**: key gadgets for exploitation

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
4. Format results for easy LLM comprehension

## Output Format (STRICT JSON)
```json
{
  "reasoning": "How you parsed and interpreted the results",
  "analysis_updates": {
    "checksec": {
      "done": true,
      "result": "NX enabled, No canary, No PIE, Partial RELRO"
    },
    "decompile": {
      "done": true,
      "functions": [
        {
          "name": "main",
          "address": "0x401156",
          "code": "int main() { ... }"
        }
      ]
    },
    "vulnerabilities": [
      {
        "type": "buffer_overflow",
        "function": "vuln",
        "location": "line 15",
        "description": "64-byte buffer with gets() - no bounds checking",
        "code_snippet": "char buf[64]; gets(buf);"
      }
    ]
  },
  "key_findings": [
    "No stack canary - buffer overflow is exploitable",
    "No PIE - addresses are fixed",
    "gets() used without bounds checking"
  ],
  "errors": []
}
```

## Parsing Guidelines
- Extract concrete values (addresses, sizes, offsets)
- Identify vulnerability patterns (gets, strcpy, format strings, etc.)
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
1. Evaluate the current Analysis Document completeness
2. Identify what's still missing for successful exploitation
3. Calculate exploit readiness score
4. Provide specific feedback to Plan Agent

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Assessment of current progress",
  "readiness_score": 0.6,
  "completed_components": [
    "Binary protections identified",
    "Main function decompiled",
    "Buffer overflow vulnerability found"
  ],
  "missing_components": [
    "Need libc leak strategy",
    "Need to find ROP gadgets for ret2libc"
  ],
  "feedback_to_plan": "Focus on finding a libc leak primitive. Check if there's a format string or other info leak. Then search for 'pop rdi' gadget.",
  "recommend_exploit": false,
  "loop_continue": true
}
```

## Readiness Score Guidelines
- **0.0-0.3**: Initial reconnaissance
- **0.4-0.5**: Vulnerability identified
- **0.6-0.7**: Exploitation strategy clear
- **0.8-0.9**: All primitives ready
- **1.0**: Ready for exploit generation

## When to Recommend Exploit
Set `recommend_exploit: true` when:
- Readiness score >= 0.8
- Clear exploitation path exists
- All necessary primitives identified
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
4. Include debugging output

## Output Format (STRICT JSON)
```json
{
  "reasoning": "Exploitation strategy explanation",
  "exploit_type": "ret2libc",
  "exploit_code": "from pwn import *\\n\\n# Setup\\ncontext.binary = './vuln'\\n...",
  "key_steps": [
    "1. Overflow buffer to control return address",
    "2. Leak libc address via puts@plt",
    "3. Return to main for second stage",
    "4. Call system('/bin/sh')"
  ],
  "assumptions": [
    "Libc version: 2.31",
    "No ASLR or handled via leak"
  ]
}
```

## Exploit Code Guidelines
- Use pwntools (`from pwn import *`)
- Include logging (`log.info`, `log.success`)
- Add comments explaining each step
- Handle common issues (alignment, null bytes)

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

## Feedback from Previous Iteration
{% if feedback %}
{{ feedback }}
{% else %}
(First iteration - no feedback yet)
{% endif %}

---

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

Based on the Analysis Document above, identify [NOT YET] sections and create tasks to fill them.
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

    return ANALYSIS_DOCUMENT_TEMPLATE.render(analysis=analysis)


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
        return {
            "analysis_document": analysis_doc,
            "binary_path": state.get("binary_path", ""),
            "protections": state.get("protections", {}),
            "strategy": state.get("analysis", {}).get("strategy", ""),
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
