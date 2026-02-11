### Pwnable-Solver

A project that analyzes and exploits local **pwnable binaries** using an LLM-powered (Plan/Instruction/Parsing/Feedback/Exploit) multi-agent workflow.

- **Supported Models**
  - **OpenAI**: `gpt-5.2`
  - **Gemini**: `gemini-3-flash-preview`
- **Output Location**: All results are saved under `./Challenge/<binary_name>/`
  - e.g., directory listing, `Run()` execution results, checksec/ghidra output, generated `exploit.py`, etc.

---

### Requirements

- Python 3.10+ (recommended: 3.12)
- Linux environment (including WSL)
- One of the following LLM API Keys
  - `OPENAI_API_KEY` or `GEMINI_API_KEY`

---

### Environment Variables

You only need to set one of the following:

```bash
export OPENAI_API_KEY="..."
# or
export GEMINI_API_KEY="..."
```

---

### Usage

#### Interactive mode

```bash
python3 main.py
```

#### Direct mode (specify binary/description/flag format)

For values containing spaces or parentheses (like `-d`), make sure to **wrap them in quotes**.

```bash
python3 main.py \
  -b "/abs/path/to/binary" \
  -t "challenge_title" \
  -d 'Description string (can include parentheses) ...' \
  -f 'Format{...}'
```

---

### Core Components

- **`LangGraph/`**
  - `workflow.py`: Graph definition for Plan → Instruction → Parsing → Feedback → (loop/Exploit)
  - `node.py`: Wraps each node as an agent
  - `state.py`: `SolverState` and task/run utilities
- **`Agent/`**
  - `base.py`: OpenAI/Gemini client + **automatic history truncation**
  - `plan.py`, `instruction.py`, `parsing.py`, `feedback.py`, `exploit.py`: Role-specific agents
- **`Tool/`**
  - `tool.py`: Tool wrappers for analysis
  - Notably, `Run(cmd=...)` **executes arbitrary bash commands** and saves output to `Challenge/`
- **`Templates/`**
  - `prompt.py`: Prompt templates for each agent (including tool guidance)

---

### Artifacts

All outputs are automatically saved to:

- `Challenge/<binary_name>/`

Examples:

- `directory_listing.txt`
- `file_YYYYMMDD_HHMMSS.txt`
- `strings_YYYYMMDD_HHMMSS.txt`
- `readelf_YYYYMMDD_HHMMSS.txt`
- `objdump_YYYYMMDD_HHMMSS.txt`
- `exploit.py`

