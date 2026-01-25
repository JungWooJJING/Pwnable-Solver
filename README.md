### Pwnable-Solver

로컬 **pwnable 바이너리**를 대상으로, LLM(Plan/Instruction/Parsing/Feedback/Exploit) 멀티 에이전트 워크플로우로 분석/익스플로잇을 진행하는 프로젝트입니다.

- **지원 모델**
  - **OpenAI**: `gpt-5.2`
  - **Gemini**: `gemini-3-flash-preview`
- **결과물 저장 위치**: 항상 `./Challenge/<binary_name>/` 아래에 저장됩니다.
  - 예: 디렉토리 리스팅, `Run()` 실행 결과, checksec/ghidra 결과, 생성된 `exploit.py` 등

---

### 요구 사항

- Python 3.10+ (권장: 3.12)
- Linux 환경 (WSL 포함)
- LLM API Key 중 하나
  - `OPENAI_API_KEY` 또는 `GEMINI_API_KEY`

---

### 설치

가상환경을 만들고 의존성을 설치하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate

# 최소 의존성(프로젝트에서 import 되는 것들)
pip install -U pip
pip install rich jinja2 langgraph openai google-generativeai
```

---

### 환경 변수

둘 중 하나만 설정하면 됩니다.

```bash
export OPENAI_API_KEY="..."
# 또는
export GEMINI_API_KEY="..."
```

---

### 실행

#### Interactive mode

```bash
python3 main.py
```

#### Direct mode (바이너리/설명/플래그 포맷 지정)

설명(`-d`)처럼 공백/괄호가 포함된 값은 **반드시 따옴표로 묶어서** 실행하세요.

```bash
python3 main.py \
  -b "/abs/path/to/binary" \
  -t "challenge_title" \
  -d '설명 문자열(괄호 포함 가능) ...' \
  -f 'DH{...}'
```

---

### 핵심 구성

- **`LangGraph/`**
  - `workflow.py`: Plan → Instruction → Parsing → Feedback → (loop/Exploit) 그래프 정의
  - `node.py`: 각 노드를 에이전트로 래핑
  - `state.py`: `SolverState` 및 task/run 유틸
- **`Agent/`**
  - `base.py`: OpenAI/Gemini 클라이언트 + **history 자동 truncation**
  - `plan.py`, `instruction.py`, `parsing.py`, `feedback.py`, `exploit.py`: 각 역할 에이전트
- **`Tool/`**
  - `tool.py`: 분석에 필요한 도구 래퍼
  - 특히 `Run(cmd=...)`로 **임의의 bash 명령을 실행**하고 출력물을 `Challenge/`에 저장
- **`Templates/`**
  - `prompt.py`: 각 에이전트 프롬프트 템플릿(도구 가이던스 포함)

---

### 결과물(Artifacts)

모든 결과물은 자동으로 다음에 저장됩니다:

- `Challenge/<binary_name>/`

예:

- `directory_listing.txt`
- `file_YYYYMMDD_HHMMSS.txt`
- `strings_YYYYMMDD_HHMMSS.txt`
- `readelf_YYYYMMDD_HHMMSS.txt`
- `objdump_YYYYMMDD_HHMMSS.txt`
- `exploit.py`

---

### 트러블슈팅

- **`bash: syntax error near unexpected token '('`**
  - `-d` 인자에 괄호/공백이 포함되어 bash가 토큰으로 해석한 경우입니다.
  - 해결: `-d '... ( ... ) ...'`처럼 **따옴표로 묶기**

- **Jinja2 템플릿 에러**
  - 프롬프트 템플릿에서 dict 키가 없을 때 발생할 수 있습니다.
  - 최근에는 `INSTRUCTION_USER`에서 실행 이력을 `r.commands[0]`로 보여주도록 수정했습니다.

