# Voice intent — Session 30 步骤 3（规则 → JSON → schema）

## 行为说明

- **入口**：`src/voice/intent.py` 中 `parse_text_to_commands(...)`、`parse_text_to_command(...)`。
- **规则优先**：中英关键词 + 方向 + 度数（阿拉伯数字或常见中文「十 / 二十 / 三十五」等）。
- **本地 LLM 兜底（可选）**：传入 `use_llm=True` 与 `llm_client`（如 `OllamaIntentClient`）时，**仅当**规则层 **没有任何非 `noop` 命令** 时才调用 LLM；LLM 返回的 JSON **再次** 经过 `validate_voice_command`，非法或仍为 `noop` 则 **丢弃**，保留规则侧的 `noop`。
- **「置信低」收紧**：若希望 **只在 ASR 不自信时** 才允许走 LLM，设 `llm_only_on_asr_low_confidence=True` 且调用方把 `asr_low_confidence=True`（否则即使规则为 `noop` 也不调 LLM）。默认 `False`：规则 `noop` 即会尝试 LLM（仍满足「规则优先」：规则一旦命中 `home`/`pan_by` 等则 **不调** LLM）。
- **输出**：每条命令都经过 `validate_voice_command`（步骤 1），超限角度会 **钳制** 到 ±60°。
- **多句 / 先再**：按分隔符切句（如 `然后`、`再`、`，`），每句可产出 **多条** 命令（例如同句「向右轉十度向下轉十度」→ `pan_by` + `tilt_by`）。

## 如何验证（自动化）

在仓库根目录：

```powershell
python -m pytest tests/test_voice_intent_parse.py -v
python -m pytest tests/ -m "not slow" -q
```

**预期：** 全部通过；`test_voice_intent_parse.py` 内含 **≥20 条** 表驱动用例（中英混合 + 录音风格长句）。

## 如何快速手测

```powershell
python -c "from src.voice.intent import parse_text_to_commands as p; print(p('回家，然后左转10度'))"
```

**预期：** `parser='rule'`，`commands` 含 `home` 与 `pan_by`。

## 一轮澄清（`VoiceIntentClarifySession`）

- **模块**：`src/voice/clarify.py`。  
- **首轮**（`enable_clarify=True`）：仅 `_intent_from_rules`；若无 actionable → **`phase=clarify`**，返回固定中文 **`clarify_prompt_zh`**，`commands` 为空，**不调 LLM**。  
- **次轮**：对跟进 ASR 文本先规则；若仍 noop 且允许 LLM → **`parse_after_clarification`** 用「原句 + 跟进句」**合并 prompt** 调 **一次** LLM，再 schema。  
- **首轮即命中规则**：与无澄清路径相同，直接 `resolved`。  
- **关闭澄清**：`enable_clarify=False` 时等价于直接调用 `parse_text_to_commands`（首轮即可 LLM 兜底）。

## 本地 Ollama 兜底（手测）

1. 安装并启动 [Ollama](https://ollama.com/)，`ollama pull qwen2.5:1.5b`（或其它小模型）。
2. 在仓库根：

```powershell
python -c "from src.voice.intent import parse_text_to_commands; from src.voice.llm_client import OllamaIntentClient; c=OllamaIntentClient(); print(parse_text_to_commands('往窗户那边稍微动一下云台', use_llm=True, llm_client=c))"
```

**预期：** 若规则为 `noop` 且 Ollama 返回合法 JSON，则 `parser='llm'`；若服务未开或解析失败，仍为规则侧 `noop`。

## 一轮澄清（`VoiceIntentClarifySession`）

- **模块**：`src/voice/clarify.py`。
- **首轮**：只做 **规则**（**不调 LLM**）。若已有可执行命令 → 直接 `resolved`。若全是 `noop` → 返回 **`phase=clarify`**、`clarify_prompt_zh`（默认中文短提示），**`commands` 为空**，内部记下首轮文本，**不写舵机**。
- **次轮**：对补充句再跑 **规则**；若仍无可用命令且 `use_llm=True` 且提供 `llm_client`，则调用 **`parse_after_clarification`**：把 **首轮 + 次轮** 拼成一条 user prompt 给 LLM **一次**，输出仍过 schema。结束后 **清除** pending（仅一轮）。
- **关闭澄清**：`enable_clarify=False` 时行为与原先一致：首轮即可 `parse_text_to_commands(..., use_llm=True)`（无「先问一句」）。
- **手测**：

```powershell
python -c "from src.voice import VoiceIntentClarifySession; s=VoiceIntentClarifySession(); print(s.feed('乱说', enable_clarify=True)); print(s.feed('回家', enable_clarify=True))"
```

**预期：** 第一次 `clarify` + 提示；第二次 `resolved` + `home`。
