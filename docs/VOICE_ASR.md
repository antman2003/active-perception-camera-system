# Voice ASR（Session 30 步骤 2）— 模型、缓存位置与验证

## 1. 当前用的是什么 ASR？

| 项目 | 说明 |
|------|------|
| **推理库** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（MIT），底层为 **CTranslate2**，权重对应 **OpenAI Whisper** 结构的公开 checkpoint。 |
| **默认模型规模** | **`base`**：`FasterWhisperAsrProvider(..., model_size="base")`；命令行 `python -m src.voice.asr_demo` 的 **`--model` 默认也是 `base`**。 |
| **其它可选规模** | `tiny`、`base`、`small`、`medium`、`large-v3` 等（与 Hugging Face 上 `Systran/faster-whisper-*` 发布名一致，以 [faster-whisper 文档](https://github.com/SYSTRAN/faster-whisper) 为准）。 |
| **VAD** | `FasterWhisperAsrProvider` 默认 **`vad_filter=True`**（faster-whisper 内置 VAD）；`asr_demo` 可用 **`--no-vad`** 关闭。 |
| **Mock** | `MockAsrProvider`：不加载任何神经网络，返回固定字符串（默认「回家」），用于 CI / 无 GPU / 无麦克风。 |

**总结：** 真 ASR 路径 = **faster-whisper + Whisper `base`（可调）**；不是云端 API。

---

## 2. 装在什么地方？

### 2.1 Python 代码（库本身）

执行 `pip install faster-whisper` 后，包在当前 Python 环境的 **`site-packages`** 里（与普通第三方库相同）。

### 2.2 模型权重（体积大，首次推理才下载）

权重由 **Hugging Face Hub** 拉取并缓存在本机，**不在**你的项目文件夹里。

| 场景 | 典型路径 / 规则 |
|------|------------------|
| **Windows 默认** | `%USERPROFILE%\.cache\huggingface\hub\` 下会出现与 `Systran/faster-whisper-<规模>` 相关的缓存目录（具体子目录名带哈希，以磁盘为准）。 |
| **自定义缓存根** | 设置环境变量 **`HF_HOME`** 后，缓存会落在该目录下（详见 [Hugging Face 缓存文档](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)）。 |

**在本机打印 Hub 缓存根目录：**

```powershell
python -c "import huggingface_hub.constants as c; print(c.HF_HUB_CACHE)"
```

---

## 3. 如何验证（按顺序做，每一步都有「命令 + 预期」）

### 步骤 A — 安装依赖

**命令（在仓库根目录）：**

```powershell
cd "e:\(10) Cursor AI\(8)Demo_active_perception_camera_system"
pip install -r requirements.txt
```

**预期：** 无报错；`pip show faster-whisper` 能显示版本。

---

### 步骤 B — Mock ASR（不下载模型、不占用麦克风）

**命令：**

```powershell
python -m src.voice.asr_demo --backend mock
```

**预期：**

- 进程退出码为 **`0`**。
- 终端出现 **`--- ASR ---`**。
- 行 `text:` 为 **`'回家'`**（若未改 `--mock-text`）。
- **`t_asr_ms: 0`**，`**language: None**`（未强制语言时）。

**自定义 Mock 文案：**

```powershell
python -m src.voice.asr_demo --backend mock --mock-text "hello"
```

**预期：** `text: 'hello'`。

---

### 步骤 C — 单元测试（默认不跑「慢速」整段 ASR）

**命令：**

```powershell
python -m pytest tests/test_voice_asr_provider.py -m "not slow" -v
```

**预期：**

- 退出码 **`0`**。
- 至少包含：`test_mock_transcribe_returns_fixed_text`、`test_mock_accepts_ndarray_without_sample_rate_for_mock`、`test_faster_provider_import_guard` **全部 PASSED**。
- **不会**下载 Whisper 权重（不跑 `@pytest.mark.slow`）。

**跑全部语音相关测试（仍排除 slow）：**

```powershell
python -m pytest tests/ -m "not slow" -q
```

**预期：** 全部通过；其中 `1 deselected` 为慢速用例被跳过，属正常。

---

### 步骤 D — 真 ASR + 本地 WAV（会下载模型，首跑较慢）

**前提：** 准备音频文件。**`.wav`** 由 `soundfile` 读入后走 ndarray 路径；**`.m4a` / `.mp3` 等** 直接把路径交给 faster-whisper，由内置 **ffmpeg** 解码（无需先转 WAV）。

**命令示例：**

```powershell
python -m src.voice.asr_demo --backend faster --input "D:\path\to\clip.wav" --lang zh --model base
```

**预期：**

- **第一次**运行：可能停顿较久（下载 `base` 权重 + 初始化），属正常。
- 结束后打印 **`text:`** 为与音频内容相近的字符串（静音/极短音频可能为空或很短）。
- **`t_asr_ms`** 为正数（毫秒级_wall time_）。
- **`n_segments`** ≥ 0。

**更小模型（下载更小、适合试通）：**

```powershell
python -m src.voice.asr_demo --backend faster --input "D:\path\to\clip.wav" --lang zh --model tiny
```

---

### 步骤 E — 麦克风录音（需要麦克风与 `sounddevice`）

**命令示例：**

```powershell
python -m src.voice.asr_demo --backend faster --record-seconds 3 --lang zh --model base
```

**预期：** 录制 3 秒后出转写；无麦克风或设备被占用时会报错退出（非 0）。

---

### 步骤 F — 可选：慢速 pytest（会下载 **tiny** 权重）

**命令：**

```powershell
python -m pytest tests/test_voice_asr_provider.py -m slow -v
```

**预期：** `test_faster_whisper_tiny_on_silence` **PASSED**；首次会下载 **tiny** 模型到上述 Hugging Face 缓存目录。

---

## 4. 相关代码入口

| 模块 | 说明 |
|------|------|
| `src/voice/asr_provider.py` | `AsrProvider`、`MockAsrProvider`、`FasterWhisperAsrProvider`、`AsrResult` |
| `src/voice/asr_demo.py` | 命令行演示（`python -m src.voice.asr_demo`） |
| `tests/test_voice_asr_provider.py` | 单测；慢测带 `slow` 标记 |
