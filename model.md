# 感知模型速览（Haar / LBPH / MediaPipe Hands）

本仓库 **Wave 2** 的人脸与手势链路用的是「经典 CV + 轻量关键点 + 规则」组合，而不是端到端大模型多模态。下面按模块简单说明各自是什么、在本项目里怎么用。

---

## Haar Cascade（人脸检测）

**是什么**  
- 一类经典的 **级联分类器**：在灰度图上用不同尺度滑动窗口，用预先训练好的弱分类器链快速筛掉背景，留下疑似人脸区域。  
- OpenCV 自带 `haarcascade_frontalface_default.xml` 等权重，**无需自训**即可做 **近似正面** 人脸框检测。

**特点**  
- **优点**：极轻、CPU 友好、依赖少、延迟低。  
- **局限**：主要适合 **正面或接近正面**；侧脸、强遮挡、极小脸、复杂光照下容易漏检或误检；输出是框，不含身份。

**在本项目**  
- `FaceDetector` 用 Haar 得到候选框，再在框内做灰度 ROI 与 LBPH 识别。见 `src/perception/face.py`。

---

## LBPH（Local Binary Patterns Histograms，人脸识别）

**是什么**  
- OpenCV `cv2.face` 里的 **LBPHFaceRecognizer**：把人脸 ROI 编成局部二值模式直方图，按 **距离（distance）** 做最近邻式匹配；**距离越小表示越像**。  
- 用 `face_registry/<人名>/*.jpg` 离线训练一个小分类器，运行时对主脸 ROI `predict` 得到标签与距离。

**特点**  
- **优点**：模型体积极小、推理快、完全本地、可解释（阈值外显示 `?`）。  
- **局限**：对 **光照、姿态、表情、分辨率** 敏感；注册照片与现场差异大时距离变差；不是深度特征意义上的「Embedding 检索」。

**在本项目**  
- 与 Haar 串联：**Haar 定位 → LBPH 认人**；`match_threshold` 控制「多像才算认识」。见 `src/perception/face.py`。

---

## MediaPipe Hands（手部关键点 / 手势上游）

**是什么**  
- Google **MediaPipe** 的 **手部模型**：从 RGB 图像估计 **21 个手部关键点**（骨骼式），可选 **旧版 `solutions.hands`** 或新版 **Tasks API `HandLandmarker`**（本仓库两种都支持，见 `src/perception/hand.py`）。  
- 仓库在关键点之上用 **几何规则**（手指张开/弯曲、拇指角度等）映射到少量标签：`fist`、`thumbs_up`、`pointing`、`victory`、`heart`（双手）等。

**特点**  
- **优点**：比从零训手势网络轻得多；关键点稳定时规则手势 **可靠、可复现**；适合实时 Demo。  
- **局限**：遮挡、运动模糊、多人多手时可能抖；**语义**仍是「几类固定手势」，不是自然语言或通用手语识别。

**在本项目**  
- `HandGestureDetector`：MediaPipe → 标签 + 置信度。  
- `GestureActionEngine`：在 **MONITOR** 且带云台时，对稳定手势做 **确认帧 + 冷却 + 短编排**（如点头、庆祝、回家），见 `src/gesture_actions.py`、`src/loop.py`。

---

## 和「大模型多模态」对比（一句话）

| 维度 | Haar + LBPH + MediaPipe（本仓库） | 大模型多模态（示例：视频→VLM） |
| --- | --- | --- |
| 延迟 / 算力 | 低，笔记本 CPU 可跑 | 通常高，依赖 GPU 或云端 |
| 可解释性 | 框、距离、关键点、规则清晰 | 端到端黑箱，需额外对齐安全策略 |
| 隐私 | 默认可全本地 | 常涉及 API 与数据出境风险 |
| 能力边界 | 固定几类任务，边界明确 | 更泛化，但难严格保证物理闭环行为 |

---

## 代码入口索引

| 能力 | 文件 |
| --- | --- |
| Haar + LBPH 人脸 | `src/perception/face.py` |
| ArUco + 人脸混合 | `src/perception/combined.py`、`src/perception/__init__.py` |
| MediaPipe 手 + 规则标签 | `src/perception/hand.py` |
| 手势 → 云台编排 | `src/gesture_actions.py` |
| 主循环接入 | `src/loop.py` |

更多 CLI 与系统接口见 [`docs/API.md`](docs/API.md)。
