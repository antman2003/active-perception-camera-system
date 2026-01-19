## Active Perception Camera System — Implementation Plan

这是一份按 **3 周 / 每天 1 小时（周末 2 小时）**节奏推进的实现计划，目标是做出一个可演示、可复盘、面试可讲的 **主动感知（Active Perception）**相机小系统：在不确定性高时，自动调整动作（曝光/软件增益/ROI zoom）来恢复感知质量。

---

## 目录

- [产出目标（你最终要拿到什么）](#产出目标你最终要拿到什么)
- [核心概念与术语](#核心概念与术语)
- [推荐目录结构（约定）](#推荐目录结构约定)
- [Week 1：跑通“感知”与“闭环骨架”（9h）](#week-1跑通感知与闭环骨架9h)
- [Week 2：把它变成“完整系统”（9h）](#week-2把它变成完整系统9h)
- [Week 3：打磨成“面试可讲”的作品（9h）](#week-3打磨成面试可讲的作品9h)
- [验收清单（最终检查）](#验收清单最终检查)

---

## 产出目标（你最终要拿到什么）

- **可运行 Demo**：`python demo.py --camera 0 --aruco --auto-exposure --roi-zoom`
- **可复盘证据**：状态变迁日志 + 关键帧截图（`logs/`）
- **可对比评估**：`before`（不开闭环） vs `after`（开闭环）的成功率/不确定性对比
- **可讲清楚的故事**（10 分钟）：为什么需要 active perception、不确定性如何定义、动作集合如何选、策略如何工作、如何评估与失败

---

## 核心概念与术语

- **Perception（感知）**：从图像里检测目标（ArUco/QR），输出 `detected`、`id`、以及可选的置信度 proxy。
- **Uncertainty（不确定性）**：每帧一个 \(0\sim1\) 的分数，越大代表“越不确定/越不可靠”。最简但有效：`没检测到 -> 高；检测到 -> 低`，再叠加清晰度（sharpness）等指标。
- **Action space（动作集合）**：系统可执行的动作，如“调整曝光档位 / brightness / gamma / ROI zoom”。
- **Closed-loop（闭环）**：当不确定性高时，系统主动执行动作改变观测条件，并选择更优动作以恢复稳定检测。

---

## 推荐目录结构（约定）

> 这是为了让 Week2 的工程化（状态机/日志/CLI）更顺滑；你也可以按自己习惯微调。

- `src/camera.py`：相机读取与显示
- `src/perception.py`：ArUco/QR 检测
- `src/uncertainty.py`：清晰度与不确定性计算
- `src/policy.py`：动作集合（曝光/软件增益/ROI zoom 等）与动作执行函数
- `src/loop.py`：闭环主循环（状态、探索、稳定）
- `demo.py`：命令行 Demo 入口
- `logs/`：关键帧截图、日志输出（运行时生成）

---

## Week 1：跑通“感知”与“闭环骨架”（9h）

### Session 1（周一 1h）：环境 & repo

- **目标**：能打开摄像头并显示画面
- **步骤**
  - 新建 repo + 目录结构
  - 建 venv，安装依赖：`opencv-python`、`numpy`
  - 编写 `src/camera.py`
    - 打开摄像头：`cv2.VideoCapture(0)`
    - 读取帧并显示：`cv2.imshow(...)`
- **完成标准**
  - 运行 `python -m src.camera` 能看到实时画面
  - 按 `q` 退出正常

### Session 2（周二 1h）：加入 ArUco/QR 检测（Perception 模块）

- **目标**：在画面里检测到 marker，并画框/输出 id
- **选择**
  - **ArUco**：OpenCV 内置（推荐）
  - **QR**：OpenCV `QRCodeDetector` 也可以（可作为备选或扩展）
- **步骤（ArUco）**
  - 编写 `src/perception.py`
    - 初始化字典：`cv2.aruco.getPredefinedDictionary(...)`
    - 检测：`detectMarkers(frame, dict)`
    - 检测到则画 marker 边框，并输出 id
  - 输出字段建议：
    - `detected: bool`
    - `confidence`（proxy：marker 数量 / 角点质量等）
- **完成标准**
  - 拿打印的 marker 在镜头前移动，能稳定检测并画框

### Session 3（周三 1h）：定义不确定性（Uncertainty）

- **目标**：每帧给一个 `uncertainty score`
- **定义（简单但有效）**
  - 检测到 marker：uncertainty 低
  - 没检测到：uncertainty 高
  - 叠加清晰度指标让它更像系统：
    - `sharpness = Laplacian variance`
    - `sharpness` 低 → `uncertainty` 增加
- **实现**
  - 新增 `src/uncertainty.py`
  - 函数建议：
    - `compute_sharpness(frame)`
    - `compute_uncertainty(detected, sharpness) -> float  # 0~1`
- **完成标准**
  - 终端每帧打印：`detected / sharpness / uncertainty`

### Session 4（周四 1h）：定义动作集合（Action space）

- **目标**：能在代码里设置相机参数，至少支持 2–3 档曝光
- **步骤**
  - 检查相机是否支持：`cv2.CAP_PROP_EXPOSURE`
  - 在 `src/policy.py` 里定义动作：
    - `exposure_levels = [-8, -6, -4, -2]`（不同相机范围不同，需要试）
    - `set_exposure(cap, value)`
- **完成标准**
  - 手动切换 exposure 后，画面亮度明显变化
- **重要提示（别卡住）**
  - 如果 webcam 不支持曝光控制：立刻换成动作 **软件 brightness/gamma** 或 **ROI zoom**（Week2 会体系化成状态机 + 搜索策略）。

### Session 5（周五 1h）：搭闭环 Skeleton（Loop）

- **目标**：当 `uncertainty` 高时，自动尝试不同曝光并记录哪档最好
- **步骤**
  - 编写 `src/loop.py`
    - 维护当前 exposure index
    - 每帧计算 `uncertainty`
    - 若 `uncertainty > threshold`：切换到下一档 exposure
    - 记录统计：
      - 对每个 exposure level，统计最近 \(N\) 帧检测成功率（例如 10 帧）
- **完成标准**
  - 在暗光/反光下，系统会自动切曝光尝试让检测恢复

### 周六（2h）：选择“最优 exposure”策略（关键一步）

- **目标**：系统不是乱试，而是能选出“最好的一档”并进入稳定
- **实现思路**
  - 对每个 exposure level 维护 `score`，例如：
    - `success_rate`：过去 10 帧 detect 比例
    - `sharpness_mean`：清晰度均值
  - 每次探索完一轮，选择 `score` 最大的 exposure，进入“稳定模式”
  - 输出日志字段：
    - `mode: explore / stabilize`
    - `best_exposure`
- **完成标准**
  - marker 放在暗处：系统探索后稳定在某个 exposure 并持续检测

### 周日（2h）：加入 ROI Zoom（更“主动感知”）

- **目标**：识别不到时，系统自动 zoom-in 中心区域再试
- **做法（简化版）**
  - 连续 \(K\) 帧检测失败：
    - 将 `frame` 裁剪中心 50%（zoom）
    - 在 zoom frame 上做 detection
    - 检测到后切回 full frame
- **完成标准**
  - marker 更远/更小：系统自动裁剪后更容易识别

---

## Week 2：把它变成“完整系统”（9h）

### Session 6（周一 1h）：加状态机（更像工程系统）

- **目标**：用状态机组织逻辑，避免一堆 `if/else` 难以维护
- **状态建议**
  - `TRACKING`：稳定检测
  - `RECOVERY`：不确定，开始探索
  - `SEARCH`：ROI zoom / exposure sweep
- **完成标准**
  - 代码结构清晰可读，状态切换规则明确

### Session 7（周二 1h）：加入 Temporal smoothing（稳定输出）

- **目标**：输出更稳定、不抖动
- **做法**
  - `uncertainty`：滑动平均（窗口 5）
  - `detected`：majority vote（窗口 5）
- **完成标准**
  - 日志/可视化输出明显更稳定

### Session 8（周三 1h）：统一日志 & 保存关键帧

- **目标**：可复盘、可展示
- **做法**
  - 状态变化时保存截图到 `logs/`
  - 记录 `exposure / uncertainty / detected`
- **完成标准**
  - 有“前后对比”的可复盘证据

### Session 9（周四 1h）：写一个 CLI demo 脚本

- **目标**：一条命令跑起来
- **命令形式**
  - `python demo.py --camera 0 --aruco --auto-exposure --roi-zoom`
- **完成标准**
  - 任何人 clone 后按命令即可运行 Demo

### Session 10（周五 1h）：写 evaluation 指标（简单版）

- **目标**：用最小评估闭环是否有效
- **输出两组指标**
  - `before`（不闭环）：检测成功率、平均 uncertainty
  - `after`（闭环）：同上
- **完成标准**
  - 能打印对比结果（最好能保存到 `logs/`）

### 周六（2h）：极端场景测试

- **目标**：刻意制造失败场景，展示系统如何恢复
- **三种情况**
  - 低光
  - 反光
  - 运动模糊
- **完成标准**
  - 每种情况都能展示：状态变化 + 动作执行 + 恢复过程

### 周日（2h）：清理代码 + README 骨架

- **README 必须包含**
  - Problem
  - Architecture 图（简单框图）
  - Key design choices
  - Failure cases
  - How to run
- **完成标准**
  - 任何人 clone 都能跑

---

## Week 3：打磨成“面试可讲”的作品（9h）

### 周中 5 个 Session（每天 1h）

- 补充注释、类型、异常处理
- 把架构图画成一张清晰图片（PPT/手绘都行）
- 整理 `logs/`：挑 3 个最典型的前后对比
- 写清楚：
  - `Trade-offs`
  - `Future work`

### 周末 2×2h

- 做一次完整录屏演示（窗口录制也行）
- 最后一次 repo polish：README、requirements、运行说明

### 完成标准（面试叙事）

你能用 10 分钟讲清楚：
- 为什么需要 active perception
- uncertainty 怎么定义
- action space 怎么选
- 策略如何探索/稳定
- 怎么评估、怎么失败、如何改进

---

## 验收清单（最终检查）

- [ ] `python -m src.camera` 能打开摄像头并退出正常
- [ ] ArUco/QR 检测可用，画框/输出 id 正常
- [ ] 每帧有 `uncertainty` 输出，且与“检测情况/清晰度”一致
- [ ] 闭环能触发动作（曝光或替代动作），并能选择更优动作进入稳定
- [ ] 状态机清晰，日志/截图可复盘（`logs/`）
- [ ] `demo.py` 一条命令可跑
- [ ] `before/after` 指标可对比
- [ ] README 包含 Problem/Architecture/Choices/Failure/Run