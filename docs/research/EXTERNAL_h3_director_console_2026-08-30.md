# H3 导演台 / "闲兔" 社区 搜索记录

**调研日期**：2026-08-30
**触发**：用户听说"闲兔"社区有一个完整 H3 导演台工作流
**结果**：未在公开 GitHub 找到名为"闲兔 / XianTu"的 H3 导演台项目；但发现 6 个相关候选

---

## 一、搜索关键词覆盖

| 关键词 | 命中 |
|---|---|
| `闲兔` | 0 |
| `闲鹿` | 0 |
| `XianTu` | 0 |
| `MiniMax H3 导演台` | 多个相关项目（见下） |
| `H3 director console` | 1 个高度相关：karuvanan/MiniMax-H3-Director-Cut-Studio |
| `H3 directing workbench` | 0（直接命中） |
| `MiniMax h3 short drama` | 7 个相关项目 |

**结论**：

- "闲兔" 可能是一个**微信/Discord 群**或**未公开的私有仓库**——GitHub 公开搜索没有命中。
- 如果它在 Discord / 飞书群 / 微信公众号上，需要你贴 URL 或邀请链接，我才能继续调研。
- 同时，**找到 6 个 GitHub 上现存的 H3 导演台/工作流项目**，按价值列出。

---

## 二、GitHub 候选项目（按价值排序）

### 🔴 1. `karuvanan/MiniMax-H3-Director-Cut-Studio` ⭐88

- **描述**：Premiere-inspired PySide6 director studio for MiniMax H3 Ref2VA with AI shot planning, semantic media enrichment, timeline prompt reconciliation and shot-aware long-video rendering through ComfyUI
- **License**：NOASSERTION ⚠️
- **特点**：
  - PySide6（Qt）桌面应用，类 Premiere 时间线 UI
  - 内置 shot planning AI
  - 内置 semantic media enrichment
  - **长视频节点内渲染**（与 T8 的 LongVideoInNodeLoopT8Advanced 目标一致）
- **借鉴价值**：⭐⭐⭐⭐（4/5）—— **PySide6 桌面工作台范式 + AI 镜头规划**值得参考
- **风险**：NOASSERTION 不是 OSI 标准 license —— 不能 fork 合并，只能**思路借鉴**

### 🔴 2. `Songssx/ComfyUI-MiniMaxH3-TimelineDirector` ⭐80

- **描述**：Editable reference-media timeline director for ComfyUI MiniMax H3 Reference to Video
- **License**：GPL-3.0 ⚠️
- **特点**：
  - ComfyUI 内的 timeline editor 节点
  - 直接编辑参考媒体时间线
  - 与 T8 `MiniMaxH3TimelineDirector` 思路类似
- **借鉴价值**：⭐⭐⭐⭐（4/5）—— **timeline editor UI 模式**可参考
- **风险**：GPL-3.0 —— 与 KJNodes 一样只能在 `custom_nodes/` 隔离使用

### 🟡 3. `MikuLXK/ComfyUI-XYUE-H3-Studio` ⭐2

- **描述**：MiniMax H3 ComfyUI nodes with two multi-stage short-drama workflows
- **License**：未列 ⚠️
- **特点**：
  - 提供**两个完整多段短剧工作流**——你的项目缺的实战工作流示例
- **借鉴价值**：⭐⭐⭐（3/5）—— 可用作工作流参考

### 🟡 4. `easyeye163/h3-video-coding` ⭐1

- **描述**：AI short-drama production workflow: MiniMax H3 6-segment prompts + RunningHub API + Feishu integration
- **License**：未列 ⚠️
- **特点**：
  - **6 段 H3 短剧 prompt 工程**（与你 p01-p05 类似）
  - **RunningHub API**（不是本地 ComfyUI）
  - **集成 Feishu 通知**——你已经用 webhook 实现类似功能
- **借鉴价值**：⭐⭐⭐（3/5）—— prompt 工程 + Feishu 模式可参考
- **风险**：不是 self-hostable，不适合你做本地工程化

### 🟡 5. `coconilu/h3-short-drama-studio` ⭐1

- **描述**：本地优先的 MiniMax H3 + ComfyUI 短剧生产工作台
- **License**：未列 ⚠️
- **特点**：与你"本地优先"定位完全一致
- **借鉴价值**：⭐⭐（2/5）—— 看具体内容后再定

### 🟡 6. `catiseyeqaq/ai-manju-shengcheng-xitong` ⭐4 (MIT)

- **描述**：AI Film & Short Drama Generation System: ComfyUI + MiniMax-H3, Qwen3.6 prompt polishing, 8×PPU-ZW810E cluster deployment
- **License**：MIT ✅
- **特点**：
  - **8 卡 PPU-ZW810E 集群**——与你单卡 RTX 3090 路径不同
  - 但 MIT 许可**可借鉴**
- **借鉴价值**：⭐⭐（2/5）—— 集群模式对你无直接用

---

## 三、给"闲兔"的进一步搜索建议

如果你希望我继续找"闲兔"社区，需要你提供以下任一：

1. **直接 URL**（Discord 邀请链接、飞书群二维码图片、GitHub 组织名）
2. **公众号/博客链接**（一篇介绍文章就能让我搜到对应作者）
3. **作者用户名**（你可能记起"闲兔"是谁的代号）

可能的命名（用户口述时可能听错）：
- `xiantubot` / `xiantu-ai`
- `XianTuStudio` / `XianTuAI`
- `空兔` / `仙兔` / `闲图`（同音字）
- 在 B 站、知乎、CSDN 上的 "闲兔" 关键词搜索结果

---

## 四、决定

由于：
- 你要求我 "针对可以借鉴的能力你就更新到我们的系统和 github上面 这样可以不断优化"
- 已经发现 6 个 GitHub 项目，**其中 2 个值得立刻评估借鉴**

**下一步建议（按价值排）**：

1. **继续实现 T8 priority 5-7**（Audio Refine policy / FastH3 VSA profile / Speech 评估）
2. **Clone 并评估 `karuvanan/MiniMax-H3-Director-Cut-Studio`**（⭐88，与你项目定位最匹配）
3. **Clone 并评估 `Songssx/ComfyUI-MiniMaxH3-TimelineDirector`**（⭐80，timeline editor 思路）
4. **等待你确认"闲兔"的具体信息**（URL/作者/平台）

如果你给我一个反馈方向，我立刻执行。否则我继续按"持续优化"模式推进 T8 剩余项。

---

## 五、本调研与 8.30 已完成 T8 集成的关系

本调研与 `EXTERNAL_T8mars_minimax-h3-audio-T8_2026-08-30.md` 互补：
- 那份对比 T8mars **官方 H3 节点包**（最全面）
- 本份对比**创作者侧 / Director 工具侧**的多个项目
- 两者合起来给出 H3 生态的完整视野

## 六、引用与来源

- 搜索时间：2026-08-30
- 搜索 API：GitHub REST `/search/repositories`
- 6 个候选项目均通过 GitHub API 验证存在
- License 信息：来自 `https://api.github.com/repos/{owner}/{repo}` 返回的 `license.spdx_id`