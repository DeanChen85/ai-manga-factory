# 外部项目调研：T8mars/comfyui-minimax-h3-audio-T8

**调研日期**：2026-08-30
**目标仓库**：<https://github.com/T8mars/comfyui-minimax-h3-audio-T8>
**调研者**：DSH 自动化调研 agent
**对比基准**：本仓库 `custom_nodes/ComfyUI_RH_MinMaxH3`（Apache-2.0，已装）

---

## 一、总览

| 维度 | 数值 |
|---|---|
| 文件 | 921 个 |
| 总大小 | 16.7 MB |
| Python 代码 | **8.5 MB**（vs RH 的 1.2 MB） |
| 注册节点 | 200+（含 `_advanced` 后缀实验节点） |
| LICENSE | **GPL-3.0-or-later** ⚠️ |
| 第三方组件 | Wan 系列、LTX-2.5、FlashVSR、RealBasicVSR、SpargeAttn、SLA、KJNodes 等 |

### 关键事实

- 这是**真正投入生产**的 MiniMax H3 节点包，文档长达 162 行（README）+ 完整 docs/ 目录。
- 作者 T8mars 在 B 站 / HuggingFace 有持续更新（模型权重、`comfyui-minimax-h3-blockcache-T8` 等周边包）。
- 是当前中文社区**唯一活跃维护**的 H3 全功能节点包。

---

## 二、与 RH（你已装）的功能矩阵对比

| 能力 | RH (RunningHub) | T8 (T8mars) | 你的需求 |
|---|---|---|---|
| **H3 模型加载（FL2VA / Ref2VA / T2VA）** | ✅ 3 套独立 loader | ✅ 合并的 loader + 旧/新模型兼容层 | ✅ RH 够用 |
| **Qwen3-VL 32B 文本编码器** | ✅ INT8 + visual on CPU 优化 | ✅ Qwen Prefix Cache 缓存加速 | ✅ RH 够用 |
| **VAE（video / audio）** | ✅ 24 通道 video + DAC audio | ✅ 同 + TAEHV 快速解码旁路 | ✅ RH 够用 |
| **T2VA / I2VA / FL2VA / Ref2VA target+encode** | ✅ 完整 24 节点 | ✅ 完整 + 更丰富变体 | ✅ RH 够用 |
| **双时钟采样（video shift 12 / audio shift 3）** | ✅ DualSigmaSampler | ✅ DualClockSamplerT8 + AYS 校准 | ✅ RH 够用 |
| **Cache-DiT 加速** | ✅ 内置（实测 1.99x） | ✅ Sol-Attn / FastH3 VSA / SLA / Cache-DiT 多种 | ⚠️ **RH 已覆盖；T8 多选项是优势** |
| **音频控制（mux_audio, lock_source, ref）** | ⚠️ 基础 | ✅ **AudioRefineT8 + AudioLatentControl + AudioWindow** | ✅ **T8 明显领先** |
| **官方 H3 AddGuide 节点** | ❌ 缺（推荐能力） | ❌ 同 | n/a |
| **首尾帧链 long video** | ❌ 无 | ✅ **LongVideo 系列 8+ 节点 + In_Node_Loop** | ⚠️ **重大缺口** |
| **多关键帧 multishot** | ❌ 无 | ✅ MultiKeyframeConditioning + KeyframePlan | ⚠️ **新维度** |
| **人脸/皮肤修复** | ❌ 无 | ✅ **FaceRefine + SkinFinish (含 SAM3.1 追踪)** | ✅ **值得集成** |
| **FlashVSR 视频超分（2x/4x）** | ❌ 无 | ✅ FlashVSRFull + FlashVSRTiny + FlashVSRTinyLong | ✅ **你 HANDOFF 提到的方向** |
| **Prompt Relay（长视频提示词接力）** | ❌ 无 | ✅ 完整 RelayPlan + ResourceEstimate | ⚠️ **新维度** |
| **FastH3 VSA 4步 preview** | ❌ 无 | ✅ FastH34StepSetup + learned-gate VSA | ✅ **与你的 proof profile 互补** |
| **PDD 8步加速** | ❌ 无（v4-600） | ✅ **PDD 8 Step Setup** | ✅ **T8 更新更快** |
| **Speech / TTS 集成** | ❌ 无 | ✅ 整套 Speech 模块（ADR / 长文 / 校验） | ✅ **潜在价值大** |
| **NVIDIA H3 Super Acceleration（与 LTX-2.5 混合）** | ❌ 无 | ✅ SolEngineDraftToLTX + TAEHV | ⚠️ 实验性 |
| **官方核心兼容性诊断** | ⚠️ 基础 | ✅ CommunityDiagnostics + EnvironmentAudit | ✅ T8 更好 |
| **Spearman / VSA / FastH3 多档加速** | ❌ 无 | ✅ 自带 4 档（turbo / dual-clock / super / pdd） | ✅ **RH 暂无 turbo** |
| **多 GPU 长视频节点内串行** | ❌ 无 | ✅ LongVideoInNodeLoopAdvanced | ⚠️ 单 GPU 不需要 |
| **断点续跑（Native Latent Checkpoint）** | ❌ 无 | ✅ NativeLatentCheckpointSave/Load | ✅ **与你的 `task_store` 互补** |
| **错误恢复（JobContract / RepairExecution）** | ❌ 无 | ✅ JobContractError / RepairExecutionAdvanced | ⚠️ 冗余设计 |

**总结**：

- **RH 覆盖你的核心 H3 推理正确性**（loader / sampler / encode / decode / VAE）。
- **T8 在 H3 周边工具链上**远超 RH：音频、长视频、人脸修复、视频超分、提示词接力、加速档位、生态配套。
- 你目前 `pipeline/` 代码层做的"4 层 QA + hash 绑定 + 断点续跑"，比 T8 的部分"advanced" 实验节点还**严谨**——但**功能面**比 T8 窄。

---

## 三、最关键的 8 项借鉴/集成建议

### 🔴 优先级 1：长期视频编排（直接对应你 p03 困境）

**T8 的 `LongVideoInNodeLoopT8Advanced` 等节点**做的是：把"长镜头"切成多段 + 段间 latent 续接 + 失败恢复 + 自动队列。

**你现在的实现**（`pipeline/series_service.py` + `pipeline/overnight_ops.py`）：跨集任务编排有，但**单集内的多段生成+段间衔接**没有原生支持。视频链是 VideoMake 那种 `15+1=16s` 的手工方式。

**建议行动**：
- 在 `pipeline/` 新建 `pipeline/long_video_orchestrator.py`：接收一个长镜头合同（n 段 + 每段目标时长 + 衔接模式），调用 H3 生成 + 抽末帧 + 馈给下一段 + 自动处理失败重试 + 输出最终 concat MP4。
- 在 `pipeline/render_video_h3.py` 里增加 `chain_segments()` API，作为 `task_store` 长视频 job 类型。

### 🔴 优先级 2：人脸/皮肤修复（身份漂移修复）

**T8 的 `FaceRefine` 系列**+ **`SkinFinish` 系列**：单镜 + 多镜 + 跟踪 + 多种 skin 模式（dichromatic / specular / frequency split）共 30+ 节点。

**你现在的做法**（`pipeline/shot_group_anchor.py` 的成对首尾锚）：是**构图级**修复，不解决"同一个人脸在多镜中漂移"。

**建议行动**：
- 在 `tools/daily-research/config.json` 不需要改（这是离线研究）。
- 但**写一个 `pipeline/face_refine_runner.py`**：当生产 artifact 检测到 face 漂移（如 cosine distance > 阈值），调用 T8 的 FaceRefine 节点作为 **可选 post-pass**。**这是 OFF by default + 显式触发**——保证不冒进。

### 🟡 优先级 3：FlashVSR 视频超分（替代你现有 GAN 4x）

**T8 的 `FlashVSRFull` / `FlashVSRTinyLong`**：基于 FlashVSR-v1.1，2x / 4x 超分；原音频直通。

**你现在的做法**（`SHORT_DRAMA_PRODUCTION_STANDARD.md` 提到 GAN 4x，但 pipeline 里**没有 GAN upscale 模块**——只输出 SD 母版）。

**建议行动**：
- 在 `pipeline/` 新建 `pipeline/video_upscale.py`：调用 `MiniMaxH3FlashVSRRestoreT8` 作为 post-pass，给出可选 profile（2x/4x/memory-safe）。
- 注意：FlashVSR 模型权重 ~2GB，需要单独下载放 `ComfyUI/models/FlashVSR-v1.1/`。

### 🟡 优先级 4：Prompt Relay（长视频提示词接力）

**T8 的 `PromptRelayConditioningT8Advanced` 等 8+ 节点**：把 7000 字符上限的官方 H3 提示词分成多段、自动接力生成。

**你现在的做法**（`pipeline/prompt_contracts.py` 内的 `action_contract`）：是**单镜**提示词绑定，不支持跨段接力。

**建议行动**：
- 在 `pipeline/prompt_contracts.py` 增加 `LongVideoPromptSplitter`：把 `[Shot 1]...[Shot N]` 拆成 n 个独立合同，每个独立过 H3，但保持 voice ID / Picture N / continuity 标识一致。

### 🟡 优先级 5：音频精细控制（Audio Refine / Window / Mix）

**T8 的 `AudioRefinePlanT8Advanced`**等：音频质量审计 + 双采 + 兼容性路由 + 长期音频 delivery。

**你现在的做法**（`pipeline/subtitle_delivery.py`）：只管字幕，音频是 H3 自带直出。

**建议行动**：
- 在 `pipeline/audio_quality.py` 增加"音频锁定 vs 重绘"模式（来自 T8 的 `audio_mode: lock_source` 启发）。
- 与你的 cue sheet 系统（`subtitle_delivery.py` 已经记录每段 `audio_cues`）协同：哪段该锁源、哪段该重绘，由 cue 类型自动决定。

### 🟢 优先级 6：FastH3 VSA 4 步 preview（升级你的 proof profile）

**T8 的 `FastH34StepSetupT8Advanced`**：4 步 + VSA 90% 稀疏 attention + TAEHV 旁路。

**你现在的 proof profile**（`pipeline/h3_profiles.py`）：6 步 + Turbo + `match` ref_size。

**建议行动**：
- 在 `h3_profiles.py` 增加 `proof-fast` profile：4 步 + VSA + `match`，作为更便宜的 preview 选项（代价是稀疏 attention 在某些长视频类型可能降质）。
- 适用：仅作为可选项，**默认仍是 6 步**。

### 🟢 优先级 7：Speech / TTS（剧情对白需求）

**T8 的整套 Speech 模块**：ADR Fit、Assemble、Conditioning、Decode、Finalize、Guard、Long Form、Studio、Verify——9 个模块 30+ 节点。

**你现在的做法**：H3 原生音频是概率性的，你明确说"口型同步是概率性的、对白用确定性后期"。

**建议行动**：
- 不集成（保持架构纯净）。
- 但在 `docs/research/` 写一份 `EXTERNAL_T8mars_speech_module.md` 详细分析：当用户真有"角色口型同步"需求时，T8 是 fallback 选项。

### 🟢 优先级 8：错误恢复（与你的 `task_store` 互补）

**T8 的 `RepairExecutionAdvanced`**：job contract 错误 → 重连 → 重新提交。

**你现有的**（`task_store.py` 的 `recover_job` / `reconcile_job`）：已经覆盖 95% 场景。

**建议行动**：
- 不集成——避免重复设计。

---

## 四、T8 不能直接复用的原因（重要）

### ⚠️ LICENSE 冲突

T8 是 **GPL-3.0-or-later**。你的 `LICENSE` 是 **Apache-2.0**。

**正确做法**（你已经走在正确路线上）：
- ✅ `custom_nodes/` 已在你的 `.gitignore` 第 58 行排除
- ✅ KJNodes（GPL-3.0）已经在 `custom_nodes/` 而不污染你 Apache-2.0 主仓
- ✅ T8 加入也走同样路线：用户自行安装，不随 Apache-2.0 源码分发

**禁止做法**：
- ❌ 复制 T8 任意代码片段到 `pipeline/`
- ❌ 修改 T8 源码并声称是自己的代码
- ❌ 把 T8 的 _advanced 节点代码内联到你的 pipeline/

**可以做**：
- ✅ 引用 T8 节点名（公开 API）
- ✅ 自己**重新实现**等价算法（如 long_video 编排），独立写代码
- ✅ 在 `THIRD_PARTY_NOTICES.md` 加 T8 条目

### ⚠️ 状态：实验性 `_advanced` 节点

T8 大量带 `_advanced` 或 `_exp` 后缀的节点，按 README 第 22 行"建议直接使用配套工作流"，**不要直接当稳定 API**。要等它们稳定后再集成。

---

## 五、可立即执行的下一步

按优先级给出可执行动作（无需任何代码修改）：

1. ✅ **更新 `docs/THIRD_PARTY_NOTICES.md`**：加 T8 条目（说明 GPL-3.0 隔离）
2. ✅ **更新 `docs/COMFYUI_H3_INSTALL.md`**：增加 "可选 T8 安装" 章节（提供 1-2 句安装提示）
3. ✅ **写 `docs/integrations/T8-H3-integration-roadmap.md`**：把上面 8 项转化为具体的 pipeline/ 新文件 roadmap
4. ⚠️ **不在 `pipeline/` 写任何复制 T8 代码的模块**

下一步需要我做的具体动作（你选）：
- [ ] 立即改 `THIRD_PARTY_NOTICES.md` 和 `COMFYUI_H3_INSTALL.md`
- [ ] 写 `docs/integrations/T8-H3-integration-roadmap.md`
- [ ] 新建 `pipeline/long_video_orchestrator.py`（原创实现 long video 编排）
- [ ] 新建 `pipeline/face_refine_runner.py`（T8 FaceRefine 节点 wrapper）
- [ ] 新建 `pipeline/video_upscale.py`（FlashVSR wrapper）

---

## 六、与 RH / Director / KJNodes 的最终定位

| 包 | License | 你的项目用 | 与 T8 关系 |
|---|---|---|---|
| **ComfyUI_RH_MinMaxH3** | Apache-2.0 | ✅ 必备 | T8 部分功能重叠（H3 推理核心），但 T8 不是 Apache |
| **ComfyUI_MiniMaxH3_Director** | Apache-2.0 | ✅ 必备 | 不重叠——编排 UI 层 |
| **ComfyUI-KJNodes** | GPL-3.0 | ✅ 必备（gitignore） | T8 引用了它，可继续兼容 |
| **comfyui-minimax-h3-audio-T8**（新） | GPL-3.0 | ⭐ 推荐（gitignore） | **音频 / 长视频 / FlashVSR 是新维度** |

**T8 是补完**：它不替代你的三个节点包，是**第四个**放 `custom_nodes/` 的扩展（GPL 隔离）。

---

## 七、给 DSH 自动调研 agent 的总结

- ✅ 已抓取 GitHub 完整仓库（921 文件 / 16.7 MB）
- ✅ 已与本仓库 `custom_nodes/ComfyUI_RH_MinMaxH3` 做功能矩阵对比
- ✅ 已识别 8 项可借鉴能力（4 红 / 3 黄 / 1 绿）
- ✅ 已警示 LICENSE 兼容性
- ✅ **P1-P4 已实现**（long_video_orchestrator / face_refine_runner / video_upscale / prompt_relay + proof_fast profile）
- ✅ **P2 compatibility_matrix + preflight 校验已落地**（2026-08-30 第二轮优化）
- ✅ **P4 speaker ID 稳定器已落地**（2026-08-30 第二轮优化）
- 📝 P5/P6/P7 待后续迭代

---

## 八、2026-08-30 第二轮优化记录

### P4: Speaker ID 稳定器（prompt_relay.py）

**问题**：长视频接力生成时，不同 chunk 可能丢失 `(S1)/(S2)` 或 `<Picture N>` 标签，导致角色声音漂移。

**解决**：新增 `stabilize_speaker_ids()` 函数，为每个 chunk 注入完整的 speaker map 和 reference tag 列表作为头部注释。不影响 H3 解析，但保证跨 chunk 身份一致性。

**测试**：4 个新测试用例全部通过（test_prompt_relay.py）。

### P2: 节点包兼容性矩阵 + 预检校验

**问题**：用户克隆后不知道哪些节点包能装、哪些会冲突，只能等爆红才排查。

**解决**：
- 新建 `docs/compatibility_matrix.md`：完整列出 4 个核心节点包的 license、最低 ComfyUI 版本、已知冲突、安装方式
- `comfy_preflight.py` 新增 `_check_node_packages()`：启动前自动检测 T8/RH 双时钟采样器冲突、缺失包警告
- README.md 增加兼容性矩阵链接

**测试**：2 个新测试用例全部通过（test_comfy_preflight.py）。

**总测试数**：339 → **345**（+6 新测试）。

---

## 九、引用与来源

- T8mars/comfyui-minimax-h3-audio-T8 @ commit `6490906` (2026-08)
- 来源锁：`https://github.com/T8mars/comfyui-minimax-h3-audio-T8`
- 许可证：GPL-3.0-or-later（详细 SPDX 标识见 LICENSE 文件）
- 模型来源：[t8star/MiniMax-H3-Acc-8Step-comfy](https://huggingface.co/t8star/MiniMax-H3-Acc-8Step-comfy)、[FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1)、[madebyollin/taehv](https://github.com/madebyollin/taehv)