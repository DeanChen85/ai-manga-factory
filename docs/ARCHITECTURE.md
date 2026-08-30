# 架构

这是一个**有状态的生产系统**，不是一个单一的 ComfyUI 工作流。每个状态转换都被持久化且可恢复。

```text
主题 + 梗概 + 集数 + 每集秒数
  -> MiniMax 结构化剧集/分集合同
  -> 人工创意审批
  -> 人物 + 场景参考资产生成
  -> 逐资产预览 + 哈希绑定人工审批
  -> 确定性 H3 导演编译器
  -> 低成本 proof 预演（不可交付）
  -> 解码内容 QA + 首/中/尾证据 + 人工晋级
  -> 正式 H3 渲染（使用锁定的 prompt/reference 哈希）
  -> 逐镜 QA + 人工审批
  -> 整集发布审批
  -> H.264/AAC 720p 平台母版 + 字幕/manifest/ZIP
```

## 职责边界

- `story_splitter.py` 问 LLM 要结构化创作合同。LLM **不直接画 Comfy 图**。
- `h3_director.py` 确定性把已批准字段编译成官方 H3 base / Ref2VA 段形。
- `render_video_h3.py` 布参考、冻结 prompt/reference 哈希、提交图。
- `task_store.py` 持久化任务、重试、proof 晋级、review evidence、release gates。
- `render_service.py` 和 `series_service.py` 是唯一对外 UI 门面。
- `web_app.py` 是用户控制台；**不**写 SQLite，**不**直调 ComfyUI。
- `video_delivery.py` 仅从发布批准的正式 artifact 创建平台母版。

## 连续性模型

人物身份和服装来自已批准的角色参考；场景布局和光线来自已批准的 scene/composition 参考。提示词标签遵循实际 ComfyUI 接线顺序。每镜记录精确的 cast 数、首态、单一主导动作、终态、画面方向、单镜头路径。

当前 Ref2VA 节点把参考图视为多模态身份/风格/布局参考，**不会**把图片标签当成有保证的硬关键帧。上游 ComfyUI 新加了 `MiniMaxH3AddGuide` 支持任意帧硬引导；夜间预检把这一能力作为**推荐能力**报告——当前本地 ComfyUI 0.33.2 未暴露该节点，所以本产品不会把语义参考冒充为硬时间锚点。

## 音频、对白与字幕

H3 同时生成原生音频与视频，但精确口型同步是概率性的。已批准的对白嵌入 H3 提示词一次并存为 cue sheet。交付字幕是**确定性后期合成**，**不**在提示词中请求画面内文本。高精度 ADR/TTS 单独保留接口。

## 韧性

- SQLite 持久化每个 job 和 worker 租约。
- job input hash 阻止未变更的成功任务被重新生成。
- artifact / prompt / reference-bundle / edit-selection / decoded-visual 五类哈希把审批绑定到精确字节。
- 失败任务可在不丢审计证据的前提下续跑。
- 夜间执行走单 GPU 租赁，缺节点 / 队列忙 / 低空间 / 低 VRAM / 温度 / 重试预算任一即 fail-closed。
