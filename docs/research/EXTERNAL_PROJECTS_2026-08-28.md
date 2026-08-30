# 外部项目复用评审（2026-08-28）

每行 = `source / authority / license_note / reusable / current decision / rollback`。

| 项目 | 决策 |
|---|---|
| MiniMax H3 官方仓库 | 已以 clean-room source-lock 编译器契约承载：保 6 段 Ref2VA 顺序、英文结构、显式 Picture 角色、hash 审计；**不** vendor guide 文本 |
| MiniMax CLI H3 video 指南 | 校验要求采纳：精确时间窗、显式首/终态、超时后禁重发 |
| ComfyUI server routes | `/prompt`、queue/history 恢复、`/free` 已用；`/ws` 是下一 P1，**SQLite 仍是重连权威** |
| WanAnimate2ToVideo | 仅评测、可选后端，**不可静默替换 H3**；未来"困难演员控制"镜再 A/B |
| WanSCAILToVideo / SCAIL-2 | 仅复用"持久 segment-state"设计；**不**往当前 H3 workflow 注 Wan 图 |
| Filmclusive ComfyUI workflows | **仅**比较图拓扑；资质/节点/VRAM/720p 通过前**不**导入 |
| ComfyUI-Expert video pipeline | FP8/Sage/RIFE/deflicker 仅作为 opt-in 实验 profile；**永不**因论坛说法全局启用；先本机短 sample benchmark |
| 2D character pipeline | 仅复用 proof-first 思路；当前保留成对锚点用于 final-state 控制 |
| `r600a-code/minimax-h3-prompt-skill` | 仅用作发现索引；不复制其中上游文本 |

## 自带优势（vs 一键 workflow）

- 合同绑故事/shot/参考顺序/prompt hash/artifact hash/decoded-visual hash/edit window。
- proof 显式不可交付。
- 被拒 artifact 入档不参与合成。
- final MP4/ZIP 路径对内容 QA、单镜审、整集发布、编解码/流/时长检查、ZIP 重开校验**全部 fail-closed**。

## 剩余 gap

1. ComfyUI 进度应消费官方 WebSocket 事件，重连后保留 SQLite/history reconciliation。
2. 长视频 adapter 需持久 `segment_index` / `previous_frames_sha256` / `video_frame_offset`，老 segment 不可盲重发。
3. Wan/SCAIL/FramePack/RIFE profiles 需隔离安装 + 许可证审查 + 可复现 720p A/B 通过。
4. 平台发布仍人工。

## 研究更新规约

每条 entry 记 `source_url` / `checked_at` / `license` / `what_reused` / `compatibility` / `decision` / `rollback`。**社区说法是假设，必须目标机复验**。绝不上传：`.env` / keys / 本地 DB / 生成项目 / 用户媒体 / 模型权重 / 机器特定路径。
