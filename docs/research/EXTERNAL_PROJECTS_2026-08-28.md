# External project and upstream reuse review — 2026-08-28

This document separates upstream facts from local product decisions. A linked
project is not automatically installed, copied or treated as production-safe.

| Source | Authority / license note | What is reusable | Current decision | Rollback |
|---|---|---|---|---|
| [MiniMax H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3) | MiniMax official; H3 weights use the MiniMax H3 Community License | Official `h3-prompt-writing` skill; `base-en` and `ref-en` prompt shapes; FL2VA/Ref2VA role limits | Already represented as a clean-room, source-locked compiler contract. Keep official six-section Ref2VA order, English structural prose, explicit Picture roles and hash audit. Do not vendor guide text. | Switch compiler contract version; retain old graph snapshots for audit. |
| [MiniMax CLI H3 video guide](https://github.com/MiniMax-AI/cli/blob/main/skill/h3-video/references/h3-video.md) | MiniMax official repository; verify repository license before copying | Ordered output/subject/timeline/scene/camera/style/sound/constraint plan; task-id persistence and bounded polling | Adopt as validation requirements: exact time windows, explicit initial/final state and no duplicate remote submission after timeout. | Disable validator version without changing stored job evidence. |
| [ComfyUI server routes](https://docs.comfy.org/development/comfyui-server/comms_routes) | ComfyUI official docs | `/prompt`, `/queue`, `/history/{prompt_id}`, `/ws`, `/interrupt`, `/free` | `/prompt`, queue/history recovery and `/free` are in use. `/ws` is the next P1 improvement for lower-latency progress; SQLite remains the restart authority. | Fall back to bounded polling while keeping `prompt_id`. |
| [WanAnimate2ToVideo](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/WanAnimate2ToVideo/en.md) | Comfy-Org docs; model/node licenses apply separately | Reference-image identity, pose driving, continuation offset and separate pose/reference strength | Evaluation-only optional backend. It is not a MiniMax H3 upgrade and must not silently replace H3. Candidate for difficult actor-control shots after a separate visual A/B gate. | Select H3 backend; stored contracts remain backend-neutral. |
| [WanSCAILToVideo / SCAIL-2](https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/WanSCAILToVideo/en.md) | Comfy-Org docs; model license applies separately | `previous_frames`, `previous_frame_count`, `video_frame_offset`, 4n+1 frame contract | Reuse the durable segment-state design for future long-video adapters. Do not inject a Wan graph into the current H3 workflow. | Remove optional adapter; no episode schema loss. |
| [Filmclusive ComfyUI workflows](https://github.com/Filmclusive/ComfyUI-workflows) | Community project; verify every workflow/node/model license | ImageRef→VideoRef and character-replacement workflow examples | Compare graph topology only. No code/workflow import until license, node inventory, VRAM and 720p sample pass. | Delete isolated experimental profile. |
| [ComfyUI-Expert video pipeline](https://github.com/MCKRUZ/ComfyUI-Expert/blob/master/skills/comfyui-video-pipeline/SKILL.md) | Community guidance, not an H3 guarantee | FP8/SageAttention, RIFE, deflicker, CRF and low-denoise face-detail suggestions | Keep as opt-in experiment profiles. Never label them official or enable globally from a forum claim; benchmark on short local samples first. | Disable profile and return to locked baseline. |
| [2D character pipeline](https://github.com/mor-o/comfyui-2d-character-pipeline/blob/main/docs/workflow-2-video-gen.md) | Community project; model/node licenses vary | Low-cost keyframe iteration and VRAM residency lessons | Reuse only the proof-first operating idea. Current product keeps paired anchors where final-state control is required. | Continue current H3 proof/anchor path. |
| [MiniMax H3 prompt-skill index](https://github.com/r600a-code/minimax-h3-prompt-skill) | Community aggregation; contains upstream material, license boundaries require care | Examples and failure taxonomy for comparison | Do not copy bundled official text. Use only as a discovery index and verify every rule against MiniMax official sources. | Remove the index from research sources. |

## Product comparison

### Already stronger than a loose “one-click workflow”

- Contracts bind story, shot, reference order, prompt hash, artifact hash,
  decoded-visual hash and selected edit window.
- Proof is explicitly non-deliverable and requires human approval.
- Rejected artifacts are archived and excluded from composition.
- The final MP4/ZIP path fails closed on content QA, per-shot review, episode
  release, codec/stream/duration checks and ZIP re-open validation.

### Remaining gaps

1. ComfyUI progress should consume official WebSocket events while retaining
   SQLite/history reconciliation after reconnect.
2. Long-video adapters need durable `segment_index`, `previous_frames_sha256`
   and `video_frame_offset`; an old segment must never be blindly resubmitted.
3. Optional Wan/SCAIL/FramePack/RIFE profiles need isolated installation,
   license review and repeatable 720p A/B acceptance before product exposure.
4. Platform publishing remains manual. A codec-compatible file does not imply
   platform content approval, traffic or revenue.

## Research update rule

Every future entry must record `source_url`, `checked_at`, `license`,
`what_reused`, `compatibility`, `decision` and `rollback`. Prefer official
repositories/docs; community claims are hypotheses until reproduced on the
target machine. Never upload `.env`, keys, local databases, generated projects,
user media, model weights or machine-specific paths to a public repository.

