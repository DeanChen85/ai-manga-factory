# ComfyUI 节点包兼容性矩阵

**最后更新**：2026-08-30  
**维护者**：ai-manga-factory 团队  
**用途**：克隆仓库后运行 `python pipeline/comfy_preflight.py` 自动校验；手动查阅此表了解各包安装约束。

---

## 核心节点包

| 节点包 | License | 最低 ComfyUI 版本 | 已知冲突 | 推荐安装方式 | 测试状态 |
|---|---|---|---|---|---|
| **ComfyUI_RH_MinMaxH3** | Apache-2.0 | 0.33.2 | 无 | Manager 搜索 "RH MinMax H3" | ✅ 已测 |
| **ComfyUI_MiniMaxH3_Director** | Apache-2.0 | 0.33.2 | 无 | git clone AIMixer/ComfyUI_MiniMaxH3_Director | ✅ 已测 |
| **ComfyUI-KJNodes** | GPL-3.0 | 0.32.0 | Sol-Attn Triton | Manager 搜索 "KJNodes" | ⚠️ 部分 |
| **comfyui-minimax-h3-audio-T8** | GPL-3.0 | 0.34.0 | RHMiniMaxH3DualSigmaSampler | git clone T8mars/comfyui-minimax-h3-audio-T8 | 待测 |

### 关键说明

1. **GPL-3.0 隔离**：KJNodes 和 T8 都是 GPL-3.0，已通过 `.gitignore` 第 58 行 `/custom_nodes/` 隔离。**不要将这两个包的源码复制到 `pipeline/` 或其他 Apache-2.0 目录**。
2. **T8 vs RH 双时钟采样器冲突**：如果同时安装了 T8 和 RH，**只启用其中一个的 DualClockSampler**。建议保留 RH 的（你当前在用），T8 的仅用于 AudioRefine / FlashVSR 等专属功能。
3. **ComfyUI 本体版本**：T8 要求 ≥ 0.34.0，RH/Director/KJNodes 要求 ≥ 0.33.2。**升级 ComfyUI 到 0.34.0+ 可兼容所有包**。
4. **Sol-Attn Triton**：KJNodes 的 `PathchSageAttentionKJ` 与独立安装的 Sol-Attn Triton 可能冲突。如果装了 KJNodes，**不要额外装 Sol-Attn Triton**（KJNodes 自带）。

---

## 模型文件清单

| 模型 | 放置目录 | 文件名 | 大小 | 来源 |
|---|---|---|---|---|
| H3 UNET (FL2VA) | `models/diffusion_models/` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 31.7 GB | huggingface.co/MiniMaxAI/MiniMax-H3 |
| H3 UNET (Ref2VA) | `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GB | 同上 |
| Qwen3-VL 32B CLIP | `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.3 GB | 同上 |
| Video VAE | `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` | 4.9 GB | 同上 |
| Audio VAE | `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | 同上 |
| Turbo LoRA (v4-600) | `models/loras/` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 0.7 GB | github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo |
| FastH3 VSA LoRA | `models/loras/FastH3-VSA/vsa-datafree/` | `adapter_model.safetensors` | ~0.3 GB | huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA |

> ⚠️ **模型权重不随本仓库分发**。请从上游链接自行下载并接受对应许可证。

---

## 预检命令

```powershell
# 启动 ComfyUI 后运行
python pipeline/comfy_preflight.py
```

输出示例：
```json
{
  "schema": "ai-manga-comfy-preflight/v1",
  "passed": true,
  "nodes": {"missing_required": [], "missing_recommended": ["MiniMaxH3AddGuide"]},
  "models": {"missing_required": [], "turbo_lora": "..."},
  "warnings": ["⚠️  comfyui-minimax-h3-audio-T8 not detected in ComfyUI"],
  "failures": []
}
```

- `"passed": true` → 可以开始生产
- `"failures"` 非空 → **必须先修复再启动 GPU 任务**
- `"warnings"` 非空 → 可选修复，不影响当前流程

---

## 更新记录

| 日期 | 变更 |
|---|---|
| 2026-08-30 | 新增 T8 节点包条目；增加双时钟采样器冲突说明；新增 FastH3 VSA LoRA |
| 2026-08-24 | 初始版本：RH + Director + KJNodes |

---

## 引用与来源

- ComfyUI_RH_MinMaxH3: https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3 (Apache-2.0)
- ComfyUI_MiniMaxH3_Director: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director (Apache-2.0)
- ComfyUI-KJNodes: https://github.com/kijai/ComfyUI-KJNodes (GPL-3.0)
- comfyui-minimax-h3-audio-T8: https://github.com/T8mars/comfyui-minimax-h3-audio-T8 (GPL-3.0)
- MiniMax H3 Models: https://huggingface.co/MiniMaxAI/MiniMax-H3 (MiniMax H3 Community License)
- FastH3 VSA LoRA: https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA (Apache-2.0)