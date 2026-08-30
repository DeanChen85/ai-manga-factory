# ComfyUI + MiniMax H3 可复现安装

本应用仓库**不重分发任何模型权重**。下载前请接受每个上游模型页面上显示的许可证。

## 1. 固定来源

| 组件 | Git 来源 / 提交 | 许可证 |
|---|---|---|
| ComfyUI | `bab6ee5f274c5231bfc072daf045104838e7147b` | 上游 |
| ComfyUI-MiniMax-H3-Turbo | `4274783a23afcfdbea3b4876cb79effd6c510785` | Apache-2.0 |
| ComfyUI-KJNodes | `3f20054214fec9f9234fd3841ae6f1e4287948f6` | MIT |
| MiniMax-H3 模型 | huggingface.co/MiniMaxAI/MiniMax-H3 | MiniMax H3 Community License |

## 2. 模型文件（本项目图契约的一部分）

把以下文件放进各自的 ComfyUI 模型目录（项目运行时会强校验确切文件名）：

| 路径 | 文件名 |
|---|---|
| `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` |
| `models/loras/` | `minimax_h3_turbo_v4_step600_ema.safetensors` |

## 3. 自定义节点

按上游文档分别安装并固定版本。本项目用到的：
- `ComfyUI_RH_MinMaxH3`（H3 原生运行时）
- `ComfyUI_MiniMaxH3_Director`（多段时间轴导演台，可选）

## 4. 预检

装好后运行：

```powershell
python pipeline/comfy_preflight.py
```

预检会核验：live `/object_info` 中有必需节点（`MiniMaxH3ReferenceToVideo` / `MiniMaxH3TurboLoRA` / `MiniMaxH3TurboSampler` / `PathchSageAttentionKJ` 等）、上述模型文件名确实存在、FFmpeg/ffprobe 可用。`MiniMaxH3AddGuide` 仅作为推荐能力报告，缺它不视为失败。

## 5. 启动参数（RTX 3090 实测稳定基线）

```powershell
python main.py --listen 127.0.0.1 --port 8188 --preview-method auto `
  --use-sage-attention --disable-cuda-malloc --disable-async-offload
```

`--disable-async-offload` 是 2026-08-27 修复 `hostbuf_file_reader_read failed` 后必须保留的稳定参数。

## 6. 许可证边界

模型权重、字体、BGM、音效、参考图与第三方代码各自的许可证、商业可用条件、地域限制由运营方独立确认。代码 Apache-2.0 不代表模型权重与生成结果自动拥有商业权利。
