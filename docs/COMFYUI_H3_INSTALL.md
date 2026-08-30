# ComfyUI + MiniMax H3 reproducible installation

The application repository never redistributes model weights. Accept the
license shown on each upstream model page before downloading anything.

## Pinned runtime sources

Use an existing ComfyUI installation or clone the following revisions. The
pins are compatibility baselines, not an instruction to overwrite a working
production machine during a render.

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git external/ComfyUI
git -C external/ComfyUI checkout bab6ee5f274c5231bfc072daf045104838e7147b

git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git external/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo
git -C external/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo checkout 4274783a23afcfdbea3b4876cb79effd6c510785

git clone https://github.com/kijai/ComfyUI-KJNodes.git external/ComfyUI/custom_nodes/ComfyUI-KJNodes
git -C external/ComfyUI/custom_nodes/ComfyUI-KJNodes checkout 3f20054214fec9f9234fd3841ae6f1e4287948f6
```

Install each repository's declared Python requirements in the same Python
environment that launches ComfyUI. Do not mix a system Python with a portable
ComfyUI Python.

## Required model files

Install the MiniMax H3 files under the standard ComfyUI model directories.
The filenames are part of this product's graph contract:

| ComfyUI directory | Required filename |
|---|---|
| `models/diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` |
| `models/loras/` | `minimax_h3_turbo_v4_step600_ema.safetensors` |

The official model card is
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). The Turbo
repository documents its LoRA download. Record each downloaded file's SHA-256
in your own deployment inventory because mirrors and quantizations may differ.

## Fail-closed verification

Start ComfyUI, set `COMFYUI_ROOT` and `COMFYUI_SERVER`, then run:

```powershell
python pipeline/comfy_preflight.py
```

The command checks the live `/object_info` node set, the exact model filenames,
the preferred Turbo LoRA, and FFmpeg/ffprobe. Missing required capabilities
produce a non-zero exit code before any paid or GPU job is submitted.

Required live nodes include `MiniMaxH3ReferenceToVideo`,
`MiniMaxH3TurboLoRA`, `MiniMaxH3TurboSampler`, `PathchSageAttentionKJ`, the
standard loaders, and `SaveVideo`. `MiniMaxH3AddGuide` is reported as a
recommended capability; the current product does not pretend ordinary Ref2VA
images are arbitrary-timestamp hard guides when that node is absent.

