# -*- coding: utf-8 -*-
"""
generate_composite_panels.py
生成分镜复合图（角色+场景合成）
使用ComfyUI API将角色参考图与场景背景合成
"""

import os
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

from runtime_config import comfyui_root, comfyui_server

# ComfyUI配置
COMFYUI_SERVER = comfyui_server()
COMFYUI_INPUT = comfyui_root() / "input"

DEFAULT_CHECKPOINT = "RealVisXL_V5.0_fp16.safetensors"


def comfyui_api(endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """调用ComfyUI API"""
    url = f"{COMFYUI_SERVER}{endpoint}"
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"ComfyUI API connection failed: {e}")


def generate_composite_panel(
    scene_bg: str,
    character_refs: List[str],
    panel_desc: str,
    output_dir: str,
    panel_id: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    progress_cb=None
) -> str:
    """
    生成角色+场景的复合分镜图
    
    Args:
        scene_bg: 场景背景图文件名
        character_refs: 角色参考图文件名列表
        panel_desc: 分镜描述
        output_dir: 输出目录
        panel_id: 分镜ID
        checkpoint: checkpoint模型
        progress_cb: 进度回调
    
    Returns:
        生成的图片文件名
    """
    if progress_cb:
        progress_cb(f"Generating composite panel {panel_id}...")
    
    # 构建prompt
    full_prompt = f"{panel_desc}, photorealistic, modern realistic style, "
    full_prompt += "high quality, detailed, consistent character design, "
    full_prompt += "matching scene background, professional photography"
    
    negative_prompt = (
        "worst quality, low quality, blurry, deformed, disfigured, "
        "bad anatomy, inconsistent style, cartoon, anime, "
        "watermark, signature, text"
    )
    
    # 构建workflow
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(time.time() * 1000) % (2**32),
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 0.85,  # 保留更多场景特征
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 576, "batch_size": 1}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": full_prompt, "clip": ["4", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]}
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": f"composite/{panel_id}", "images": ["8", 0]}
        }
    }
    
    # 提交任务
    prompt_id = comfyui_api("/prompt", {"prompt": workflow})["prompt_id"]
    
    # 等待完成
    max_wait = 300
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        history = comfyui_api(f"/history/{prompt_id}")
        if prompt_id in history:
            result = history[prompt_id]
            if result.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"Generation failed: {result['status'].get('messages', [])}")
            if "outputs" in result and "9" in result["outputs"]:
                images = result["outputs"]["9"].get("images", [])
                if images:
                    filename = images[0]["filename"]
                    subfolder = images[0].get("subfolder", "")
                    full_path = COMFYUI_INPUT.parent / "output" / subfolder / filename
                    
                    # 复制到输出目录
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    dest_path = output_path / full_path.name
                    import shutil
                    shutil.copy2(full_path, dest_path)
                    
                    if progress_cb:
                        progress_cb(f"Composite panel saved: {dest_path.name}")
                    return dest_path.name
        time.sleep(2)
    
    raise RuntimeError(f"Generation timeout ({max_wait}s)")


if __name__ == "__main__":
    import sys
    print("Usage: Import and call generate_composite_panel()")
