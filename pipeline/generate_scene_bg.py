# -*- coding: utf-8 -*-
"""
generate_scene_bg.py
场景背景图生成脚本
使用ComfyUI API和RealVisXL模型生成写实风格的场景背景
"""

import os
import json
import time
import uuid
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

from runtime_config import comfyui_root, comfyui_server

# ComfyUI配置
COMFYUI_SERVER = comfyui_server()
COMFYUI_INPUT = comfyui_root() / "input"
COMFYUI_OUTPUT = comfyui_root() / "output"

# 默认模型
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


def check_comfyui_ready() -> bool:
    """检查ComfyUI是否就绪"""
    try:
        comfyui_api("/system_stats")
        return True
    except:
        return False


def generate_scene_background(
    scene_desc: str,
    style: str = "",
    output_dir: Optional[str] = None,
    scene_id: Optional[str] = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    progress_cb=None
) -> str:
    """
    生成场景背景图
    
    Args:
        scene_desc: 场景描述
        style: 风格前缀
        output_dir: 输出目录（可选，默认保存到ComfyUI input）
        scene_id: 场景ID（可选）
        checkpoint: 使用的checkpoint模型
        progress_cb: 进度回调函数
    
    Returns:
        生成的图片文件名
    """
    if not check_comfyui_ready():
        raise RuntimeError("ComfyUI not running. Please start ComfyUI first.")
    
    if progress_cb:
        progress_cb("Generating scene background...")
    
    # 构建完整prompt
    full_prompt = f"{style}{scene_desc}, high quality, detailed, photorealistic"
    
    # 负面prompt
    negative_prompt = (
        "worst quality, low quality, blurry, deformed, disfigured, "
        "bad anatomy, bad hands, extra fingers, missing fingers, "
        "multiple views, watermark, signature, text, cropped, "
        "people, person, character, human, face, body"
    )
    
    # 构建ComfyUI workflow
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(time.time() * 1000) % (2**32),
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 576,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": full_prompt,
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"scene/{scene_id or 'scene'}",
                "images": ["8", 0]
            }
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
                    full_path = COMFYUI_OUTPUT / subfolder / filename
                    
                    # 如果指定了输出目录，复制过去
                    if output_dir:
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        dest_path = output_path / full_path.name
                        import shutil
                        shutil.copy2(full_path, dest_path)
                        
                        if progress_cb:
                            progress_cb(f"Scene background saved: {dest_path.name}")
                        
                        return dest_path.name
                    else:
                        # 复制到ComfyUI input目录
                        dest_filename = f"scene_{scene_id or 'bg'}_{uuid.uuid4().hex[:8]}.png"
                        dest_path = COMFYUI_INPUT / dest_filename
                        import shutil
                        shutil.copy2(full_path, dest_path)
                        
                        if progress_cb:
                            progress_cb(f"Scene background saved: {dest_filename}")
                        
                        return dest_filename
        
        time.sleep(2)
    
    raise RuntimeError(f"Generation timeout ({max_wait}s)")


if __name__ == "__main__":
    # 测试代码
    import sys
    
    test_desc = "modern office server room, small 3 square meter room, compact space, afternoon lighting"
    test_style = "modern realistic style, office environment, soft lighting, "
    
    print(f"Scene description: {test_desc}")
    print(f"Style: {test_style}")
    print()
    
    try:
        scene_file = generate_scene_background(
            test_desc,
            test_style,
            scene_id="test_scene",
            progress_cb=lambda msg: print(f"  {msg}")
        )
        print(f"\n✓ Scene background: {scene_file}")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}", file=sys.stderr)
        sys.exit(1)
