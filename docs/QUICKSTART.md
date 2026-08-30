# 快速上手

## 1. 前置条件

- Windows 或 Linux，Python 3.11+
- NVIDIA GPU，已装好可用的 ComfyUI
- FFmpeg 和 ffprobe 在 `PATH` 中，或通过环境变量指定
- MiniMax API Key（用于剧本/分镜合同生成）
- MiniMax H3 模型文件：自行按上游许可证接受并安装

## 2. 安装主应用依赖

```powershell
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env` 并填入你的密钥。`MiniMax_API_KEY` 必须走进程环境或密钥管理器，**不要**写进仓库文件。国内默认使用 Anthropic 兼容协议 `https://api.minimaxi.com/anthropic/v1/messages`，模型 `MiniMax-M2.7`。

## 3. 启动 ComfyUI 并预检

安装好 H3 模型与自定义节点（参见 `docs/COMFYUI_H3_INSTALL.md`）。然后运行预检：

```powershell
python pipeline/comfy_preflight.py
```

此为只读检查，必须通过后再提交任何 GPU 任务。

## 4. 启动网页

```powershell
python -m streamlit run pipeline/web_app.py --server.port 8501
```

Windows 用户可直接运行仓库根目录的 `启动.bat`（含 Python 检查、依赖检查、ComfyUI 探活、端口占用拒绝）。打开 `http://127.0.0.1:8501`。

## 5. 用户旅程

1. 选择「单集 V3」或「整季 V4」，输入主题、梗概、风格、平台、语言与时长。
2. 系统以总编剧/分镜导演身份生成合同；无 Key 时不会悄悄替换故事；显式 DEMO 只适用于单集且禁止进入生产。
3. 整季 V4 先审核共享人物/声线、世界、场景、视觉圣经与精确 N 集连续大纲；逐集 `state_out` 必须等于下一集 `state_in`，换装/受伤/道具/时间跳跃只能作为显式事件。
4. 资产门禁全部通过后启动单集或整季；后台 worker 为每镜注入已批参考图；第 N 集必须等待第 N-1 集成功并继承其末片尾帧。
5. 默认先生成 5.167 秒、0.4MP、Turbo 6、`match` 的低成本预演。页面展示最终发给 H3 的提示词、导演 skill 版本、参照角色、prompt/reference 哈希与首中尾 QA 证据。
6. 人工确认动作、首尾状态与叙事信息后，系统以不可变晋级记录排入 10.125 秒、0.9MP、Turbo 8、`max` 的正式生产；预演永远不能合片或导出。
7. 页面每两秒刷新共享资产、逐集任务、图片和视频；可按资产、镜头、单集或整季重试/继续/取消，刷新页面或重启后从 SQLite 恢复；成功且输入未变化的项目不会重做。
8. 交付面板以 `fit` 或 `fill` 导出 9:16、16:9、1:1；ffprobe 通过后生成逐集 MP4、manifest、字幕和 ZIP，全部集成功后才允许生成 `.season.zip`。

## 6. 测试与发布预检

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

发布前另运行：

```powershell
python pipeline/release_preflight.py
```

这两条命令**不调付费 API，不启 GPU**，发布预检也不会打印密钥值。
