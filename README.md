# AI 漫剧工厂

本地生产台：既支持单集 V3，也支持从“主题 + 故事梗概 + 总集数 + 每集秒数”生成整季 V4。系统先锁定整季人物、声线、服装、世界、场景和连续状态，再通过确定性的 H3 导演编译器生成官方结构提示词，制作并人工批准共享参考图。视频默认先跑不可交付的低成本预演，绑定提示词/参照/产物哈希通过内容 QA 和人工验收后才晋级正式生产，最终导出逐集成片和整季包。

> 平台交付基线为竖屏 720×1280、横屏 1280×720（H.264/AAC）。技术生成成功不等于内容合格；真实 H3 原生语音仍属于生成式对齐，不承诺逐采样点口型同步。交付判定以测试、真实内容验收和发布预检三者为准。

新机器先阅读 [Quick start](docs/QUICKSTART.md)、[ComfyUI/H3 固定版本安装](docs/COMFYUI_H3_INSTALL.md)、[架构](docs/ARCHITECTURE.md)、[H3 提示词/渲染档位](docs/H3_PROMPT_AND_RENDER_PROFILES.md)、[节点包兼容性矩阵](docs/compatibility_matrix.md)、[外部项目复用评审](docs/research/EXTERNAL_PROJECTS_2026-08-28.md) 和 [更新记录](CHANGELOG.md)。

## 启动

1. 安装主应用依赖：

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. 启动配置好 H3 模型和自定义节点的 ComfyUI，默认地址为 `http://127.0.0.1:8188`。

   ```powershell
   python pipeline/comfy_preflight.py
   ```

   该只读检查必须通过后再提交付费或 GPU 任务；固定节点、模型文件名和上游 commit 见安装文档。

3. 设置 `MiniMax_API_KEY`，或在网页中临时输入。不要把真实密钥写入仓库文件。国内默认使用
   Anthropic 兼容协议 `https://api.minimaxi.com/anthropic/v1/messages` 与 `MiniMax-M2.7`；
   国际用户可显式设置 `MiniMax_BASE_URL=https://api.minimax.io/anthropic`。V3 两阶段通过
   强制工具调用提交结构化对象，不接受纯文本合同回退。

4. 启动网页：

   ```powershell
   python -m streamlit run pipeline/web_app.py --server.port 8501
   ```

   本机现有整合环境也可运行 `启动.bat`。打开 `http://127.0.0.1:8501`。

## Web 生产流程

1. 选择“单集 V3”或“整季 V4”，输入主题、故事梗概、风格、平台、语言和时长；整季模式还要求总集数及每集秒数。
2. MiniMax 以总编剧/分镜导演身份生成合同。无 MiniMax key 时不会悄悄替换故事；显式 DEMO 只适用于单集且禁止进入生产。
3. 整季 V4 先审核共享人物/声线、世界、场景、视觉圣经和 exact-N 连续大纲；逐集 `state_out` 必须等于下一集 `state_in`，换装/受伤/道具/时间跳跃只能作为显式事件。
4. 所有逐集 V3 合同生成并批准后，整季服务一次性注册精确 N 集。共享人物和场景资产只生成一次，并经人工批准后才开启视频门禁。
5. 后台 worker 为每镜按 `character_ids` / `scene_id` 注入批准参考图；第 N 集必须等待第 N-1 集成功，并继承其状态、末片和尾帧。
6. 默认先生成 5.167 秒、0.4MP、Turbo 6、`match` 的低成本预演。页面展示最终发给 H3 的提示词、导演 skill 版本、参照角色、prompt/reference 哈希和首中尾 QA 证据。
7. 人工确认动作、首尾状态和叙事信息后，系统以不可变晋级记录排入 10.125 秒、0.9MP、Turbo 8、`max` 的正式生产；预演永远不能合片或导出。
8. 页面每两秒刷新共享资产、逐集任务、图片和视频；可按资产、镜头、单集或整季重试/继续/取消，刷新页面或重启后从 SQLite 恢复，成功且输入未变化的项目不会重做。
9. 交付面板以安全 `fit` 或裁切 `fill` 导出 9:16、16:9、1:1；ffprobe 通过后生成逐集 MP4、manifest、字幕和 ZIP，全部集成功后才允许生成 `.season.zip`。

## 核心数据合同

```text
Creative Brief
  → Series V4 contract (exact N / exact seconds)
    → shared character / voice / world / visual / scene bible
      → season outline + continuity state chain
        → Episode V3 contracts
          → shared approved assets
            → non-deliverable proof jobs 1..N
              → hash-bound content QA + human promotion
                → formal render jobs 1..N
                  → validated episode MP4 + subtitles + delivery ZIP
                → complete season ZIP
```

口播、屏幕文字和声音提示是三条独立轨道：

```json
{
  "spoken_dialogue": [
    {"start_s": 0.5, "end_s": 2.5, "speaker_id": "char_hero", "text": "我们到了。"}
  ],
  "on_screen_text": [
    {"start_s": 3.0, "end_s": 4.0, "text": "午夜", "position": "top-safe"}
  ],
  "audio_cues": [
    {"start_s": 4.2, "end_s": 4.8, "cue_type": "sfx", "prompt": "metal door click"}
  ]
}
```

H3 会按 24fps 的合法帧网格生成原生音视频；提示词合同会检查越界、重叠、说话人和语速。需要确定性配音/字幕时，可复用保存的 cue sheet 接外部 TTS 和混音阶段。

## CLI 与服务入口

```powershell
# 状态（不触发 GPU）
python pipeline/orchestrator.py status <ep_id>

# 同步运行持久任务流程
python pipeline/orchestrator.py render <ep_id>

# 导出平台成片
python pipeline/orchestrator.py export <ep_id> --preset=vertical_9_16 --resize-mode=fit

# 后台 worker
python pipeline/worker.py --ep-id <ep_id>
```

网页只调用 `pipeline/render_service.py` 和 `pipeline/series_service.py` 的公共接口，不直接操作 SQLite、ComfyUI 或 FFmpeg。整季服务主要入口包括 `prepare_series_contract`、`register_series_contract_episodes`、`prepare_shared_assets`、`approve_shared_assets`、`start/resume/retry/cancel_series`、`status_series` 与 `export_season`。

## 输出结构

```text
output/projects/<ep_id>/
├── episode.json
├── charrefs/
│   ├── <character>_*.png
│   └── <character>.manifest.json
├── previews/
│   ├── <panel>.mp4
│   └── <panel>.graph.json
├── videos/
│   ├── <panel>.mp4
│   ├── <panel>.graph.json
│   ├── <panel>.cues.json
│   └── <panel>.artifact.json
└── exports/
    ├── <ep_id>_<preset>.mp4
    ├── <ep_id>_<preset>.manifest.json
    ├── <ep_id>_<preset>.vtt
    └── <ep_id>_<preset>.delivery.zip
```

运行状态默认存放于 `state/render_jobs.sqlite3`，worker 日志位于 `logs/workers/`。

## 环境变量

参考 `.env.example`。主要变量：

| 变量 | 用途 |
|---|---|
| `MiniMax_API_KEY` | 实时故事拆分；不会回退读取 `OPENAI_API_KEY` |
| `MiniMax_PROTOCOL` / `MiniMax_BASE_URL` / `MiniMax_MODEL` | 默认 `anthropic` / `https://api.minimaxi.com/anthropic` / `MiniMax-M2.7`；旧 `/v1` base 会迁移到同域 Anthropic Messages。只有显式 `MiniMax_PROTOCOL=openai` 才使用旧 OpenAI 兼容协议（不支持 V3 强制工具合同） |
| `AI_MANGA_MINIMAX_TIMEOUT_SECONDS` | 单次同步请求超时，默认 180 秒，可配置 10-600 秒；系统不自动重试付费请求 |
| `AI_MANGA_ROOT` / `AI_FACTORY_ROOT` | 项目根目录 |
| `AI_MANGA_PROJECTS_DIR` | 项目产物目录 |
| `AI_MANGA_JOB_DB` | SQLite 任务库 |
| `COMFYUI_ROOT` / `COMFYUI_SERVER` | ComfyUI 根目录与 API 地址 |
| `FFMPEG_EXE` / `FFMPEG_PATH` | FFmpeg 可执行文件 |
| `FFPROBE_EXE` / `FFPROBE_PATH` | ffprobe 可执行文件 |

## 测试

测试不调用付费 API、不启动 GPU：

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

测试总数由 CI 运行结果作为唯一事实来源，不在文档手填易过期数字。覆盖官方结构 H3 提示词、预演晋级、哈希审计、显式 DEMO、密钥边界、V4 exact-N/时长/state-chain、共享资产哈希、跨集尾帧依赖、SQLite 重启恢复、失败续跑、H3 参考图节点、画幅/Sage、音画 cue、FFmpeg 命令、ffprobe 门禁、逐集 ZIP、整季 ZIP 与 GitHub 发布预检。

发布前另运行：

```powershell
python pipeline/release_preflight.py
```

该命令不会调用付费 API 或 GPU，也不会打印密钥值。

## 安全与已知限制

- 本地两处历史明文 MiniMax key 已清空并替换为环境变量说明；旧 key 仍必须由所有者在 MiniMax 后台吊销/轮换。
- `api.minimax.chat` 与 `abab*` 旧配置仅为显式兼容；网页会提示 deprecated。生产默认走
  Anthropic Messages 的 `max_tokens` 与强制 `tool_use.input`；`stop_reason=max_tokens` 会硬失败。
  OpenAI 2048 completion 路径只保留为显式 legacy 兼容，不作为 V3 结构化生产默认。
- `models/`、ComfyUI 与自定义节点不随应用代码发行；克隆者必须按文档安装，并分别接受第三方及 MiniMax H3 模型许可证。
- 当前本机 ComfyUI 0.33.2 能运行 Ref2VA，但未暴露上游新加入的 `MiniMaxH3AddGuide`。因此当前版本把参考图如实定义为身份/场景/构图参考，不冒充任意时间点硬关键帧；夜间预检会提示该推荐能力缺失。
- 原生 H3 音频的口型/台词对齐是概率性的。当前改造修正了字段混用、时间范围和语速问题，并保存可用于确定性后期的 cue sheet，但高要求项目仍建议外部 TTS + 字幕 + 混音。
- 第三轮 Animagine 真实参考图测试虽解决了错误性别和多人拼图，但服装分层、标志性眉疤及两个场景的关键语义仍未全部通过人工视觉验收；资产门禁已正确拒绝，真实整季 H3 不会在错误素材上继续。量产前仍需用最终题材通过共享资产人工审批。
- 平台规则会变化；导出预设是兼容母版，不代表平台内容审核或账号发布保证。

第三方来源、许可证边界和固定版本见 `THIRD_PARTY_NOTICES.md` 与 `skills/minimax-h3-drama-director/sources.lock.json`。
