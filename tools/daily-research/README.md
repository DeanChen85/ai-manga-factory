# 每日竞品调研自动化

每天 09:00 与 21:00 自动扫描 GitHub 与 Reddit 上的 AI 视频/短剧/H3 相关新项目，
对比 `ai-manga-factory` 现有能力，生成研究报告并推送到公开仓库。

## 文件

| 文件 | 用途 |
|---|---|
| `daily_research.py` | 主脚本：扫描 + 对比 + 写报告 + 推送 |
| `config.json` | 关键词、订阅列表、回溯天数 |
| `install_task.ps1` | 一次性创建两个 Windows 定时任务 |
| `last-state.json` | 已见项目去重状态（自动维护） |
| `outputs/<date>-research.md` | 每次扫描的研报 |
| `RUNS.md` | 累积的所有运行记录 |
| `logs/<task>.log` | Task Scheduler 输出 |

## 安装

在 PowerShell（管理员）里执行：

```powershell
.\install_task.ps1
```

会创建：
- `DSH-DailyResearch-09` — 每天 09:00
- `DSH-DailyResearch-21` — 每天 21:00

## 手动跑

```powershell
& "C:\Users\Dean\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
    "F:\new ai factory\tools\daily-research\daily_research.py"
```

或带 `--dry-run` 只读扫描、不推送：

```powershell
python tools/daily-research/daily_research.py --dry-run
```

## 凭证

脚本通过 `git credential-manager` 读取 `host=github.com` 的 PAT。
请保证 `C:\Users\Dean\.gitconfig` 配了 `credential.helper = manager`
（默认已配），并在该凭证里有 `repo` 权限。

## 推送目标

`DeanChen85/ai-manga-factory` 的 `main` 分支的 `docs/research/daily/` 与
`docs/research/DAILY-RUNS.md`。每个发现都会作为独立 commit 推上去。

## 自定义

改 `config.json`：
- `githubKeywords`：新增/修改搜索关键词
- `redditSubs`：新增/修改订阅
- `hnQueries`：Hacker News 搜索词
- `githubDaysBack`：回溯天数（默认 14）
- `minStarsForAlert`：star 阈值（小于它不进研报）
- `feishuWebhook`：飞书机器人 webhook，**每次有发现时自动发卡片到群**；留空则不发

## 飞书通知

每次扫描有**新发现**时，自动向 `feishuWebhook` 推送一张交互式卡片，包含：
- 项目名 / 链接 / ⭐ / 语言
- 描述（GitHub）/ 标题（Reddit、HN）
- 最多 10 条/卡，超过会提示

**0 发现时不通知**（避免噪音）。

**安全提醒**：webhook URL 等同于群聊密码。它已经在公开 Git 仓库的 `config.json` 里——任何人都能拿到并往群里发消息。
- 想换 URL：编辑 `config.json` → 重跑脚本
- 想"完全私有"：把 webhook 移到环境变量 `$env:DSH_FEISHU_WEBHOOK`，脚本优先读 env，再读 config