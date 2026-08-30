# AI 漫剧工厂 · 可发布短视频生产标准

版本：2026-08-13 P0

## 1. 产品目标

输入主题、故事梗概、集数和每集时长后，系统生成整季圣经、分集大纲、人物/场景/服装/道具资产、逐集剧本、分镜、对白、声音与字幕。资产人工批准后可夜间串行生成；每个镜头必须经过技术 QA、自动内容 QA、人工逐镜批准和整集发布批准，才允许导出平台包。

“模型生成完了”只是素材完成，不等于镜头完成；“六镜有 MP4”不等于短剧完成。

## 2. 平台硬要求与披露

### 抖音

- 默认竖屏 9:16、720×1280；横屏 16:9、1280×720；H.264 High/yuv420p、AAC 48kHz stereo；字幕必须位于UI安全区。
- 对可能造成公众混淆的生成式AI非真实音视频显著标识。
- 禁止谣言、冒充侵权、垃圾信息和过度营销；上传前独立完成音乐、字体、模型、参考图和声音的商业权利检查。
- 平台没有公开一个适用于所有题材的“3秒完播率达到X就推荐”保证线；只能用账号自己的数据做A/B基线。

### TikTok

- TikTok Creative Codes：9:16、720p以上、UI安全区、sound-on，采用 hook→body→close，开头立即进入行动。
- 第一批内容使用统一720p母版输出：竖屏720×1280、横屏1280×720；标题/字幕不能贴底、贴右侧互动区。
- 官方资料把最初2–3秒作为关键创意窗口，但这是创作指导，不是保证流量的算法阈值。
- Creator Rewards（具体地区资格需另查）要求原创、高质量且时长超过1分钟；循环视频、单/多照片或纯文字叠层不属于合格原创内容。静态图换字幕不能作为收益路线。

### YouTube Shorts

- 输出竖屏母版并保留独立字幕/封面/元数据。
- 原创、非重复、非批量模板化是变现硬前提；不同对白覆盖同一画面属于高度重复风险。
- 写实且可能误导的合成内容必须在上传流程披露；动画类通常不强制，但版权、广告友好和原创性仍适用。

### B站

- 标记自制必须具备明确独创表达；单纯补帧、调色、倍速或轻微加工不构成自制。
- 可能误导的深度合成非真实信息需显著标识。
- B站用户习惯与抖音不同，竖屏短版作为引流版；有条件时另导横屏/合集版，而不是机械上传同一个母版。

## 3. 叙事结构

每集必须具备以下可验证事件，而不是只更换对白：

1. Hook：0–2秒出现冲突、异常、强问题、结果前置或视觉反差。
2. Setup：快速说明人物当前目标和阻碍。
3. Escalation：至少一次动作升级或信息变化。
4. Reversal/Payoff：至少一次反转、兑现或代价。
5. Cliffhanger/Close：系列剧留下下一集问题；单集给出结果和明确情绪收束。

每个事件要对应镜头和画面证据。LLM不能生成“大家开始游戏→一起上→赢了”这种只有口号、没有具体事件和可视动作的脚本并通过创作门。

## 4. 镜头生产策略

### 不再使用“6个10秒镜头=60秒短剧”

10秒是模型素材长度，不是最终剪辑镜头长度。内部生产基线：

- 最终有效镜头通常 1.5–4.0 秒；情绪停顿/建立镜头可以更长，但必须有理由。
- 30秒成片通常需要约8–14个有效镜头；60秒通常需要约14–24个有效镜头。该范围是工厂内部实验基线，不是平台官方规则。
- 每个10.125秒生成素材允许自动选择合格区间，未选区间不进入最终成片。
- 相邻镜头必须至少在景别、机位、主体动作、信息或情绪上发生一项有意义变化。

### 控制人物复杂度

- 动态动作/对白镜头默认只显示1–2名关键人物。
- 三人以上群像主要用于1–2秒建立镜、结尾合照或短反应，不要求模型在10秒内同时维持五人复杂动作。
- 多人场景拆成：群像建立 → 说话人近景 → 对手反应 → 手/手机/道具特写 → 双人关系镜 → 群像结果。
- 角色一致性来自项目级人物DNA、多视图资产、服装版本、逐镜视觉锚；不是把所有人物参考图同时塞给每个镜头。

### 对白与声音

- H3的原生语音只作为生成式氛围，不承诺逐字、逐角色、逐帧口型同步。
- 可发布对白使用按角色固定voice profile的TTS或授权语音，并以时间线为权威；需要说话人表演时单独走lip-sync步骤。
- 字幕只来自批准对白，最终交付烧录一次；模型画面提示禁止可见文字。
- BGM、环境音、SFX分轨，台词出现时自动ducking；不得使用无商业授权素材。

## 5. 四层质量状态

### A. Technical render

文件存在，时长/尺寸/fps/codec/音频流正确。状态文案只能是“技术生成完成”。

### B. Automated content QA

逐镜至少检查：

- 静态率、相邻帧变化、首尾变化、主体运动。
- 视觉流decoded fingerprint；同集exact duplicate硬失败，near duplicate进入人工或失败。
- 人物身份、人数、服装、场景、关键道具。
- OCR随机文字、字幕安全区、黑帧/花屏/严重形变。
- 合同要求的action、first→last state和镜头变化是否有首中尾证据。
- 前镜tail→本镜head的连续性。

任一硬项失败不得标内容通过。最多定向重生2次，之后进入dead letter或人工处理；不能无限烧卡。

### C. Editorial review

网页并排显示：合同目标、参考资产、0%/50%/100%帧、短预览、自动QA指标。人工批准必须绑定当前`artifact_sha256 + contract_hash + qa_report_hash`；产物或合同改变自动撤销。

### D. Release approval

整集联系表检查：hook、节奏、重复镜、剧情完整、字幕声音、结尾、平台安全区、AI披露和商业授权。逐镜全部批准且整集批准后才允许导出。历史manifest缺`release_status=approved`一律不可发布。

## 6. 夜间自动化

1. 晚上只提交故事/资产/分镜已经人工批准的项目。
2. 单GPU视频并发1；CPU/LLM预处理可有限并发。
3. 任务有lease、heartbeat、prompt_id恢复、指数退避、错误分类和dead letter。
4. 每镜：生成候选 → 技术QA → 内容QA → 合格则checkpoint；失败按原因定向重生，最多2次。
5. 成功镜不重复；上游资产变化只失效依赖镜。
6. 健康预检：Comfy、磁盘、VRAM、温度、模型/节点hash、许可证清单。
7. 到停止时间或资源门限后不再开新镜；完成当前安全提交并写晨报。
8. 早晨只需处理内容QA失败、near duplicate、dead letter和人工审批；未经发布批准不自动上传。

发布接口必须经过平台官方 OAuth、应用审核和用户授权。在此之前保持“人工上传”，系统不得通过群控、模拟点击或非官方接口自动发布。

## 7. 增长实验

没有系统能保证引流或收入。首轮用3个题材×3–5集建立账号自己的数据基线，记录：

- 2秒/3秒/5秒留存
- 平均观看时长和完播率
- 重播率
- 点赞、评论、分享、收藏
- 关注转化
- 下一集点击/主页访问
- 每集GPU时长、人工分钟数、合格镜率、重生率

每个题材至少准备2种hook和2种标题/封面；一次只改变一个变量。停止低留存题材，把资源投入“内容有效且生产成本可控”的系列。

## 8. 当前项目判定

`ep_1786340037` 六镜decoded video完全相同，只更换字幕与配音：技术封装通过、内容QA失败、0/6镜可接受、release revoked、禁止发布。它必须成为未来回归用的负样本：任何版本再次允许该项目导出，都视为P0回归失败。

## 9. 一手资料

- TikTok Creative Codes: https://ads.tiktok.com/business/en-US/creative-codes
- TikTok Creative Tips: https://ads.tiktok.com/business/creativecenter/quicktok/online/5_creative_tips/pc/en
- TikTok In-feed technical specifications: https://ads.tiktok.com/help/article/tiktok-auction-in-feed-ads?lang=en-GB
- TikTok AI-generated content: https://support.tiktok.com/en/using-tiktok/creating-videos/ai-generated-content
- TikTok Creator Rewards originality: https://support.tiktok.com/en/business-and-creator/creator-rewards-program/how-is-the-creator-rewards-program-different-from-the-tiktok-creator-fund
- YouTube Shorts monetization: https://support.google.com/youtube/answer/12504220
- YouTube channel monetization / repetitive content: https://support.google.com/youtube/answer/1311392
- YouTube AI disclosure: https://support.google.com/youtube/answer/14328491
- YouTube recommended encoding: https://support.google.com/youtube/answer/1722171
- YouTube Shorts analytics: https://support.google.com/youtube/answer/12942217
- 抖音用户协议（AI内容标识）: https://www.douyin.com/agreements/?id=6773906068725565448
- 抖音视频上传: https://open.douyin.com/platform/resource/docs/openapi/video-management/douyin/create/upload/
- 抖音发布解决方案: https://open.douyin.com/platform/resource/docs/ability/content-management/douyin-publish-solution/
- 抖音视频数据: https://open.douyin.com/platform/resource/docs/openapi/video-management/douyin/search-video/video-data/
- B站投稿规范: https://member.bilibili.com/studio/convention/content?index=3-1&navhide=1
- B站用户协议（深度合成标识）: https://www.bilibili.com/blackboard/user-rule-linux.html
- Wind Comic: https://github.com/ChrisChen667788/wind-comic
- Jellyfish: https://github.com/Forget-C/Jellyfish
- LocalMiniDrama: https://github.com/xuanyustudio/LocalMiniDrama
- ViMax: https://github.com/HKUDS/ViMax
- LumenX: https://github.com/alibaba/lumenx
- ArcReel: https://github.com/ArcReel/ArcReel
