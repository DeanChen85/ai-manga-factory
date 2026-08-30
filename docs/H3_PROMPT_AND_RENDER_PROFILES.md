# H3 提示词与渲染档位

## 为什么有两个渲染档位

提示词、身份、blocking、相机错误应该在便宜的渲染上被发现，而不是在昂贵的最终渲染上。本项目用 proof 档位先把所有合同层面错误暴露出来；只有通过 proof → 人工晋升才能进入 production 档位。

| 档位 | 帧数 | 时长 | 像素 | Turbo 步数 | ref_image_size | 可交付 |
|---|---:|---:|---:|---:|---|---|
| **proof** | 124 | 5.1667s | 0.4 | 6 | `match` | ❌ 不可交付 |
| **production** | 243 | 10.125s | 0.9 | 8 | `max` | ✅ 仅晋升后 |

`megapixels × seconds × turbo_steps` 是相对算力代理（运行时无承诺）；锁定配置下 proof ≈ production 的 1/6。

## 官方契约要点

- 低像素跑得更快；原生画布短边 768；时间轴 `17k+5` @ 24 fps；`ref_image_size=max` 提升身份保真度但显著变慢。
- 提示词保留 MiniMax 发布的段顺序；台词逐字 + stable speaker ID；每个参考一职；cast 数固定；单主导动作/相机；**排除模型生成的交付字幕**。
- canonical character ID 编入与 `<Picture N>` 绑定的精确 `<Subject N>`；持久化 UTF-8 prompt hash + ordered reference bundle hash。

## H3 Ref2VA 关键限制（提示词设计必须遵守）

**H3 Ref2VA 只有一个正向多模态 prompt，没有 SD 那种 negative lane**。实测证明：`exit sign` / `green panel` / `pictogram` / `ceiling` / `door header` 等词即便每个前面加 `no`/`crop out` 也会**渗出**到画面。

因此本项目使用 `h3-runtime/v7-positive-only-frame-authority` 合约：**只说画面应该是什么，不说不要什么**。例如：

- ❌ 错：正向描述 + 末尾加 `no exit sign, no green panel`
- ✅ 对：frame 跨主题头部到收银台；可见物体只有人、纯玻璃台、收银台；室内面"uniformly dry, blank and unlettered"

graph snapshot 中同时保存合约版本 + 精确 prompt，以区分编译器改动 vs seed 重试。SD 角色/场景生成仍可用它们自己的 negative，规则**只**适用于单向 H3 Ref2VA。

## 场景板验证

如需控制场景布局（位置/柜台/货架），优先用 SD1.5 ControlNet + Anything V5 的 lineart 控制图。若已有 `RealVisXL_V5.0_fp16.safetensors`，走 RealVisXL 结构 pass + 0.62 denoise Animagine style pass；否则单 pass Animagine。custom scene generator 保留原 callable 契约。

## 速度 vs 质量

proof 用 `match` + 6 Turbo 步**快速发现失败**；production 用 `max` + 8 Turbo 步（4 步在大动作/音频上有损）。Sage Attention 仍为可选能力探测。**加速 ≠ 质量放行**——技术成功的 MP4 也必须过内容 QA + 人工批才能交付。
