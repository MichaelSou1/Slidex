# Slidex

> 面向真实交付的 Agentic PowerPoint 生成系统：从资料研究、内容组织、视觉设计，到逐页检查、受控修复与最终 PPTX 渲染校验。

Slidex 的当前主运行时位于 `deeppresenter/`。它不是一次性让模型“吐出一份 PPT”，而是提供一套完整的 Agent Harness：把模型、工具、工作区、子 Agent、浏览器观测、质量 Critic、修复轨迹和导出门禁连接成可执行、可追踪、可验证的生成流程。

```text
用户 Prompt / 附件
        │
        ▼
  Planner（可选，支持人工确认与修改大纲）
        │
        ▼
  Research Agent ── 检索 / 抓取 / 文档解析 / 素材整理
        │
        ▼
  Design Agent ─── HTML 幻灯片 / 并行子 Agent / 浏览器预览
        │
        ▼
  Slidex Harness ─ 声明 IR + 浏览器计算 IR + 混合 Critic
        │                         │
        │                         └── RepairAction → 确定性修复 → 重新检查
        ▼
  Deck Gate ────── 跨页一致性与硬缺陷门禁
        │
        ▼
  HTML → PPTX → LibreOffice 重渲染 → 像素/感知/文本保真校验
        │
        ▼
  可交付 PPTX + 完整检查报告、轨迹与 Manifest
```

## 为什么是 Slidex

传统 PPT 生成链路通常止步于“生成了一个文件”。Slidex 更关注这个文件是否真的可以交付：

- **研究与设计分离**：Research Agent 先构建有来源、有叙事的 Markdown 文稿，Design Agent 再负责视觉表达。
- **工具原生 Agent**：Agent 可主动检索网页、抓取正文、解析附件、处理图片、读写文件、执行命令并调用专用模型工具。
- **页面不是黑盒**：系统同时保留 HTML 声明结构、浏览器实际布局和渲染图，检查结果能定位到具体元素。
- **修复必须闭环**：修复动作绑定检查报告和 artifact 版本；任何源码修改都必须重新检查，不能沿用过期结论。
- **导出不是终点**：PPTX 会再次渲染，并与 HTML 基准对比；未通过保真校验的文件不会被标记为成功。
- **全过程可审计**：输入、对话、工具日志、页面观测、缺陷报告、修复轨迹和导出 Manifest 都保存在独立工作区。

## Agent 能力

### 1. 可交互规划

通过 `--planner` 启用 Planner Agent。系统先生成结构化大纲，在正式研究前允许用户：

- 查看页级标题、目标与关键内容；
- 直接确认大纲；
- 用自然语言要求 Planner 修改；
- 将确认后的大纲作为 Research Agent 的硬约束。

### 2. 深度研究与内容策划

Research Agent 面向“信息美学”组织内容，而不是简单堆叠搜索摘要：

- 从宽到窄进行主题检索，并抓取网页正文；
- 优先使用官方来源、论文、权威媒体等可验证信息；
- 将 PDF、DOCX、图片等附件转换为 Markdown，并提取可复用图片；
- 下载网页文件和图片到本地，避免最终页面依赖远程资源；
- 调用长上下文模型分析大型文本；
- 调用视觉模型理解图片，或调用文生图模型补充素材；
- 输出以 `---` 分页的 Markdown 文稿，并在结束前检查文稿质量。

### 3. 视觉设计与页面实现

Design Agent 将文稿转换为独立 HTML 页面：

- 建立全局视觉系统，包括色彩、字体、间距和组件语言；
- 支持 `16:9`、`4:3`、`A1`、`A2`、`A3`、`A4`；
- 用真实文本、矢量形状、图片与信息图表达内容；
- 通过浏览器渲染页面预览；
- 每页生成后调用 `inspect_slide`，根据结构化缺陷报告继续修复；
- 最终通过 `html2pptx` 生成可编辑的 PPTX。

### 4. 子 Agent 并行执行

Research 和 Design 内置 `delegate_subagent`，可将自包含任务委派给隔离的 SubAgent。`multiagent_mode` 还会启用面向长文档的模型工具与对应运行模式：

- 主 Agent 先把背景、约束、输入输出和 handoff 要求写入上下文文件；
- 每个 SubAgent 使用独立目录，避免中间文件互相污染；
- 研究阶段可并行处理长文档分片或多视角检索；
- 设计阶段可在统一视觉规范下并行生成多页 HTML；
- SubAgent 以文件作为交付物，再由主 Agent 汇总。

### 5. 长任务上下文管理

开启 `context_folding` 后，Agent 接近上下文上限时会：

1. 将已收集事实、生成产物、未决问题和下一步写入工作区摘要；
2. 压缩旧对话，只保留必要上下文和近期消息；
3. 从本地状态继续执行，而不是因上下文溢出中止。

## 工具体系

Slidex 通过 MCP 工具服务器与进程内本地工具组合能力。角色可在 YAML 中声明允许的服务器和工具，实现最小权限工具集。

| 工具组 | 主要能力 |
| --- | --- |
| `local` | 工作区内读取、写入、精确编辑、文件检索、执行命令、显式思考检查点、子任务委派 |
| `search` | Google/SerpAPI 或 Tavily 网页检索、图片检索、网页正文抓取、文件下载 |
| `any2markdown` | PDF、Office 文档和图片等转 Markdown；可接入 MinerU 解析 PDF |
| `tool_agents` | 图片描述、文生图、长文档总结等模型驱动工具 |
| `deeppresenter` | 页面渲染、文稿检查；运行时注入更严格的 `inspect_slide`、`apply_repair` 与预览工具 |
| `task` | 统一 `finalize` 协议，校验并返回 Agent 产物 |
| `pptagent` | 旧版模板驱动 PPT 生成能力，通过独立 MCP 服务保留 |

MCP 配置位于 `~/.config/deeppresenter/mcp.json`，可替换、关闭或扩展工具服务器。离线模式会移除需要网络的工具。

## Harness 亮点

### 工作区隔离与可恢复执行

每次任务默认创建在 `~/.cache/deeppresenter/<session_id>/`；也可通过 `DEEPPRESENTER_WORKSPACE_BASE` 修改根目录。附件先复制进工作区，文件工具拒绝路径逃逸，Agent 的所有副作用都被收敛到当前会话。

典型工作区包含：

```text
<workspace>/
├── attachments/                  # 输入附件副本
├── slides/                       # Design Agent 生成的 HTML/CSS
├── subagents/                    # 隔离的子 Agent 工作区
├── .input_request.json           # 归一化后的输入
├── .history/
│   ├── observations/             # 截图、声明 IR、浏览器计算 IR
│   ├── slidex/                   # 页面/整套检查、修复轨迹、导出 Manifest
│   └── *.jsonl / *.log           # Agent 历史与工具日志
└── intermediate_output.json      # 阶段产物索引
```

### 双 IR + 浏览器真实观测

`inspect_slide` 不只看截图。它会同时采集：

- **Declared IR**：从 HTML/CSS 提取作者声明的元素、层级、边界框、主题 token 与素材引用；
- **Computed IR**：通过 Playwright 获取浏览器实际计算后的几何、样式、资源错误与页面错误；
- **Render Artifact**：页面截图及调试 overlay；
- **Provenance**：源码哈希、渲染器版本、配置和 artifact 依赖关系。

这使“越界、重叠、错位、字体、调色板、密度、语义矛盾”等问题可以基于可定位证据判断，而不是只依赖模糊的整图审美评分。

### 路由式混合 Critic

`HybridCritic` 会按缺陷类型和现有证据选择检查器：

- 确定性几何与样式问题优先交给 symbolic inspector；
- 开放式语义问题才路由到独立 Critic/VLM；
- 必要时使用 clean reference，但受 `reference_policy` 控制；
- 检查器输出 `pass / fail / defer / error`、严重度、元素 ID、证据和修复提示；
- Deck Inspector 再检查跨页术语、叙事与整体一致性。

这种设计避免让大模型覆盖本可确定计算的问题，也避免符号规则对开放语义做过度断言。

### 受控修复，而非任意重写

Critic 产生结构化 `RepairAction`，确定性修复器只执行受支持的低风险操作。Harness 会校验：

- 修复是否针对最新 artifact；
- 修改是否超出建议的元素和属性范围；
- 修复后缺陷是否真正改善；
- 是否引入新的硬缺陷或策略违规；
- 是否超过 `max_repair_rounds`。

每次动作写入 `repair_actions.jsonl`，形成可回放的修复轨迹。

### 硬门禁与最终渲染保真

最终导出采用 fail-closed 策略：

1. 所有 `slide_*.html` 必须完成最新版本检查；
2. Deck 级硬缺陷必须通过；
3. HTML 经 Node `html2pptx` 转换为 PPTX；
4. PPTX 由 LibreOffice 重新渲染；
5. 逐页比较像素差异、感知相似度和文本保留率；
6. 只有状态达到 `pptx_render_validated` 才返回成功。

默认阈值与检查版本在 `config.yaml` 的 `slidex` 节点中冻结，导出结果记录在 `export_manifest.json`。

## 快速开始

### 环境要求

- Python `>= 3.11`
- macOS 或 Linux；Windows 请使用 WSL
- Node.js / npm
- Playwright Chromium
- Poppler
- LibreOffice（严格 PPTX 重渲染与保真校验需要）
- 至少一个 OpenAI-compatible 模型端点，或使用可选的 LiteLLM provider

### 安装

推荐使用 `uv`：

```bash
git clone git@github.com:MichaelSou1/Slidex.git
cd Slidex
uv sync
```

如需直接接入 LiteLLM 支持的 provider：

```bash
uv sync --extra litellm
```

### 初始化

```bash
uv run deeppresenter onboard
```

`onboard` 会检查并准备 Playwright、Node 依赖和 Poppler，引导配置模型及可选的 Tavily、SerpAPI、MinerU，并将配置保存到：

```text
~/.config/deeppresenter/config.yaml
~/.config/deeppresenter/mcp.json
```

模型 API Key 可以写为环境变量引用，例如：

```yaml
research_agent:
  base_url: "https://openrouter.ai/api/v1"
  model: "anthropic/claude-sonnet-4.5"
  api_key: "$SLIDEX_API_KEY"
  capabilities:
    text: true
    vision: true
    tools: true
    structured_output: true
```

完整配置模板见 `deeppresenter/config.yaml.example` 和 `deeppresenter/mcp.json.example`。

### 生成第一份 PPT

```bash
uv run deeppresenter generate \
  "为技术团队制作一份 8 页的 Agent Harness 设计复盘，强调可靠性、可观测性和修复闭环" \
  --output agent-harness.pptx \
  --pages 8 \
  --lang zh \
  --aspect 16:9
```

带附件和可交互大纲：

```bash
uv run deeppresenter generate \
  "基于附件制作一份研究汇报，保留关键数据与图表" \
  --file report.pdf \
  --file metrics.xlsx \
  --output research-review.pptx \
  --pages 10-12 \
  --lang zh \
  --planner
```

CLI 同时保留 `pptagent` 别名：

```bash
uv run pptagent generate "制作一份产品发布会演示" -o launch.pptx -l zh
```

## CLI 命令

| 命令 | 作用 |
| --- | --- |
| `deeppresenter onboard` | 检查依赖并初始化模型、MCP 配置 |
| `deeppresenter generate` | 从 Prompt 和附件生成 PPTX |
| `deeppresenter serve` | 启动或检查本地模型服务 |
| `deeppresenter config` | 显示当前模型与配置路径 |
| `deeppresenter clean` | 删除用户配置和运行缓存 |

`generate` 常用参数：

| 参数 | 说明 |
| --- | --- |
| `--output, -o` | 输出 `.pptx` 路径，必填 |
| `--file, -f` | 附件，可重复传入 |
| `--pages, -p` | 页数或范围，如 `8`、`8-12` |
| `--aspect, -a` | `16:9`、`4:3`、`A1`、`A2`、`A3`、`A4` |
| `--lang, -l` | Agent 工作语言：`zh` 或 `en` |
| `--planner` | 在研究前生成并交互式确认大纲 |

## 配置关键项

```yaml
context_folding: true     # 长任务自动压缩上下文并写入状态摘要
multiagent_mode: false    # 启用长文档模型工具与多 Agent 运行模式
offline_mode: false       # 移除网络工具
async_tool_mode: false    # 慢工具异步执行
heavy_reflect: false      # 将渲染图加入设计反思上下文

slidex:
  max_repair_rounds: 3
  max_episode_steps: 20
  strict_export: true
  pptx_rerender: true
  export_max_pixel_difference: 0.12
  export_min_perceptual_similarity: 0.90
  export_min_text_presence: 0.95
```

模型按职责独立配置：

- `research_agent`：检索、分析、文稿与工具调用；
- `design_agent`：视觉设计、HTML 生成与修复；
- `long_context_model`：长文档总结与上下文压缩；
- `vision_model`：可选，图片理解；
- `t2i_model`：可选，文生图；
- `critic_model`：可选，独立视觉/结构化 Critic；
- `semantic_model`：可选，独立语义检查。

## 项目结构

```text
deeppresenter/
├── cli/                  # Typer CLI：onboard / generate / serve / config / clean
├── agents/               # Planner、Research、Design、PPTAgent、SubAgent 与 AgentEnv
├── roles/                # 中英文角色 Prompt、模型选择和工具权限
├── tools/                # MCP 工具服务器
├── slidex/               # Artifact、浏览器观测、Critic、路由、修复、奖励、导出门禁
├── html2pptx/            # Node HTML → PPTX 转换器
├── utils/                # 模型配置、日志、Web 转换、MCP 客户端等
├── main.py               # 主 Agent Loop 与运行时工具注入
└── test/                 # Harness、浏览器、导出与集成测试

pptagent/                 # 旧版模板驱动生成与评测库，以及 pptagent-mcp 入口
pyproject.toml            # 包元数据、依赖和 CLI 入口
```

当前产品主路径是 `deeppresenter`。`pptagent/` 仍用于兼容旧版模板生成与 `pptagent-mcp`，两条路径不会自动同步。

## 测试

运行不依赖外部模型的核心测试：

```bash
uv run pytest -m unit
```

运行 `deeppresenter` 测试：

```bash
uv run pytest deeppresenter/test
```

按环境选择集成测试：

```bash
uv run pytest -m browser   # 需要 Playwright / Chromium
uv run pytest -m export    # 需要 Node、Poppler 或 LibreOffice
uv run pytest -m llm       # 需要模型凭据
uv run pytest -m parse     # 需要 MinerU
```

## 已知边界

- Windows 原生环境不支持，请使用 WSL。
- 浏览器、Node 与导出依赖是主链路的一部分，不只是开发依赖。
- 搜索、在线 PDF 解析、视觉模型和文生图能力取决于对应 API 配置。
- `strict_export` 与 PPTX 重渲染依赖 LibreOffice；缺少能力时系统会明确失败，而不是返回未经验证的文件。
- 模型必须显式声明 `text`、`vision`、`tools`、`structured_output` 能力；角色使用工具时，模型需要支持 tool calling。

## License

本项目基于 [MIT License](LICENSE) 开源。
