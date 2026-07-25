# Slidex 完整工程改造 Roadmap

## 0. 文档目标

本路线图用于将当前 PPTAgent / DeepPresenter 代码库改造成 **Slidex**：一个面向幻灯片生成、诊断、修复和 agentic RL 的 source-aware agent 系统。

改造必须同时完成三项主线：

1. **彻底切断 Docker 运行时依赖**，允许直接侵入源码、检查中间状态并在本机运行工具。
2. **落地 `slide-examiner.pdf` 的核心方法**：可信原生 IR、失败归因、确定性 symbolic linter、原子化神经检查、reference-assisted comparison，以及可定位、可修复、可评分的 hybrid critic。
3. **接入 OpenAI-compatible 模型 API**，让 generation policy、critic 和未来 agentic RL policy 都能通过 `base_url`、`model`、`api_key` 调用外部兼容服务；Slidex 本身不对外实现 OpenAI API server。

本文中的复选框是工程执行清单。每个阶段必须满足退出条件后再进入下一阶段，避免同时重构生成、评估、模型客户端和训练接口而失去可验证基线。**执行顺序固定为：可交付 agent → critic/reward 与 E2E 评测 → RL environment/训练 → RL 独立评测；Phase 13 是进入 RL 的硬门禁。**

---

## 1. 总体设计原则

### 1.1 产品边界

- [ ] 将 `deeppresenter/` 确认为 Slidex 的主运行时和主要改造对象。
- [ ] 将 `pptagent/` 定义为 legacy template-generation backend；除兼容层和必要 bug fix 外，不在第一轮重写。
- [ ] 将 `pptagent/ppteval/` 定义为 legacy evaluator；新 hybrid critic 不依赖其宽泛 whole-rubric 评分流程。
- [ ] 保留 HTML → PPTX 作为主要生成链路，因为 HTML DOM 和浏览器 computed layout 可提供可信源结构。
- [ ] 将最终导出的 PPTX 重新渲染并检查，不能仅以 HTML preview 作为最终质量真值。

### 1.2 方法原则

- [ ] 检查器必须按证据类型分工，禁止将所有缺陷放进一次 VLM whole-rubric 调用。
- [ ] 可信 native IR 可判定的规则优先使用确定性 checker。
- [ ] render-only 问题使用单一缺陷、要求证据和定位的 atomic VLM query。
- [ ] 单视图证据不足时返回 `defer` 或请求 clean reference，不得悄悄换用低可信检查器。
- [ ] semantic defect 交给直接神经检查，但必须保留结构化 verdict、证据和原始输出。
- [ ] critic router 第一版必须是手工指定、冻结、带版本号的规则表，不使用 learned router。
- [ ] reward 必须保留分量，不允许只写一个不可解释的总分。
- [ ] 所有训练和评测样本必须验证 mutation 在最终像素中真实存在，避免 template snapping 导致零信号标签。

### 1.3 工程原则

- [ ] CLI、生成流程和 RL environment 复用同一领域服务，禁止复制多套业务流程。
- [ ] 每个 artifact、critic 配置、router 配置和 reward 配置都必须可版本化、可哈希、可回放。
- [ ] 默认 strict validation；忽略错误的 soft mode 只能显式开启，并写入 trajectory。
- [ ] 所有新函数和方法添加类型标注；技术注释和 docstring 使用英文。
- [ ] 优先使用现有 `AgentEnv.register_tool()`、Playwright 和 Pydantic，不引入不必要框架。

---

## 2. 目标架构

```text
CLI / Python RL Environment
             |
      Application Service
             |
 OpenAI-compatible model clients
                     |
       +-------------+-------------+
       |                           |
Generation Workflow          Slidex Environment
Research -> Design           reset -> step -> reward
       |                           |
       +-------------+-------------+
                     |
                 Slide IR
        declared / computed / render
                     |
             Failure Attribution
                     |
              Frozen Router
       +------+------+------+------+
       |             |             |
 Symbolic linter  Atomic VLM  Reference/Semantic
       +-------------+-------------+
                     |
           Inspection + Repair Hint
                     |
             Reward + Trajectory
                     |
         HTML -> PPTX -> Re-render
```

### 2.1 建议的代码布局

- [ ] 新建 `deeppresenter/slidex/` 作为领域核心，不把新逻辑继续堆进 `tools/reflect.py`。
- [ ] 新建 `deeppresenter/slidex/models.py`：IR、defect、inspection、reward、trajectory schema。
- [ ] 新建 `deeppresenter/slidex/artifacts.py`：artifact 路径、哈希、manifest 和 lineage。
- [ ] 新建 `deeppresenter/slidex/browser.py`：Playwright DOM/computed-style/geometry extraction。
- [ ] 新建 `deeppresenter/slidex/router.py`：冻结的 defect → inspector 映射。
- [ ] 新建 `deeppresenter/slidex/critic.py`：hybrid critic orchestration。
- [ ] 新建 `deeppresenter/slidex/reward.py`：reward vector、hard gates 和 aggregation。
- [ ] 新建 `deeppresenter/slidex/environment.py`：RL reset/step/observe/terminate。
- [ ] 新建 `deeppresenter/slidex/inspectors/`：geometry、style、terminology、render、neural、reference inspector。
- [ ] 新建 `deeppresenter/tools/filesystem.py`：替代 Docker sandbox 的本地工作区工具。
- [ ] 保持 `deeppresenter/main.py` 为 generation workflow facade，逐步将底层能力下沉到 application service。

---

# Phase 0：建立基线与改造护栏

> 状态（2026-07-24）：核心测试门禁和单页真实生成基线已建立；完整的三类 fixture 与 history 格式归档仍待补齐。

## 0.1 固定当前行为

- [x] 记录当前 `pptagent generate` 的最小可运行调用和输出目录结构。
  - 最小调用：`pptagent generate "<prompt>" --output output.pptx --pages 1 --aspect 16:9 --lang en`。
  - workspace 包含输入请求、Markdown、HTML、PPTX、PDF、渲染图、`intermediate_output.json` 和 `.history/`。
- [ ] 选取一个 3 页纯文本样例、一个含图片样例、一个故意 overflow 的 HTML 样例作为 smoke fixtures。
- [x] 记录当前单页 HTML、PDF、PPTX 输出及 `.history/` 目录结构；完整 history schema 固化留待 Artifact Store 阶段。
- [x] 明确集成依赖：strict/export 需要 Node，browser/PDF 需要 Playwright 与 Poppler，PPTX 重渲染需要 LibreOffice，真实 `generate` 需要 LLM/VLM 凭证。
- [x] 将测试分成 `unit`、`browser`、`export`、`llm`、`api`、`rl` 六类。

## 0.2 建立基础测试门禁

- [x] 为 `DeepPresenterConfig` 加载增加无网络 unit test。
- [x] 为 `InputRequest.copy_to_workspace()` 增加文件和目录测试。
- [x] 为 `AgentEnv.register_tool()` 增加 sync/async local tool 测试。
- [x] 为 `convert_html_to_pptx()` 增加 strict validation smoke test。
- [x] 为 Playwright HTML → image/PDF 增加 browser smoke test。
- [x] 保存基线测试结果，后续每个 phase 都运行最窄相关集合。
  - 基线命令：`PYTHONPATH=. .venv/bin/pytest -q -m "unit or browser or export" deeppresenter/test`
  - 结果（2026-07-24）：`8 passed`；`ruff check` 通过。

### Phase 0 退出条件

- [x] 无模型凭证时可运行 unit tests。
- [x] 有浏览器依赖时可运行单页 HTML render smoke test。
- [x] 已有最小输入可以验证后续 Docker 移除、critic 和模型客户端改造没有破坏主链路。

---

# Phase 1：彻底切断 Docker 依赖

> 状态（2026-07-24）：DeepPresenter 主运行时已切断 Docker；已使用 `.env` 配置的真实 LLM/VLM 完成单页 `pptagent generate`、strict inspection、PPTX 导出及最终重渲染。legacy `pptagent/docker/` 不属于本阶段删除范围。

## 1.1 删除 Python 运行时依赖

- [x] 从 `pyproject.toml` 主 dependencies 删除 `docker>=7.1.0`。
- [x] 检查并更新 `uv.lock`，清理 Docker SDK package 和根项目依赖记录。
- [x] 从 `deeppresenter/agents/env.py` 删除 `import docker`、`DockerException` 和 `NotFound`。
- [x] 删除 `AgentEnv.__aenter__()` 中连接 Docker daemon、查找同名容器和强制退出的代码。
- [x] 删除只服务于 Docker volume mapping 的 `DEEPPRESENTER_HOST_WORKSPACE_BASE` 分支。
- [x] 保留 `WORKSPACE`、`WORKSPACE_ID`、`CONFIG_FILE` 等普通子进程环境变量。

## 1.2 用本地工具替代 sandbox MCP

- [x] 在 `deeppresenter/tools/filesystem.py` 实现 `read_file(path)`。
- [x] 实现 `write_file(path, content)`，自动创建父目录。
- [x] 实现 `edit_file(path, old, new)`，要求唯一匹配，避免模糊写入。
- [x] 实现 `list_files(path, pattern)`。
- [x] 实现 `search_files(query, path, glob)`，优先使用 `rg`。
- [x] 实现 `run_command(command, cwd, timeout)`，返回 exit code、stdout、stderr。
- [x] 实现统一的 workspace path resolver，拒绝路径逃逸到 workspace 外部；附件由 `InputRequest.copy_to_workspace()` 显式复制进入 workspace。
- [x] 限制命令工作目录在 workspace 内，不提供 Docker 式安全承诺，但保证路径和超时行为可预测。
- [x] 在 `AgentLoop` 创建 `AgentEnv` 后注册这些 local tools。
- [x] 确认 local tool 输出仍经过当前 cutoff、history 和 timing 记录。

## 1.3 修改角色工具配置

- [x] 将 `deeppresenter/roles/Design.yaml` 中 `include_tool_servers: [sandbox]` 改为显式 local filesystem tools。
- [x] 将 `deeppresenter/roles/PPTAgent.yaml` 中的 sandbox 依赖替换为 local tools。
- [x] 检查 `Research.yaml`、`Planner.yaml`、`SubAgent.yaml` 的 `include_tool_servers: all`，确保不会隐式依赖已删除的 sandbox。
- [x] 保证 `delegate_subagent`、`thinking`、`finalize` 等 local tools 仍可被显式加入。
- [x] 当某个 role 配置引用不存在的 server/tool 时，启动阶段给出明确配置错误。

## 1.4 清理 MCP 默认配置

- [x] 从 `deeppresenter/mcp.json.example` 删除 `sandbox` Docker server 条目。
- [x] 检查 onboarding 生成的 MCP 配置不会再次加入 sandbox。
- [x] 保留 `any2markdown`、`task`、`deeppresenter`、`pptagent`、`tool_agents`、`search` 等 stdio MCP。
- [ ] 将新 critic 尽量做成进程内 service；只有确有跨进程复用需求时才暴露 MCP wrapper。（后续 critic phase 的架构约束，不阻塞 Phase 1 验收。）

## 1.5 清理 onboarding 和平台依赖

- [x] 从 `deeppresenter/cli/commands.py:onboard()` 删除 `check_docker_image()`。
- [x] 从 `deeppresenter/cli/dependency.py` 删除 Docker 安装、镜像构建和检查函数。
- [x] 删除 CLI 中相关 imports 和提示文案。
- [x] 保留并验证 Node/npm、Playwright Chromium、Poppler 检查。
- [x] Linux 本地模型服务不再通过 `deeppresenter/serve.sh` 的 Docker/SGLang 方案启动。
- [x] 将 Linux 本地模型服务改为显式外部 OpenAI-compatible endpoint，或使用与 macOS 一致的 `llama-server` 可执行文件。
- [x] 将 `serve` 命令与“启动 Slidex API”区分；当前 CLI 明确标注其启动的是 local model service。

## 1.6 清理仓库级 Docker 文件

- [x] 在代码不再引用后删除或停止分发 `deeppresenter/docker/`。
- [x] 删除或停止使用根目录 `docker-compose.yml`。
- [x] 删除或停止使用 `.dockerignore`。
- [x] 检查 `MANIFEST.in` 和 package data 不再包含 Docker 资源。
- [x] 全仓运行 `rg -n "docker|Docker|sandbox container"`；剩余命中仅为 legacy `pptagent/docker/`、历史说明和 Chromium `--no-sandbox` 参数。

## 1.7 Docker-free 验证

- [x] 配置加载及 unit tests 不再探测 Docker CLI 或 daemon。
- [x] 启动 `AgentEnv` 并验证 local/non-Docker tool server 注册。
- [x] 使用 local tools 和真实 LLM 生成一页 HTML。
- [x] 完成 HTML → PPTX，并分别完成 HTML 与最终 PPTX 的 PDF/image 渲染验证。
- [x] 使用 `.env` 中的 LLM/VLM API 运行 `pptagent generate` 兼容命令。
- [x] 确认测试和运行日志中不存在 Docker probe。
  - 真实验证产物：1 页 16:9 PPTX（约 593 KB），LibreOffice 重渲染 PDF/JPG 成功，OOXML ZIP 完整性通过。
  - strict inspection 首次发现 HTML 规则错误，Design Agent 修改后复检通过。

### Phase 1 退出条件

- [x] `pyproject.toml` 和运行时代码不依赖 Docker SDK。
- [x] 默认 MCP 配置不执行 Docker。
- [x] 无 Docker 环境可完成至少一页生成、strict inspection、修复、导出和最终 PPTX 重渲染。
- [x] 所有 agent 文件操作通过可记录的 local tools 完成。

---

# Phase 2：建立 Slidex 领域模型与 Artifact Store

> 状态（2026-07-24）：版本化领域 schema、稳定元素 ID 约束、原子 Artifact Store、严格配置和无模型测试已完成。

## 2.1 定义版本化 schema

- [x] 定义 `DefectClass`，至少包含 G1–G7、S1–S6，并允许未来扩展。
- [x] 定义 `InspectionStatus = pass | fail | defer | not_applicable | error`。
- [x] 定义 `EvidenceSource = declared_ir | computed_ir | render | clean_reference | deck_text`。
- [x] 定义 `BoundingBox`，明确坐标系、单位、页面尺寸和序列化格式。
- [x] 定义 `SlideElement`：稳定 ID、tag/type、semantic role、text、bbox、style、parent/children。
- [x] 定义 `DeclaredSlideIR`：源文件声明的结构、容器、主题 token 和预期角色。
- [x] 定义 `ComputedSlideIR`：浏览器实际 bbox、computed style、scroll size、visibility、stacking 和 font fallback。
- [x] 定义 `RenderArtifact`：HTML render、PDF render、PPTX re-render 的路径、尺寸、哈希和 renderer 信息。
- [x] 定义 `SlideArtifact`，聚合 source、declared IR、computed IR、renders 和 provenance。
- [x] 定义 `InspectionResult`：class、status、severity、confidence、evidence、element IDs、repair hint、latency、cost、inspector version。
- [x] 定义 `InspectionReport`：单页结果列表、summary、router/taxonomy version。
- [x] 定义 `RewardBreakdown`：hard constraints、soft scores、cost penalty、aggregate 和 gating reason。
- [x] 定义 `TrajectoryStep` 和 `EpisodeManifest`。
- [x] 所有 mutable list/dict 使用 `default_factory`，清理现有 Pydantic mutable defaults。

## 2.2 稳定元素 ID

- [x] 规定生成 HTML 中每个可检查元素必须有 `data-slidex-id`。
- [x] Design prompt 要求 ID 在修订时保持稳定，不得每轮全部重编号。
- [x] 浏览器 extractor 对缺失 ID 的元素生成 deterministic fallback ID，并发出 warning。
- [x] ID 必须能跨 source HTML、DOM snapshot、render annotation、critic report 和 repair action 关联。
- [x] 对重复 ID、空 ID 和层级变化增加 validation。

## 2.3 Artifact Store

- [x] 为每次 episode 创建独立 workspace，不复用可变全局目录。
- [x] 采用 `artifacts/<artifact_id>/` 保存 source、IR、renders、inspection 和 reward。
- [x] artifact ID 使用内容哈希或 UUID + 内容哈希，避免仅靠文件名。
- [x] 保存 `manifest.json`：父 artifact、创建 action、模型、sampling 参数、工具调用和版本。
- [x] 对 HTML、CSS、图片、IR JSON、PNG、PDF、PPTX 分别计算 SHA-256。
- [x] 记录 renderer 名称和版本，如 Chromium、html2pptx、LibreOffice。
- [x] artifact 写入采用临时目录 + atomic rename，避免并发任务读到半成品。
- [x] 大文件不嵌入 trajectory JSONL，只记录 artifact URI 和 hash。
- [x] 增加 workspace/artifact 配额和清理策略，但不得在活跃 episode 中自动删除。

## 2.4 配置模型

- [x] 在 `DeepPresenterConfig` 中增加 `slidex` 子配置。
- [x] 增加 `taxonomy_version`、`router_version`、`reward_version`。
- [x] 增加安全边距、alignment tolerance、overlap tolerance、palette threshold。
- [x] 增加 max repair rounds、max episode steps、command timeout。
- [x] 增加 strict export、PPTX re-render 和 reference policy 开关。
- [x] 增加独立 `critic_model` 和可选 `semantic_model`，不能默认与 policy history 共享状态。
- [x] 为旧配置提供清晰迁移默认值；未知关键字段应报错而不是静默忽略。
- [x] 更新 `deeppresenter/config.yaml.example` 展示 OpenAI-compatible outbound endpoint 和 critic 配置。

## 2.5 Schema 测试

- [x] 为每个 Pydantic model 增加 round-trip JSON test。
- [x] 测试旧版本 manifest 的显式拒绝或 migration。
- [x] 测试坐标越界、负尺寸、重复 ID、未知 status。
- [x] 生成一份单页 artifact fixture 并验证所有 hash 可复算。

### Phase 2 退出条件

- [x] 一页 HTML 可以被表示为完整 `SlideArtifact`。
- [x] source、computed IR、render 和 provenance 可通过稳定 ID 关联。
- [x] schema 和 artifact manifest 可独立于 agent/LLM 测试。

---

# Phase 3：实现可信 Native IR 和浏览器观测

## 3.1 Source/declared IR 提取

- [x] 从 HTML 提取 `data-slidex-id`、semantic role、容器关系、文本和资源引用。
- [x] 从 `global.css` 和页面 style 提取设计 token：字体 scale、palette、safe area 和 grid hints。
- [x] 明确 declared IR 是生成管线拥有的源结构，不使用像素 layout detector 替代。
- [x] 对动态脚本、远程字体和交互依赖默认禁止或标记为不可复现。
- [x] 对没有 declared semantic role 的元素标记 `unknown`，不猜测为 ground truth。

## 3.2 Playwright computed IR

- [x] 重构 `PlaywrightConverter`，允许一次 page load 同时生成 DOM snapshot、截图和 PDF。
- [x] 获取每个元素的 `getBoundingClientRect()`。
- [x] 获取 `clientWidth/clientHeight`、`scrollWidth/scrollHeight`。
- [x] 获取 text node range bounding rects，支持文本真实占用范围检测。
- [x] 获取关键 computed styles：font family/size/weight、color、background、overflow、display、visibility、opacity、z-index、transform。
- [x] 获取实际使用字体/字体 fallback；复用 html2pptx 已有 CDP font detection 思路。
- [x] 获取图片 natural size、object-fit、clip 和 load status。
- [x] 获取元素可见区域和页面交集，识别完全/部分出界。
- [x] 等待 `document.fonts.ready`、图片加载和网络空闲后再采样。
- [x] 固定 viewport、device scale factor、locale、timezone 和 browser flags，减少 reward 漂移。
- [x] 将 JS console error、page error 和资源加载失败写入 artifact。

## 3.3 DOM 与 render 一致性

- [x] 在 screenshot 上支持绘制 debug overlay，标出 element ID 和 bbox。
- [x] 检查 CSS transform 后 bbox 与 source geometry 的差异。
- [x] 检查 clipping、pseudo-element、shadow 和 SVG 等无法完整表达在 declared IR 中的内容。
- [x] 记录每页 render readiness；未加载完成的页面不得进入 critic。
- [x] 将当前 `inspect_slide` 的临时目录行为改为 artifact store 管理，避免无法回放。

## 3.4 浏览器观测测试

- [x] fixture：正常文本框，验证 client/scroll 相等。
- [x] fixture：文本 overflow，验证 scroll dimension 超出。
- [x] fixture：hidden overflow，验证内容超出但像素被裁剪。
- [x] fixture：absolute positioned child 越过 container。
- [x] fixture：页面边界和 safe margin 违规。
- [x] fixture：transform、pseudo-element 和 SVG。
- [x] fixture：字体 fallback 导致换行变化。
- [x] 同一 fixture 连续运行多次，确认 IR 和 reward 输入稳定。

### Phase 3 退出条件

- [x] 单次浏览器加载可产出 computed IR、PNG 和 PDF。
- [x] G2/G6/G7 所需几何证据无需 VLM 即可取得。
- [x] 所有证据可定位到稳定 element ID。

---

# Phase 4：实现论文中的 Symbolic Linters

## 4.1 G2 元素重叠

- [x] 基于 computed/native bbox 计算 pairwise intersection。
- [x] 区分预期重叠与异常重叠：背景、装饰、容器子元素、overlay allowlist。
- [x] 使用面积比例和最小像素阈值过滤抗锯齿级误差。
- [x] 输出两个 element ID、intersection bbox、面积和严重度。
- [x] clean fixtures 必须零误报。

## 4.2 G3 alignment offset

- [x] 提取 left/right/center/top/bottom alignment candidates。
- [x] 对重复列、卡片和文本基线进行 clustering。
- [x] 使用配置化 tolerance，不将所有自由布局强制吸附到统一 grid。
- [x] 只有存在足够 sibling evidence 时才判定 fail，否则 `not_applicable` 或 `defer`。
- [x] 输出偏移方向、像素值、参考元素集合和 repair target。

## 4.3 G4 font-size inconsistency

- [x] 建立 deck typography scale，而不是单页孤立阈值。
- [x] 按 semantic role 比较 title/subtitle/body/caption/footer。
- [x] 识别同角色的异常字号、字体和 weight。
- [x] 对 intentional emphasis 使用 role/attribute allowlist。
- [x] 输出 expected scale、actual value 和相关页面。

## 4.4 G5 brand-color violation

- [x] 从 `global.css` / theme 配置读取可信 palette。
- [x] 将 CSS colors 规范化到统一色彩空间。
- [x] 使用 CIEDE2000，而不是 RGB 欧氏距离。
- [x] 分开处理文本色、背景色、边框色和装饰色。
- [x] 透明度合成后再判断最终可见颜色。
- [x] 输出最近 palette color、Delta E 和修改建议。

## 4.5 G6 margin violation

- [x] 按 aspect ratio 定义 safe area。
- [x] 检查可见 bbox，而非仅 CSS 声明值。
- [x] 背景 bleed 和装饰性 full-bleed 元素可显式豁免。
- [x] 输出越界边、距离和允许边界。
- [x] 对 poster 与 presentation 使用独立默认阈值。

## 4.6 S3 terminology inconsistency

- [x] 从 deck 全文建立 occurrence table：term、slide、element、context。
- [x] 先做确定性 normalization：大小写、全半角、连字符、复数和空白。
- [x] 使用近重复聚类发现 subtle variants，但保留可解释 edit evidence。
- [x] 允许用户提供 glossary 和 accepted aliases。
- [x] 只输出候选冲突，不自动把所有近义词视为错误。
- [x] 输出 canonical suggestion、所有 occurrences 和修改目标。

## 4.7 G1 与 G7 的工程化区分

- [x] G1 定义为 declared text/container 约束违规；在 Slidex 源结构中可判定的部分走 symbolic。
- [x] G7 定义为 declared container 合法、但 computed/rendered content 越界。
- [x] 使用 scroll metrics、text ranges 和 child union 检测 DOM-level G7。
- [x] 对 DOM 无法解释的 pixel-only anomaly 标记为 unresolved，交给 atomic render inspector。
- [x] 不因为 Slidex 可直接检测 DOM overflow 就删除论文中的 render-gap 概念；必须保留 export 后再次检查。

## 4.8 统一 linter 输出

- [x] 所有 linters 实现统一 `Inspector` protocol。
- [x] 所有结果必须使用 `InspectionResult`，禁止返回自由文本字典。
- [x] 每个 inspector 记录名称、版本、输入 hash 和耗时。
- [x] checker 抛出的内部异常转换为 `error`，不能伪装成 `pass`。
- [x] 无足够证据时返回 `defer/not_applicable`，不能用默认阈值硬判。

## 4.9 Linter 测试

- [x] 每个 class 至少准备 clean/defective matched pair。
- [x] 检查 defective 触发且 clean 静默，避免“全报错”获得高 recall。
- [x] 测试阈值边界和 magnitude sweep。
- [x] 测试真实自由布局，避免 checker 只适合合成网格。
- [x] 对 clean suite 设定零误报或明确可接受上限。

### Phase 4 退出条件

- [x] G2–G6 和 S3 可在无模型环境运行。
- [x] G1/G7 的 declared/computed/render 边界被显式记录。
- [x] 每个 fail 都能定位元素并给出机器可读 repair hint。

---

# Phase 5：实现 Atomic Neural Inspectors 与 Failure Attribution

> 状态（2026-07-24）：原子神经检查、reference 顺序控制、A/B/C failure attribution、fake provider 测试与真实 API smoke test 已完成。

## 5.1 独立 critic 模型配置

- [x] 增加 `critic_model`，要求显式声明 `is_multimodal`，不要只根据模型名称猜测。
- [x] 增加可选 `semantic_model` 处理 deck-level S2/S5。
- [x] critic 调用使用独立、无生成 history 的 request。
- [x] 固定 temperature、top_p、seed（服务支持时）和 response schema。
- [x] 保存 endpoint identifier、model、sampling 参数、usage、latency 和 raw response。
- [x] 对不支持 image、JSON schema 或 tool call 的 OpenAI-compatible provider 给出 capability error。

## 5.2 原子查询协议

- [x] 每次 neural call 只检查一个 defect class。
- [x] prompt 必须给出 operational definition，而不是抽象“是否美观”。
- [x] 要求 `verdict = pass | fail | defer`。
- [x] 要求 localization：element ID 或归一化 bbox。
- [x] 要求 evidence：可观察事实，不接受只给总体分数。
- [x] 要求 repair suggestion，但 repair suggestion 不参与 defect verdict。
- [x] 使用 Pydantic structured output；soft parsing 失败时返回 `error`。
- [x] 禁止在同一 prompt 中展示 ground-truth label 或 mutation metadata。

## 5.3 S1 title/body mismatch

- [x] 从 IR 提供 title 和 body text，同时提供 render 作为补充证据。
- [x] 原子询问正文是否与标题主题矛盾或明显不匹配。
- [x] 区分“不完整”与“矛盾”，避免泛化为内容质量评分。
- [x] 输出冲突文本片段和 element IDs。

## 5.4 S4 density violation

- [x] 先计算 symbolic statistics：字符数、占用面积、字号、空白率、元素数。
- [x] 明显超阈值情况可直接 deterministic fail。
- [x] 边界情况由 semantic examiner 判断信息量是否与页面作用匹配。
- [x] 区分 over-packed、under-packed 和 intentional minimal title slide。
- [x] 输出统计证据和语义理由。

## 5.5 S6 image/text contradiction

- [x] 原子检查单个 image-caption/claim pair，不一次检查整页所有语义关系。
- [x] 从 IR 提供周围 claim/caption，并提供 crop 或整页 render。
- [x] 单视图证据不足时返回 `defer`。
- [x] 有 clean/reference 时使用 pairwise comparison。
- [x] 增加顺序对调控制，记录 positional disagreement。

## 5.6 unresolved G7 render anomaly

- [x] 仅对 DOM symbolic checker 无法判定的 render anomaly 调用 atomic VLM。
- [x] query 明确询问“内容是否越过或被裁剪于指定容器”，并提供目标 bbox overlay。
- [x] 要求定位目标元素，禁止 broad visual-quality rubric。
- [x] DOM 已确定 fail 时不得重复调用 VLM 浪费预算。

## 5.7 Deck-level S2/S5

- [x] S2 narrative-order break 使用 deck outline、每页标题和摘要，不依赖逐页截图堆叠。
- [x] S5 missing logic section 根据任务/批准 outline 判断缺失步骤。
- [x] 明确这是 deck-level semantic inspector，结果关联相关 slide IDs。
- [x] 允许任务本身没有固定逻辑结构时返回 `not_applicable`。

## 5.8 Reference-assisted inspector

- [x] `InspectionContext` 支持 clean/reference artifact ID。
- [x] reference 必须与目标页在 aspect ratio、renderer 和页面角色上兼容。
- [x] pairwise prompt 支持 `left | right | tie | defer`。
- [x] 同一 pair 运行 AB/BA 两种顺序，并聚合为 order-controlled verdict。
- [x] clean-vs-clean 控制用于检测 forced-choice 偏差。
- [x] 无 reference 时返回 `defer: clean_reference_required`，不自动伪造 synthetic twin。

## 5.9 Failure attribution protocol

- [x] 实现论文中的观察条件：A=image、B=trusted structured IR、C=image+IR、可选 reference。
- [x] attribution 只用于开发/评测和未知 class 诊断，不要求生产每次都运行全部条件。
- [x] 记录 `image_sufficient`、`structure_rescued`、`format_suppressed`、`reference_assisted`、`unresolved`。
- [x] attribution 结论描述操作层面的证据需求，不声称揭示模型内部机制。
- [x] 为同一模型/同一 item 保存 whole-rubric 与 atomic query 对照。
- [x] 支持 repeated whole-rubric budget control，用于验证收益不是仅来自更多采样。

## 5.10 Neural inspector 测试

- [x] 使用 fake OpenAI-compatible server 测试请求 payload 和 structured response。
- [x] 测试 timeout、429、invalid JSON、missing verdict、provider 不支持 image。
- [x] 对真实模型测试放入 `llm` marker，不阻塞无凭证 CI。
- [x] 保存一组人工审核的 atomic-query fixtures，检查 prompt 版本变化。

### Phase 5 退出条件

- [x] 每个 neural inspector 都是单类、可定位、可 defer 的结构化调用。
- [x] reference-required 情况不会被错误判为 pass/fail。
- [x] failure attribution 可重现实验性 A/B/C 对照。

---

# Phase 6：冻结 Hybrid Critic Router

## 6.1 Router v1 映射

- [x] 定义默认 frozen mapping：G2/G3/G4/G5/G6 → symbolic native-IR inspectors。
- [x] S3 → terminology linter。
- [x] G7 → DOM render overflow；unresolved 时 atomic VLM。
- [x] S1 → atomic semantic inspector。
- [x] S4 → symbolic density statistics + semantic boundary inspector。
- [x] S2/S5 → deck-level semantic inspector。
- [x] S6 → atomic VLM；证据不足时 reference inspector 或 defer。
- [x] G1 → source/computed checker；无法从单页确定的情形进入 reference policy。

## 6.2 Router 行为约束

- [x] router 输入仅包含 defect class、available evidence、artifact trust level 和配置。
- [x] router 不读取模型输出后再 post-hoc 改路线以提高分数。
- [x] inspector `defer` 是终态之一；只有 frozen policy 明确允许时才进入下一 inspector。
- [x] router 输出记录选择原因和缺失证据。
- [x] router 配置保存为机器可读对象并计算 hash。
- [x] taxonomy/router 变更必须提升版本，旧 trajectory 保留旧版本解释。

## 6.3 Trust policy

- [x] `native_html` 和 Slidex 生成的 DOM 标为 trusted source IR。
- [x] 第三方 PPTX 提取的原生 XML 标为 partial trusted，并声明缺少哪些 bookkeeping。
- [x] 从 PNG/PDF layout detector 恢复的 boxes 标为 recovered/untrusted。
- [x] symbolic linter 对 recovered structure 不宣称 native-IR 保证。
- [x] open-world image-only 输入自动降级为 VLM-only，并在 report 中显式标注能力上限。

## 6.4 Critic 聚合

- [x] 汇总 per-class inspection results，不用平均值掩盖 hard failure。
- [x] summary 明确列出 fail、defer、error 和 not-applicable 数量。
- [x] 对 inspector 冲突保留双方结果，不直接覆盖。
- [x] 增加 deterministic priority 规则，但仅适用于 trusted native predicates。
- [x] critic report 写入 artifact store，并可由 CLI/Python environment 读取。

## 6.5 与 `inspect_slide` 集成

- [x] 将 `tools/reflect.py:inspect_slide` 改为调用 Slidex critic service。
- [x] 返回结构化 JSON 文本或 artifact URI，不再仅返回 image block。
- [x] 删除未启用 multimodal 时返回 `This slide is valid.` 的错误语义。
- [x] 无 critic model 时仍运行 symbolic inspectors；神经项返回明确 `defer/unavailable`。
- [x] 提供单独 `render_slide` 工具给 agent 请求视觉预览，避免检查与渲染职责混合。
- [x] Design Agent 收到 report 后必须逐项处理 hard failures，并在下一 revision 关联前一 artifact。

### Phase 6 退出条件

- [x] `inspect_slide` 在无 VLM 时仍能提供可信 symbolic report。
- [x] router 版本、路线和 defer 行为均可回放。
- [x] 当前“看图自省”已替换为论文式 hybrid critic，而不是在其旁边叠加另一个评分器。

---

# Phase 7：Repair Loop 与生成流程集成

> 状态（2026-07-24）：机器可读 repair action、显式 deterministic/policy repair、增量 deck gate、防 reward hacking 与生成闭环已完成。

## 7.1 机器可读 RepairAction

- [x] 定义 `RepairAction`：operation、target IDs、constraints、source inspection IDs。
- [x] 支持 `move_element`、`resize_container`、`reduce_text`、`change_font_size`、`replace_color`、`rename_term` 等操作类型。
- [x] repair hint 是建议，不直接修改 source；执行后必须重新 inspect。
- [x] 每个 action 保存 before/after artifact ID。
- [x] 不可执行的自由文本建议标记为 `policy_edit`。

## 7.2 Policy repair

- [x] 更新 Design prompt：先读结构化 report，再只修改被定位的元素。
- [x] 禁止为了通过 checker 删除核心内容或把元素隐藏。
- [x] 修订时保留稳定 element ID。
- [x] 每页设置 max repair rounds，超过后返回带 unresolved defects 的终态。
- [x] agent 不得在 inspector `error` 时假装页面通过。

## 7.3 Deterministic repair（可选工具）

- [x] 实现安全边距 clamp。
- [x] 实现 palette replacement。
- [x] 实现 terminology canonical replacement。
- [x] 对 alignment snap、font shrink 等可能改变设计意图的操作默认仅建议，不自动执行。
- [x] deterministic repair 必须作为显式 action 写入 trajectory，不能后台静默修改。

## 7.4 Deck-level final inspection

- [x] 所有单页通过后运行 S2/S3/S5 和 typography/palette deck consistency。
- [x] deck-level repair 必须指明受影响页面，避免全 deck 无差别重生成。
- [x] 修订某页后只重跑受影响的 page inspectors，加上必要 deck inspectors。
- [x] finalization 前若存在 hard fail，默认阻止导出；允许显式 override，并记录原因。

## 7.5 防 reward hacking 检查

- [x] 检查 opacity、visibility、off-screen positioning、zero-size 等隐藏内容行为。
- [x] 检查重要文本被转为不可解析图片以绕过 terminology/semantic checker。
- [x] 检查字体过小但无 overflow 的规避行为。
- [x] 检查将所有元素标记为 decorative/allow-overlap 的滥用。
- [x] 检查删除 manuscript 必需内容以降低 density。
- [x] 将这些行为列为 hard policy violations。

### Phase 7 退出条件

- [x] 生成 → inspect → localized repair → re-inspect 形成闭环。
- [x] 每次修订有明确父 artifact 和 defect delta。
- [x] hard failure 不能被 aesthetic/semantic 高分抵消。

---

# Phase 8：最终导出物验证与 Render Fidelity

> 状态（2026-07-24）：严格 HTML→PPTX、LibreOffice 最终重渲染、多信号 fidelity gate、mutation zero-signal 检测和可追溯 export manifest 已完成。

## 8.1 Strict HTML → PPTX

- [x] 将训练与常规生成默认 `soft_parsing` 改为 `False`。
- [x] validation error 转成明确 invalid artifact 和 hard penalty。
- [x] soft mode 只能由请求显式开启，并记录所有 ignored warnings。
- [x] 将 html2pptx stdout/stderr、版本和命令参数写入 export manifest。

## 8.2 PPTX 重新渲染

- [x] 确定首选 headless renderer（例如 LibreOffice）并检测版本。
- [x] 将生成的 PPTX 渲染为 PDF/PNG。
- [x] 每页输出稳定命名并关联 source slide ID。
- [x] renderer 缺失时返回 capability error，不把 HTML render 冒充 PPTX render。
- [ ] 可选支持 PowerPoint render worker，但不作为本地必需依赖。（可选扩展，不阻塞 Phase 8 验收。）

## 8.3 Render-fidelity gate

- [x] 比较 HTML screenshot 与 PPTX re-render 的页面尺寸和页数。
- [x] 计算像素差/感知差，但不把单一相似度当质量结论。
- [x] 对关键元素比较 OCR/text presence 或可用的导出结构。
- [x] 对 G1/G7、margin、missing image 重新检查最终 render。
- [x] 检测 PPTX 中字体替换、换行和元素位置变化。
- [x] render 差异超过阈值时标记 `export_fidelity_failure`。

## 8.4 Template snapping / mutation fidelity

- [x] 对所有注入式训练数据保存 clean 和 defective 最终 render。
- [x] 如果最终像素相同，则样本标记 `zero_signal` 并从 detection/reward 训练中排除。
- [x] 统计每类 mutation 的 render survival rate。
- [x] 标签来自最终可观察 artifact，而不是仅来自 IR mutation 操作。
- [x] clean twin 也必须经过同一 renderer 和版本。

## 8.5 最终 artifact 状态

- [x] 区分 `draft_html_valid`、`pptx_exported`、`pptx_render_validated`。
- [x] CLI/Python environment 默认只将 `pptx_render_validated` 标为成功终态。
- [x] 导出失败时保留 HTML/PDF 调试 artifact，但不得把 PDF fallback 宣称为 PPTX 成功。
- [x] `intermediate_output.json` 迁移到更明确的 artifact manifest，同时提供兼容字段。

### Phase 8 退出条件

- [x] reward 和成功状态基于最终交付物，而非仅基于 HTML 草稿。
- [x] template snapping/导出重排造成的标签失真可被自动检测。
- [x] 任一最终 PPTX 可以追溯到 source、critic report 和 renderer。

---

# Phase 9：Reward System

## 9.1 Reward vector

- [x] 定义 `validity_reward`：source、browser、export 和 re-render 是否成功。
- [x] 定义 `geometry_reward`：G1–G7 的 per-class pass/fail/severity。
- [x] 定义 `semantic_reward`：S1–S6 的结果，defer 与 error 单独记录。
- [x] 定义 `fidelity_reward`：HTML 与最终 PPTX render 一致性。
- [x] 定义 `task_reward`：页面数、outline、必需内容和用户约束。
- [x] 定义 `efficiency_reward`：token、模型调用、工具调用、repair steps 和 latency。
- [x] 定义 `policy_violation_penalty`：隐藏内容、路径逃逸、无效 action 等。

## 9.2 Hard-gated aggregation

- [x] invalid export 触发 terminal hard negative。
- [x] 存在严重 overflow、missing asset、页面出界时不发 aesthetic bonus。
- [x] defer 不等于 pass；聚合时保留 coverage。
- [x] inspector error 不直接算 defect miss，但触发 reliability penalty/episode invalidation。
- [x] 所有 hard gate 的阈值进入 `reward_version` 配置。
- [x] Python environment 同时返回 reward vector 和 aggregate scalar。

## 9.3 Repair delta reward

- [x] 计算修订前后 fail 数量和 severity delta。
- [x] 奖励解决目标 defect，同时惩罚引入新 defect。
- [x] 保留未变化、改善、恶化三类 per-class transition。
- [x] 避免仅因多次微小编辑累积无限正奖励；采用 potential-based 或终态奖励约束。
- [x] 对 deterministic tool repair 与 policy repair 分开统计。

## 9.4 Reward calibration

- [x] 使用 matched clean/defective pairs 校准 hard checker。
- [x] 报告 recall、specificity、balanced accuracy 和 localization，而非只报 accuracy。
- [x] 神经 checker 按模型/provider 分开校准。
- [x] 对 clean-vs-clean、AB/BA 和 repeated query 做偏差控制。
- [x] reward 配置冻结后再跑 held-out evaluation。
- [x] development 调参数据与最终评测数据严格分离。

## 9.5 Reward 输出与审计

- [x] 每个 reward component 引用产生它的 inspection result ID。
- [x] 保存聚合公式、权重和 gate reason。
- [x] 提供 `explain_reward()`，输出机器可读解释。
- [x] 支持离线根据相同 artifacts 重新计算 reward。
- [x] reward 重算不能重新调用不确定模型，除非显式开始新的 evaluation run。

### Phase 9 退出条件

- [x] 同一 artifact + 同一 critic/reward version 可得到可复算结果。
- [x] hard defect 无法被 soft score抵消。
- [x] reward 能用于单页 repair RL，也能汇总 deck generation。

---

# Phase 10：OpenAI-compatible 模型接入加固（Outbound Client）

> 本阶段只要求 Slidex **调用外部 OpenAI-compatible API**。不实现 `/v1/models`、`/v1/chat/completions`、SSE server、文件服务或对外 episode API。

## 10.1 统一模型客户端

- [x] 继续使用 `deeppresenter.utils.config.Endpoint` / `LLM` 作为 outbound client 主入口。
- [x] generation policy、critic、semantic examiner 和未来 RL policy 均使用统一 `base_url`、`model`、`api_key` 配置。
- [x] 明确 `provider: openai` 表示通过 OpenAI Python SDK 调用兼容 endpoint，而不是仅支持 OpenAI 官方服务。
- [x] 保留 `provider: litellm` 作为可选 provider adapter，不让核心流程依赖 LiteLLM。
- [x] 支持 endpoint-specific `client_kwargs` 与 `sampling_parameters`，并保存到 trajectory/provenance。
- [x] API key 支持从环境变量解析，配置、日志和 artifact manifest 中必须脱敏。
- [x] 不在 import/config load 阶段发起网络请求；只在显式 validate 或执行模型调用时连接。

## 10.2 Chat Completions 能力

- [x] 验证普通文本 chat completion。
- [x] 验证 multimodal `image_url` / data URL 输入，供 slide critic 使用。
- [x] 验证 tools、tool choice 和多 tool-call 响应，供 agent loop 使用。
- [x] 验证 structured output / `response_format`；provider 不支持时返回明确 capability error。
- [x] 兼容常见 OpenAI-compatible 响应差异，但禁止用宽泛异常捕获伪造成功结果。
- [x] 统一解析 usage、finish reason、reasoning 字段和 tool calls。
- [x] 保留 endpoint rotation/retry，但记录每次实际使用的 endpoint/model。

## 10.3 Provider 能力声明

- [x] 在配置中显式声明 text、vision、tools、structured output 等 capability，减少按模型名称猜测。
- [x] generation agent 启动前验证 tools capability。
- [x] critic 启动前验证 vision/structured-output capability。
- [x] capability 缺失时尽早报错；可选 inspector 则返回 `defer/unavailable`，不能假装 pass。
- [x] 不要求兼容 provider 实现 OpenAI API 的所有端点，只依赖项目实际使用的 Chat Completions 子集。

## 10.4 Agentic RL 接入方式

- [x] internal policy 通过同一 outbound client 调用训练或推理服务，例如 vLLM、SGLang、llama.cpp 或其他 OpenAI-compatible endpoint。
- [x] external policy 可直接使用 Python environment 的 observation/action 接口，不要求 Slidex 启动 HTTP server。
- [x] 每个 trajectory step 保存 policy endpoint identifier、model、sampling 参数、usage 和 response hash。
- [x] policy 与 critic 使用独立 endpoint/config/history，避免奖励模型与策略模型状态串扰。
- [x] 支持为 RL rollout 关闭客户端内部重试，避免一次 environment step 隐式对应多次 policy sample。

## 10.5 兼容性测试

- [x] 使用本地 fake OpenAI-compatible server 测试 request payload 和响应解析。
- [x] 测试文本、图片、tools、structured output、usage 和 reasoning 字段。
- [x] 测试 401、404、429、5xx、timeout、invalid JSON 和不完整 tool call。
- [x] 使用至少一个本地兼容服务进行 smoke test，不依赖 OpenAI 官方 endpoint。
- [x] 真实 provider tests 使用 `llm` marker，不阻塞无凭证 CI。

### Phase 10 退出条件

- [x] generation policy 和 critic 均可配置为任意满足所需能力的 OpenAI-compatible endpoint。
- [x] agentic RL internal policy 可复用同一客户端，且调用参数完整写入 trajectory。
- [x] 项目不包含或维护对外 OpenAI-compatible server。

---

---

# Phase 11：品牌、配置和兼容迁移

## 11.1 包与命令

- [ ] 将项目展示名称改为 Slidex。
- [ ] 在 `pyproject.toml` 增加 `slidex = "deeppresenter.cli:main"`。
- [ ] 暂时保留 `pptagent` 为兼容 alias，并输出 deprecation 提示策略。
- [ ] 保留 `pptagent-mcp` 仅服务 legacy backend；若新 MCP 需要入口，使用独立 `slidex-mcp`。
- [ ] 更新 package description、keywords 和默认 workspace 环境变量。

## 11.2 配置路径

- [ ] 引入 `SLIDEX_WORKSPACE_BASE`，兼容读取旧 `DEEPPRESENTER_WORKSPACE_BASE`。
- [ ] 引入 Slidex config directory，提供旧配置迁移而非静默丢失。
- [ ] 配置输出隐藏 API key。
- [ ] onboarding 默认生成 Docker-free MCP 配置。
- [ ] outbound policy/critic endpoints 均采用 OpenAI-compatible `base_url/model/api_key`。

## 11.3 日志与观测

- [ ] 日志名称从 DeepPresenter 迁移为 Slidex，同时兼容旧 history reader。
- [ ] 每个 request/episode/step/artifact 使用关联 ID。
- [ ] 记录 inspector 和 tool timing。
- [ ] 记录 LLM usage 和 estimated cost，但不将 provider 定价硬编码为 reward truth。
- [ ] 模型调用日志不输出完整附件、base64 图片或 secret。

## 11.4 Legacy compatibility

- [ ] legacy `ConvertType.PPTAGENT` 仍能走原 template backend。
- [ ] legacy backend 输出也可包装为 artifact，并运行有限的 final critic。
- [ ] 明确 legacy PPTX 缺少完整 HTML native IR 时的 trust downgrade。
- [ ] 不为追求统一而重写 `pptagent/` 全部内部模块。

### Phase 11 退出条件

- [ ] 新用户看到和调用的是 Slidex。
- [ ] 旧 CLI/配置有可控兼容路径。
- [ ] legacy backend 不阻塞 Slidex IR/critic/RL 主线。

---

---

# Phase 12：测试矩阵、性能与安全

## 12.1 Unit tests

- [ ] schema、hash、artifact lineage。
- [ ] geometry/style/terminology linters。
- [ ] router 和 trust policy。
- [ ] reward gates 和 delta reward。
- [ ] local filesystem tools 和 path traversal。
- [ ] OpenAI request/response serialization。
- [ ] episode state machine 和重复 action 防护。

## 12.2 Browser/export tests

- [ ] DOM extraction fixtures。
- [ ] browser determinism 重复测试。
- [ ] HTML → PPTX strict validation。
- [ ] PPTX re-render 和 render-fidelity。
- [ ] missing font/image、JS error、network timeout。
- [ ] 不同 aspect ratio。

## 12.3 LLM tests

- [ ] fake server 覆盖所有错误和 structured response。
- [ ] 少量真实 provider smoke tests，使用 marker 和环境变量。
- [ ] atomic vs whole-rubric 对照。
- [ ] AB/BA reference order control。
- [ ] defer/abstain 行为。

## 12.4 OpenAI-compatible Client Contract Tests

- [ ] 使用 fake server 验证 OpenAI Python SDK async client 请求。
- [ ] 验证文本、图片、tools 和 structured output payload。
- [ ] 验证 401、404、429、5xx、timeout 与 cancellation。
- [ ] 验证 unknown model、invalid response 和不完整 tool call。
- [ ] 验证同一 environment 不允许并发 step。

## 12.5 性能

- [ ] browser/context pooling，避免每个 inspector 重启 Chromium。
- [ ] 同一 artifact 的 IR/render/inspection 按 hash 缓存。
- [ ] symbolic inspectors 并行运行。
- [ ] neural calls 按 class 和 artifact hash 缓存。
- [ ] 限制并发模型调用、浏览器页面和导出进程。
- [ ] 记录 p50/p95 latency 和每 episode cost。
- [ ] 不通过跳过 hard inspection 来优化延迟。

## 12.6 本地源码执行安全

- [ ] 明确 Docker 移除后不是强隔离执行环境。
- [ ] 本地 `run_command` 仅供受信任工作区 agent 使用，不将其包装为对外网络接口。
- [ ] workspace path resolver 防止 `..`、symlink 和绝对路径逃逸。
- [ ] 命令 timeout、输出上限和进程组清理。
- [ ] 附件文件名清洗，压缩包防 zip-slip/zip-bomb。
- [ ] API key 不写入日志、trajectory 或 artifact manifest。
- [ ] 如未来需要不可信多租户执行，另接外部 sandbox runner；主项目不依赖 Docker。

## 12.7 CI 分层

- [ ] PR 默认运行 unit + OpenAI-compatible fake-server tests。
- [ ] browser tests 在具备 Chromium 的 job 运行。
- [ ] export tests 在具备 Node/Poppler/LibreOffice 的 job 运行。
- [ ] real LLM tests 手动或定时运行，不进入普通 PR 门禁。
- [ ] benchmark 和 frozen evaluation 独立运行并保存版本化结果。

### Phase 12 退出条件

- [ ] 无凭证 CI 可验证绝大多数核心逻辑。
- [ ] browser/export/LLM 失败可以区分依赖缺失与代码回归。
- [ ] Slidex 不对外暴露任意本地命令网络接口。

---

---

# Phase 13：论文级评测体系与真实场景验证

> 目标：在现有工程测试之外建立可发表、可复现、可审计的评测体系，同时测量 critic 的 intrinsic accuracy、真实分布迁移能力，以及 critic-in-the-loop 对最终 PPT 质量的因果收益。默认采用中等预算：100 个 sealed E2E tasks、3 个 seeds、三臂配对和单人专家盲评。

## 13.1 评测目标与预注册

- [ ] 定义三项研究问题：critic 检测是否准确、hybrid 路由是否优于 whole-rubric VLM、critic 是否改善最终 PPT。
- [ ] 将 primary endpoints 冻结为 macro balanced accuracy 和“无严重缺陷且通过导出门”的 deck 比例。
- [ ] 在 sealed test 前冻结 taxonomy、router、prompt、阈值、模型配置、统计方法和最小有意义效果。
- [ ] 将 confirmatory、secondary、exploratory 指标分开，禁止测试后更改主指标。
- [ ] 保存预注册配置 hash、Git commit、运行环境和冻结时间。

## 13.2 评测数据模型与目录

- [ ] 新建 `deeppresenter/eval/`，承载数据准备、注入、运行、汇总和统计逻辑，不污染运行时 critic。
- [ ] 定义 `BenchmarkManifest`：数据来源、许可证、revision、SHA-256、split、case ID 和父 deck ID。
- [ ] 定义 `EvaluationCase`：输入、clean reference、缺陷标签、严重度、定位和任务 brief。
- [ ] 定义 `EvaluationRun`：实验臂、模型、seed、配置 hash、artifact lineage、token、费用、延迟和错误。
- [ ] 定义 `EvaluationResult`：逐 case verdict、定位、修复结果、人工标签和指标。
- [ ] 大型数据默认存放于 `~/.cache/deeppresenter/eval`，仓库只保存小型 fixture、manifest 和脚本。
- [ ] 所有失败、超时、defer、error 和缺失 verdict 均进入结果，禁止只保存成功样本。

## 13.3 Zenodo10K 可控配对集

- [ ] 从 `Forceless/Zenodo10K` 获取保留原始 `.pptx` 的 CC 授权 deck。
- [ ] 固定 dataset revision、下载 URL、许可证、文件 hash 和获取时间。
- [ ] 按源 deck 和模板近重复去重，禁止同源 deck 或变体跨 split。
- [ ] 划分 20% development set 和 80% sealed test set。
- [ ] 在 PowerPoint XML/native IR 中注入 G1–G7 和 S1–S6 单一缺陷。
- [ ] 每个缺陷类准备至少 30 个 defective/clean pair，并配套等量 clean negative。
- [ ] clean 与 defective 使用完全相同的 LibreOffice/Playwright 环境渲染。
- [ ] 验证 defective 与 clean pixel diff 非零，防止 template snapping 吞掉缺陷。
- [ ] 验证目标规则成立，且非目标 inspector 未出现新的高严重度缺陷。
- [ ] 对语义注入进行逐例专家核对，对 geometry 注入随机复核至少 20%。
- [ ] 单独冻结 G1、G2、G3、G5、G6、G7、S1、S4、S6 九类 image arm，保持与论文结果可比。
- [ ] snapping 或渲染失败样本记为 dataset integrity failure，不记为模型漏检。

## 13.4 SlideAudit 外部分布集

- [ ] 获取 SlideAudit 的公开数据、标注和 taxonomy，固定发布版本。
- [ ] 在下载阶段硬校验许可证；不可再分发时只保存 URL、revision、hash 和本地缓存路径。
- [ ] 建立 SlideAudit taxonomy 与 G1–G7/S1–S6 的版本化 crosswalk。
- [ ] 一对多 taxonomy 映射使用 multi-label 评测，不强制转换成单标签。
- [ ] 对无 native IR 的图片显式运行 image-only 模式。
- [ ] 将 symbolic inspector 的不可用标记为 capability downgrade，而不是检测错误。
- [ ] 分别报告 synthetic/native-IR、real-layout 和 open-world image-only 三种证据条件。
- [ ] 对 SlideAudit 的 detection、localization、defer 和错误模式单独汇总，禁止与 native-IR 结果直接混合。

## 13.5 Real-agent failure corpus

- [ ] 从 E2E `No critic` 首轮生成结果收集自然缺陷，不从 hybrid 结果反向挑选样本。
- [ ] 保存原始 artifact、render、inspection、人工标签、修复动作和修复后 artifact。
- [ ] 记录自然缺陷发生率、严重度、共现关系、修复成功率和 collateral defects。
- [ ] 只有许可证允许公开的输入和生成结果可进入公开 corpus。
- [ ] 用户私有、商业或敏感数据默认不进入公开 benchmark。
- [ ] 自然失败 corpus 只用于真实场景分析，不参与 critic 阈值选择。
- [ ] 按模型、任务类型和 deck 聚类保存来源，避免把同一失败的多个页面视为独立样本。

## 13.6 E2E 任务集获取与加工

- [ ] 构造 120 个公开来源任务，其中 20 个 pilot、100 个 sealed test。
- [ ] sealed test 包含学术汇报、商业分析、产品介绍和教学讲义各 25 个。
- [ ] 学术材料只使用允许再分发的 arXiv/open-access 来源。
- [ ] 商业分析优先使用 World Bank、政府开放数据等明确许可来源。
- [ ] 产品介绍使用官方开放材料、Wikimedia Commons 和可再利用数据。
- [ ] 教学任务使用 OpenStax、MIT OCW 等开放教育资源。
- [ ] 保存每个来源的 URL、许可证、revision、hash 和本地规范化副本。
- [ ] 将原始材料转换为规范化 Markdown，并保留页码/段落到来源的映射。
- [ ] 使用来源 ID、标题和文本 MinHash 去除重复及近重复任务。
- [ ] 为每个任务生成结构化 brief：受众、目的、页数、语言、必需事实、必需章节、可用素材、风格约束和禁止虚构项。
- [ ] 单人专家逐项核验 brief 能从来源材料完成。
- [ ] pilot 仅用于发现 harness 和任务问题，不进入最终结果。

## 13.7 三臂配对 E2E 实验

- [ ] 为每个任务和 seed 只生成一次首轮 artifact。
- [ ] 从完全相同的首轮 artifact 分叉三种 critic 实验臂。
- [ ] `No critic` 不反馈、不修复，只保留必要的导出安全检查。
- [ ] `Generic critic` 使用一次 whole-rubric VLM verdict，最多 3 轮修复。
- [ ] `Slide Examiner` 使用冻结 symbolic–neural–reference router，最多 3 轮修复。
- [ ] 三臂共享 generation model、初始 artifact、seed、修复轮数和模型预算。
- [ ] sealed test 的 100 个任务各运行 3 个 seed，共得到 900 个最终 deck。
- [ ] 每一轮保存 parent-child lineage、critic report、repair action、成本和最终导出状态。
- [ ] 禁止某实验臂在失败后获得额外人工修复或额外模型预算。
- [ ] 模型服务临时故障只允许按统一重试策略处理，并保留失败记录。

## 13.8 Intrinsic critic 对照与消融

- [ ] 实现 C0 单次 whole-rubric VLM。
- [ ] 实现 C0×10 repeated whole-rubric，控制纯采样预算。
- [ ] 实现 C0+，只在 whole-rubric 中加入目标缺陷名称。
- [ ] 实现 atomic evidence-bearing query。
- [ ] 实现 symbolic-only、VLM-only 和 reference-disabled 对照。
- [ ] 实现 frozen hybrid critic。
- [ ] 实现 mismatched-router negative control，验证路由分工而非单纯组件能力。
- [ ] 对 A=image、B=IR、B′=VLM caption、C=image+IR 条件运行 failure attribution。
- [ ] 对 reference-assisted 类使用 AB/BA 顺序控制。
- [ ] 对每个模型保存 prompt hash、原始输出、结构化 verdict 和调用预算。

## 13.9 Critic 指标

- [ ] 按缺陷类计算 recall、specificity、precision、F1 和 balanced accuracy。
- [ ] 将 macro balanced accuracy 作为 intrinsic primary endpoint。
- [ ] 定位同时报告 element ID exact match 和 bbox IoU≥0.5。
- [ ] 报告 `pass/fail/defer/error/not_applicable` 完整分布。
- [ ] defer 和 error 不得折算为 pass。
- [ ] 计算 Brier score、ECE 和 confidence–accuracy curve。
- [ ] 报告每页调用数、token、延迟、费用和模型失败率。
- [ ] 分别报告 trusted native-IR、neural、reference-assisted 和 image-only 类。
- [ ] 不把论文中的 `0.826` 硬编码成通过阈值，仅作为复现参照。

## 13.10 Repair 指标

- [ ] 计算 target defect removal rate。
- [ ] 报告首轮和三轮累计修复成功率。
- [ ] 计算 collateral defect rate。
- [ ] 比较修复前后严重度、缺陷总数和 hard-gate 状态。
- [ ] 验证文本、页数、图片和必需事实未被修复过程删除。
- [ ] 验证最终 PPTX render fidelity。
- [ ] 将“隐藏内容、移出页面、缩为零、文本转图片”等 reward hacking 计为修复失败。
- [ ] 修复后必须使用新 artifact 重新检查，禁止沿用旧 report。

## 13.11 E2E 指标与人工盲评

- [ ] 将“无严重缺陷且通过导出保真门的 deck 比例”设为 E2E primary endpoint。
- [ ] 对内容正确性、完整性、叙事、视觉设计、可读性和整体可用性进行人工评分。
- [ ] 隐藏实验臂、模型和文件元数据，随机化展示顺序。
- [ ] 进行 `hybrid vs generic` 和 `hybrid vs no critic` 配对偏好判断。
- [ ] 使用独立自动指标评估任务约束、章节覆盖、grounding、页数和 render fidelity。
- [ ] PPTEval 或独立 judge 只能作为 secondary metric，不能充当唯一质量真值。
- [ ] critic 收益不得以 grounding、必需事实保留率或导出成功率下降为代价。
- [ ] 单人专家对至少 15% 样本间隔两周重复盲评。
- [ ] 报告 intra-rater weighted κ。
- [ ] 明确声明该设计不提供 inter-rater reliability 证据。

## 13.12 统计分析

- [ ] 分类指标采用按源 deck 聚类的 bootstrap 95% CI。
- [ ] 配对检测差异使用 McNemar test 或 paired cluster bootstrap。
- [ ] E2E 以 task 为聚类单位、seed 为重复测量。
- [ ] 对二元 endpoint 使用 mixed-effects logistic model。
- [ ] 对人工 ordinal score 使用 mixed-effects ordinal model。
- [ ] 同时报告绝对差、相对差、effect size 和 95% CI。
- [ ] 多类和多对照检验使用 Holm correction。
- [ ] 最小有意义效果预设为 hybrid 相对 generic 的 macro BA 提升至少 5 个百分点。
- [ ] E2E 最小有意义效果预设为 hybrid 相对 no-critic 的 primary endpoint 提升至少 5 个百分点。
- [ ] grounding 和导出成功率的非劣界值设为下降不超过 2 个百分点。
- [ ] 分布迁移、自然缺陷发生率和模型家族差异作为 exploratory analysis。
- [ ] 禁止按最终结果选择 seed、模型或样本子集。

## 13.13 CLI 与执行接口

- [ ] 增加 `pptagent eval prepare`，完成下载、许可校验、去重、注入和 manifest 冻结。
- [ ] 增加 `pptagent eval run --suite intrinsic|e2e --arm ...`。
- [ ] `eval run` 支持固定 seed、断点续跑、并发限制和只重跑失败 case。
- [ ] 增加 `pptagent eval summarize`，只从不可变 run records 重算指标。
- [ ] summarize 不得重新调用模型或修改原始 verdict。
- [ ] 为 full benchmark 增加 `benchmark` pytest marker。
- [ ] CI 仅运行不需要下载大数据和真实模型的小型 smoke fixture。
- [ ] 完整 benchmark 在离线 job 执行并保存版本化结果。

## 13.14 可复现性与审计

- [ ] 保存 generator、critic、semantic model 和 judge 的 provider/model identifier。
- [ ] 保存 sampling parameters、capability flags、prompt hash、router hash 和 reward hash。
- [ ] 保存 Python、Node、Chromium、LibreOffice、Poppler 和系统字体版本。
- [ ] 保存 Git commit、依赖 lock 信息和运行时间。
- [ ] 每个 case 可从 manifest、run record 和 artifact store 完整回放。
- [ ] API key 和敏感路径不得进入结果。
- [ ] 提供完整结果与过滤结果时，必须同时保存过滤规则和被排除 case。
- [ ] 结果汇总必须包含数据完整性失败、模型错误和导出失败数量。

## 13.15 验收门禁

- [ ] 重复运行 `eval prepare` 得到相同 case IDs、split 和 hashes。
- [ ] 不存在源 deck、近重复模板或同源变体跨 split。
- [ ] 所有 injected defective 均通过非零 pixel-diff 验证。
- [ ] 三臂确实从同一首轮 artifact 分叉。
- [ ] 所有实验臂使用相同修复和模型预算。
- [ ] native-IR 类 frozen hybrid balanced accuracy 不低于 0.95；低于阈值先按实现或数据故障调查。
- [ ] sealed test 运行后不得修改 router、prompt 或阈值并覆盖原结果。
- [ ] 最终报告同时包含 intrinsic、SlideAudit image-only、自然失败 corpus 和 E2E 结果。
- [ ] 最终报告同时呈现 detection gain、repair gain、成本和 failure boundary。
- [ ] neural transfer failure、reference unresolved 和 capability downgrade 必须原样报告。
- [ ] 不允许只汇报成功导出的 deck 或最有利模型/实验臂。

## 13.16 测试计划

- [ ] 为 manifest、去重、split 隔离、许可校验和 deterministic IDs 编写 unit tests。
- [ ] 为 XML/IR mutation、pixel-diff、snapping rejection 和非目标缺陷检查编写 paired-fixture tests。
- [ ] 使用 fake model 验证三臂预算、prompt、defer/error、断点恢复和不可变结果。
- [ ] 使用小型公开 fixture 完成 prepare → run → summarize 的离线 E2E smoke test。
- [ ] 使用 browser/export marker 验证 clean/defective 渲染与最终 PPTX fidelity。
- [ ] 使用少量真实模型 pilot 验证 whole-rubric、atomic、reference 和 hybrid 对照能够完整运行。

## 13.17 固定假设与边界

- [ ] 采用论文级研究标准和中等预算。
- [ ] E2E 测试固定使用 100 个 sealed tasks、3 个 seeds 和三臂配对。
- [ ] 任务场景固定为学术、商业、产品和教学等比例分层。
- [ ] 人工评测由单人专家执行，并通过重复盲评估计 intra-rater reliability。
- [ ] 大型数据和模型输出不提交 Git，只提交代码、小型 fixture 和版本化 manifest。
- [ ] 不将单人评测结果表述为跨评审者一致性或普适的人类偏好结论。

### Phase 13 退出条件

- [ ] 数据来源、许可证、版本、hash、split 和加工过程全部可审计。
- [ ] critic benchmark 可独立重放并复算所有指标。
- [ ] 三臂 E2E 实验能够隔离生成随机性与 critic 机制差异。
- [ ] 评测结果能够回答 critic 是否准确、hybrid 路由是否有效、以及 critic 是否改善最终 PPT。
- [ ] 真实 agent failure corpus 能补充 `slide-examiner.pdf` 在自然生成场景中的发生率和修复效果数据。
- [ ] 单人专家评测的限制被明确披露，不声称跨评审者一致性。

---

---

# Phase 14：Agentic RL Environment

> 前置门禁：只有 Phase 13 的 frozen critic、repair、reward 与三臂 E2E 评测通过后才进入本阶段。RL 不得用于弥补尚未验证的 critic 或 reward。

## 14.1 Environment 边界

- [ ] 第一优先实现单页 repair environment，再扩展完整 deck generation。
- [ ] 定义 observation：source excerpt、IR、render URI、inspection report、step budget。
- [ ] 定义 action：policy text/tool calls、source patch、repair action、finalize。
- [ ] 定义 terminal：success、max_steps、invalid_action、export_failure、cancelled。
- [ ] 环境不隐藏自动修改；任何 source 变化都必须对应 action。
- [ ] 对 observation 做稳定序列化，支持离线 replay。

## 14.2 Python Environment 接口

- [ ] 提供 `reset(task, config) -> Observation`。
- [ ] 提供 `step(action) -> StepResult`，包含 observation、reward、done 和 termination reason。
- [ ] 提供 `inspect()`，显式运行 critic 但不修改环境。
- [ ] 提供 `finalize()`，执行最终 export/re-render gate。
- [ ] 提供 `close()`，清理 browser、子进程和临时资源。
- [ ] 环境接口可被本地 trainer 直接 import，避免 HTTP serialization 和服务生命周期复杂度。
- [ ] 如未来确需远程 rollout，再单独增加薄 transport adapter，不进入 v1 核心范围。

## 14.3 Step 执行

- [ ] 每步记录 observation hash 和 action hash。
- [ ] 验证 action 基于当前 revision，拒绝 stale parent artifact。
- [ ] 执行 action 后生成 child artifact。
- [ ] 运行增量 critic 和 reward。
- [ ] 返回 reward vector、aggregate、done、termination reason。
- [ ] 超时、invalid patch 和工具失败作为结构化 step result。
- [ ] 同一 environment instance 禁止并发 step；显式 branch 通过 artifact 创建新的 environment。

## 14.4 Trajectory

- [ ] 使用 append-only JSONL 保存 step envelope。
- [ ] 保存 policy request/response、tool calls、artifact IDs、critic report IDs、reward。
- [ ] 对大型二进制只保存 URI/hash。
- [ ] 保存所有随机 seed、provider/model、sampling parameters。
- [ ] 保存代码版本/commit、config hash、taxonomy/router/reward version。
- [ ] 轨迹支持脱敏导出，移除 API key 和用户敏感附件内容。

## 14.5 Replay

- [ ] `replay --verify` 验证 hashes、parent chain 和 stored rewards。
- [ ] deterministic checker 可离线重跑并与原结果比较。
- [ ] neural checker 默认读取缓存结果；显式 `--rejudge` 才发起新调用。
- [ ] renderer/version 不一致时报告 non-comparable，不静默覆盖。
- [ ] 支持从任意 artifact 创建 branch episode，用于 counterfactual repair。

## 14.6 RL adapters

- [ ] 提供轻量 Python environment adapter，不引入 trainer 框架依赖到主包。
- [ ] 提供 Gymnasium-like adapter 时放到 optional dependency 或独立模块。
- [ ] 支持 synchronous single-env baseline。
- [ ] 支持 async vectorized episodes，控制 browser/model 并发。
- [ ] observation 使用 artifact ID/本地路径或按需加载方法，避免复制大型二进制。
- [ ] 支持 external policy：trainer 直接读取 observation 并提交 action。
- [ ] 支持 internal policy：Slidex 调用配置的 OpenAI-compatible policy endpoint。

## 14.7 单页 repair benchmark

- [ ] 从 clean HTML 注入 G2–G7/S3 可控缺陷。
- [ ] 保留 clean twin、defective artifact 和 mutation manifest。
- [ ] 只保留最终 render 中真实可见/可测的 mutations。
- [ ] 评估 repair success、new-defect rate、steps、tokens 和 wall time。
- [ ] 划分 development/test，冻结 critic 后才跑 test。

### Phase 14 退出条件

- [ ] 本地 trainer 可通过 Python 接口完成 reset → step → reward → done。
- [ ] episode 可回放，reward 可审计。
- [ ] OpenAI-compatible outbound endpoint 可作为 internal policy 插入同一环境。

---

---

# Phase 15：RL 后评测与发布门禁

> 本阶段不复用 Phase 13 sealed test 调参。先冻结 RL policy，再在独立 holdout 上比较 base agent 与 RL agent；只有证明收益且未放大 reward hacking、grounding 或导出风险后才允许发布。

## 15.1 RL 启动与训练门禁

- [ ] Phase 13 的 critic intrinsic benchmark、三臂 E2E benchmark 和 reward calibration 已完成并冻结。
- [ ] reward 不依赖隐藏标签泄漏，clean reference 只在协议明确允许的 class 中提供。
- [ ] mutation 通过 final render fidelity 检查，zero-signal 样本不进入训练或评测。
- [ ] observation、action、reward、trajectory 和 replay schema 冻结为 v1。
- [ ] 环境支持固定 seed、版本、预算和 deterministic replay。
- [ ] 单页 repair baseline 能稳定运行，且 reward hacking probes 已有基线结果。
- [ ] 任何未通过 Phase 13 的 critic/reward 配置不得通过 RL 训练“边跑边修”。

## 15.2 RL policy 冻结与独立评测

- [ ] 在训练前预留与 Phase 13 sealed test 不重叠的 RL holdout，按源文档、deck 和模板隔离。
- [ ] 冻结训练数据、policy checkpoint、sampling 参数、环境版本和停止规则后再打开 holdout。
- [ ] 在相同初始 artifact、seed、模型预算和最大步数下比较 base agent 与 RL agent。
- [ ] 分别报告单页 repair、完整 deck generation 和跨任务分布迁移结果。
- [ ] 主指标沿用 Phase 13 的“无严重缺陷且通过导出保真门的 deck 比例”，不得为 RL 另选有利指标。
- [ ] 同时报告 critic detection、target defect removal、collateral defects、grounding、任务完成率、导出成功率、成本和延迟。
- [ ] 检查 reward improvement 与独立人工/自动质量指标是否一致，识别 Goodhart 或 reward hacking。
- [ ] 对隐藏内容、移出页面、零尺寸、文本转图片、删除必需内容等策略进行专项 adversarial evaluation。
- [ ] 失败、超时、无效 action 和未导出 episode 全部纳入 intention-to-treat 汇总。

## 15.3 复现与结果冻结

- [ ] 保存 base agent 与 RL agent 的完整 artifact lineage、trajectory、reward breakdown 和模型调用记录。
- [ ] 对同一 checkpoint 进行独立 replay，确认离线 reward 可复算且终态 artifact hash 一致。
- [ ] 报告按 task/deck 聚类的置信区间、配对效应量和多重比较校正结果。
- [ ] 区分 Phase 13 的 critic/agent 基线结论与本阶段的 RL 增量结论。
- [ ] 神经 transfer failure、reference unresolved、capability downgrade 和 reward disagreement 按原样报告。
- [ ] 禁止使用 RL holdout 选择 checkpoint、阈值、prompt、router 或 reward 权重。
- [ ] 冻结并保存最终 taxonomy、router、prompt、checker、reward、environment 和 policy 版本。

## 15.4 发布验收

- [ ] 全仓无运行时 Docker 依赖。
- [ ] Slidex CLI 可完成 generate、inspect、repair 和 eval。
- [ ] generation policy、critic 和 RL policy 可调用外部 OpenAI-compatible 模型服务。
- [ ] Python environment 可完成单页 repair episode，并可 replay 已保存 trajectory。
- [ ] 最终 PPTX 经过 re-render validation。
- [ ] 每个输出有完整 artifact manifest、inspection report 和 reward breakdown。
- [ ] RL agent 在独立 holdout 上达到预注册的最小收益，且 grounding 与导出成功率满足非劣界值。
- [ ] RL agent 未显著提高 collateral defect、policy violation 或 inspector error/defer 比例。
- [ ] 未达到 RL 发布门禁时保留并发布通过 Phase 13 的 base agent，不用 RL 结果覆盖可靠基线。
- [ ] legacy `pptagent` 路径仍有明确兼容状态。

### Phase 15 退出条件

- [ ] critic、reward 和 base agent 已先于 RL 得到独立验证。
- [ ] RL 增量效果来自独立 holdout，而不是训练集、development set 或 Phase 13 sealed test 的反复调参。
- [ ] base-vs-RL 的质量、可靠性、成本和风险均可审计、可回放、可复算。
- [ ] 只有通过非劣性与 reward-hacking 门禁的 RL policy 才成为默认发布候选。

---

# 16. 推荐实施批次与依赖关系

> 核心顺序：先完成可交付 agent，再验证 critic/reward 与 E2E 收益，最后才进入 RL。RL 是经过验证系统上的优化阶段，不是 critic 或 reward 的验证工具。

## Batch A：可运行基础（Phase 0–1）

- [ ] 基线测试。
- [ ] Docker SDK、Docker MCP、Docker onboarding 清除。
- [ ] local filesystem tools。
- [ ] Docker-free 单页生成和导出。

**阻塞关系：** 未完成 Batch A，不开始后续领域改造；否则所有阶段都会继承不可控运行时。

## Batch B：可信检查核心（Phase 2–4）

- [ ] Slidex schemas 和 artifact store。
- [ ] Playwright computed IR。
- [ ] symbolic G2–G7/S3。

**里程碑：** 无模型即可对一页 HTML 输出结构化 critic report。

## Batch C：论文 hybrid critic（Phase 5–7）

- [ ] Atomic VLM、semantic、reference inspectors。
- [ ] failure attribution。
- [ ] frozen router。
- [ ] repair loop。

**里程碑：** 完成“检测—定位—修复”闭环，不再依赖宽泛视觉反思。

## Batch D：最终交付物与奖励（Phase 8–9）

- [ ] PPTX re-render。
- [ ] render fidelity 和 zero-signal rejection。
- [ ] reward vector、hard gates 和 replayable reward。

**里程碑：** reward 基于最终 PPTX，而不是 HTML 草稿或 IR 标签。

## Batch E：评测前工程加固（Phase 10–12）

- [ ] OpenAI-compatible outbound model client。
- [ ] 品牌、配置和 legacy compatibility。
- [ ] unit/browser/export/LLM contract tests。
- [ ] 性能、安全和 CI 分层。

**阻塞关系：** Phase 13 必须在统一模型客户端、稳定配置和测试门禁上运行，避免把基础设施故障计为 critic 误差。

## Batch F：Agent 与 Critic 评测（Phase 13）

- [ ] 构建 Zenodo10K、SlideAudit、real-agent failure corpus 和多场景 E2E task set。
- [ ] 完成 intrinsic controls、failure attribution 和 frozen critic evaluation。
- [ ] 完成 No critic / Generic critic / Slide Examiner 三臂配对 E2E。
- [ ] 校准并冻结 critic、router、repair、reward 和主指标。

**RL 前置门禁：** Phase 13 未证明 critic/reward 与最终质量一致时，不开始 RL environment 或 policy training。

## Batch G：Agentic RL 基础设施（Phase 14）

- [ ] Python RL environment。
- [ ] reset/step、trajectory 和 replay。
- [ ] trainer adapters 与 single-slide repair environment。
- [ ] 固定 seed、预算和并发语义。

**里程碑：** trainer 可以驱动已经通过评测的 Slidex environment，但本阶段不宣称 RL 有效。

## Batch H：RL 独立评测与发布（Phase 15）

- [ ] 冻结 policy checkpoint 和 RL holdout。
- [ ] 完成 base-vs-RL 配对评测、非劣性检查和 reward-hacking audit。
- [ ] 冻结最终复现配置和发布产物。
- [ ] RL 未过门禁时发布通过 Phase 13 的 base agent。

**最终里程碑：** agent 基线、critic 机制和 RL 增量效果分别有独立证据，结果可审计、可回放、可复算。

---

# 17. 第一轮可直接执行的文件级 TODO

## `pyproject.toml`

- [ ] 删除 `docker` dependency。
- [ ] 增加 `slidex` console script。
- [ ] 增加 RL 新包的 package data（仅确有非 Python 资源时）。
- [ ] 增加/整理 pytest markers：browser、export、llm、rl。

## `deeppresenter/agents/env.py`

- [ ] 删除 Docker imports 和 daemon/container cleanup。
- [ ] 简化 workspace env。
- [ ] 保留并强化 local tool registration。
- [ ] 增加工具重名检测顺序修复：注册前检查，避免先覆盖再报错。
- [ ] 为 local tool 增加结构化错误和类型标注。

## `deeppresenter/mcp.json.example`

- [ ] 删除 sandbox server。
- [ ] 检查所有 `$PACKAGE_DIR` 路径在安装包中有效。

## `deeppresenter/roles/Design.yaml`

- [ ] 替换 sandbox server 为 local tools。
- [ ] 要求稳定 `data-slidex-id`。
- [ ] 要求读取结构化 inspection report 并按 defect IDs 修复。
- [ ] 禁止把 `defer/error` 当通过。

## `deeppresenter/roles/PPTAgent.yaml`

- [ ] 去掉 sandbox server 依赖。
- [ ] 明确 legacy backend 的有限 critic 能力。

## `deeppresenter/cli/commands.py`

- [ ] 删除 Docker onboarding/check imports。
- [ ] 明确 `serve` 仅管理可选本地模型服务，不代表 Slidex 对外提供 API。
- [ ] 后续增加 inspect、repair 命令。

## `deeppresenter/cli/dependency.py`

- [ ] 删除 Docker 安装、构建和检查函数。
- [ ] 保留 Node、Playwright、Poppler。
- [ ] 增加可选 LibreOffice renderer 检查。

## `deeppresenter/tools/reflect.py`

- [ ] 将 render 与 inspect 拆开。
- [ ] 删除 `This slide is valid.` fallback。
- [ ] 接入 Slidex critic service。
- [ ] 返回 report URI 和 concise summary。

## `deeppresenter/utils/config.py`

- [ ] 增加 critic/semantic/slidex/api 配置。
- [ ] 显式 model capability，减少名称猜测。
- [ ] 配置序列化时隐藏 secrets。

## `deeppresenter/utils/typings.py`

- [ ] 清理 mutable defaults。
- [ ] 不在该通用文件堆放全部 Slidex domain models；新模型放 `slidex/models.py`。
- [ ] InputRequest 增加可选 reference、strict export 和 episode metadata。

## `deeppresenter/utils/webview.py`

- [ ] 扩展一次加载生成 DOM snapshot + PNG/PDF。
- [ ] 固定 browser observation 环境。
- [ ] 支持 artifact store 输出。

## `deeppresenter/main.py`

- [ ] 将 export、final inspection 和 artifact recording 下沉为 services。
- [ ] 最终 PPTX re-render 后再标记 success。
- [ ] 保持现有 generator 行为兼容，避免首轮同时重写所有 agent loop。

## `deeppresenter/html2pptx/`

- [ ] 暴露机器可读 validation report，而不是仅 stderr 文本。
- [ ] 保留 strict/soft 状态和 warnings。
- [ ] 记录字体和 element conversion diagnostics。

---

# 18. 明确不做的事情

- [ ] 第一版不训练 learned critic router。
- [ ] 第一版不把所有 aesthetic quality 压成单一 reward model。
- [ ] 第一版不从 PNG 恢复 boxes 后宣称等价于 native IR。
- [ ] 第一版不重写整个 legacy `pptagent/`。
- [ ] 第一版不实现 Slidex 对外 OpenAI-compatible server；仅实现调用外部 Chat Completions-compatible 模型服务。
- [ ] 第一版不提供任何公网服务或多租户任意 shell execution。
- [ ] 第一版不把 Docker 改成“可选但默认探测”；运行时应彻底不触碰 Docker。
- [ ] 第一版不在 export 失败后把 PDF fallback 伪装成 PPTX 成功。
- [ ] 第一版不将 inspector 的 `defer` 当作零缺陷。

---

# 19. Definition of Done

只有同时满足以下条件，完整改造才算完成：

- [ ] **Docker-free**：安装、onboarding、generation、inspection、export 和 RL episode 均不需要 Docker CLI/daemon/image。
- [ ] **Source-aware**：Slidex 保存 declared IR、computed IR 和最终 render，并明确其证据边界。
- [ ] **Paper-grounded critic**：实现 symbolic、atomic neural、semantic、reference-assisted 四类检查，并由冻结 router 分工。
- [ ] **Structured diagnosis**：每个 defect 有 class、status、evidence、localization、severity 和 repair hint。
- [ ] **Final-artifact validation**：PPTX 重新渲染，mutation/render fidelity 被检查。
- [ ] **Repair loop**：生成和修订形成可审计的 parent-child artifact 链。
- [ ] **Verifiable reward**：reward vector、hard gates、delta reward 可离线复算。
- [ ] **OpenAI compatibility**：generation policy、critic 和 RL policy 可通过 OpenAI Python SDK 调用外部兼容 Chat Completions endpoint。
- [ ] **Reproducible agent evaluation**：critic/router/reward 版本冻结，开发集与测试集分离，三臂 E2E 完成，defer 和 transfer failure 如实保留。
- [ ] **Agentic RL ready**：仅在 agent 评测通过后建设 Python environment，并支持 reset/step/reward/done、trajectory 和 replay。
- [ ] **RL independently validated**：冻结 policy 后在独立 holdout 上证明相对 base agent 的增量收益，且未放大 grounding、导出、collateral defect 或 reward-hacking 风险。
