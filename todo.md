# Slidex 完整工程改造 Roadmap

## 0. 文档目标

本路线图用于将当前 PPTAgent / DeepPresenter 代码库改造成 **Slidex**：一个面向幻灯片生成、诊断、修复和 agentic RL 的 source-aware agent 系统。

改造必须同时完成三项主线：

1. **彻底切断 Docker 运行时依赖**，允许直接侵入源码、检查中间状态并在本机运行工具。
2. **落地 `slide-examiner.pdf` 的核心方法**：可信原生 IR、失败归因、确定性 symbolic linter、原子化神经检查、reference-assisted comparison，以及可定位、可修复、可评分的 hybrid critic。
3. **提供 OpenAI-compatible API**，既能作为 OpenAI SDK 可调用的 agent 服务，也能为后续 agentic RL 暴露稳定、可回放的 episode/step/reward 接口。

本文中的复选框是工程执行清单。每个阶段必须满足退出条件后再进入下一阶段，避免同时重构生成、评估、API 和训练接口而失去可验证基线。

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

- [ ] CLI、HTTP API 和 RL environment 调用同一 application service，禁止复制三套业务流程。
- [ ] 每个 artifact、critic 配置、router 配置和 reward 配置都必须可版本化、可哈希、可回放。
- [ ] 默认 strict validation；忽略错误的 soft mode 只能显式开启，并写入 trajectory。
- [ ] 所有新函数和方法添加类型标注；技术注释和 docstring 使用英文。
- [ ] 优先使用现有 `AgentEnv.register_tool()`、Playwright 和 Pydantic，不引入不必要框架。

---

## 2. 目标架构

```text
OpenAI-compatible API / Native RL API / CLI
                     |
              Application Service
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
- [ ] 新建 `deeppresenter/api/`：FastAPI app、OpenAI compatibility 和 native episode routes。
- [ ] 新建 `deeppresenter/tools/filesystem.py`：替代 Docker sandbox 的本地工作区工具。
- [ ] 保持 `deeppresenter/main.py` 为 generation workflow facade，逐步将底层能力下沉到 application service。

---

# Phase 0：建立基线与改造护栏

## 0.1 固定当前行为

- [ ] 记录当前 `pptagent generate` 的最小可运行调用和输出目录结构。
- [ ] 选取一个 3 页纯文本样例、一个含图片样例、一个故意 overflow 的 HTML 样例作为 smoke fixtures。
- [ ] 记录当前 HTML、PDF、PPTX 输出及 `.history/` 文件格式。
- [ ] 明确哪些集成测试需要 Playwright、Node、Poppler、LibreOffice 或模型凭证。
- [ ] 将测试分成 `unit`、`browser`、`export`、`llm`、`api`、`rl` 六类。

## 0.2 建立基础测试门禁

- [ ] 为 `DeepPresenterConfig` 加载增加无网络 unit test。
- [ ] 为 `InputRequest.copy_to_workspace()` 增加文件和目录测试。
- [ ] 为 `AgentEnv.register_tool()` 增加 sync/async local tool 测试。
- [ ] 为 `convert_html_to_pptx()` 增加 strict validation smoke test。
- [ ] 为 Playwright HTML → image/PDF 增加 browser smoke test。
- [ ] 保存基线测试结果，后续每个 phase 都运行最窄相关集合。

### Phase 0 退出条件

- [ ] 无模型凭证时可运行 unit tests。
- [ ] 有浏览器依赖时可运行单页 HTML render smoke test。
- [ ] 已有最小输入可以验证后续 Docker 移除、critic 和 API 改造没有破坏主链路。

---

# Phase 1：彻底切断 Docker 依赖

## 1.1 删除 Python 运行时依赖

- [ ] 从 `pyproject.toml` 主 dependencies 删除 `docker>=7.1.0`。
- [ ] 检查 `uv.lock`，运行 `uv lock` 清理仅由 Docker SDK 引入的依赖。
- [ ] 从 `deeppresenter/agents/env.py` 删除 `import docker`、`DockerException` 和 `NotFound`。
- [ ] 删除 `AgentEnv.__aenter__()` 中连接 Docker daemon、查找同名容器和强制退出的代码。
- [ ] 删除只服务于 Docker volume mapping 的 `DEEPPRESENTER_HOST_WORKSPACE_BASE` 分支。
- [ ] 保留 `WORKSPACE`、`WORKSPACE_ID`、`CONFIG_FILE` 等普通子进程环境变量。

## 1.2 用本地工具替代 sandbox MCP

- [ ] 在 `deeppresenter/tools/filesystem.py` 实现 `read_file(path)`。
- [ ] 实现 `write_file(path, content)`，自动创建父目录。
- [ ] 实现 `edit_file(path, old, new)`，要求唯一匹配，避免模糊写入。
- [ ] 实现 `list_files(path, pattern)`。
- [ ] 实现 `search_files(query, path, glob)`，优先使用 `rg`。
- [ ] 实现 `run_command(command, cwd, timeout)`，返回 exit code、stdout、stderr。
- [ ] 实现统一的 workspace path resolver，拒绝路径逃逸到 workspace 外部；若研究模式需要外部只读附件，使用显式 allowlist。
- [ ] 限制命令工作目录在 workspace 内，不提供 Docker 式安全承诺，但保证路径和超时行为可预测。
- [ ] 在 `AgentLoop` 创建 `AgentEnv` 后注册这些 local tools。
- [ ] 确认 local tool 输出仍经过当前 cutoff、history 和 timing 记录。

## 1.3 修改角色工具配置

- [ ] 将 `deeppresenter/roles/Design.yaml` 中 `include_tool_servers: [sandbox]` 改为显式 local filesystem tools。
- [ ] 将 `deeppresenter/roles/PPTAgent.yaml` 中的 sandbox 依赖替换为 local tools。
- [ ] 检查 `Research.yaml`、`Planner.yaml`、`SubAgent.yaml` 的 `include_tool_servers: all`，确保不会隐式依赖已删除的 sandbox。
- [ ] 保证 `delegate_subagent`、`thinking`、`finalize` 等 local tools 仍可被显式加入。
- [ ] 当某个 role 配置引用不存在的 server/tool 时，启动阶段给出明确配置错误。

## 1.4 清理 MCP 默认配置

- [ ] 从 `deeppresenter/mcp.json.example` 删除 `sandbox` Docker server 条目。
- [ ] 检查 onboarding 生成的 MCP 配置不会再次加入 sandbox。
- [ ] 保留 `any2markdown`、`task`、`deeppresenter`、`pptagent`、`tool_agents`、`search` 等 stdio MCP。
- [ ] 将新 critic 尽量做成进程内 service；只有确有跨进程复用需求时才暴露 MCP wrapper。

## 1.5 清理 onboarding 和平台依赖

- [ ] 从 `deeppresenter/cli/commands.py:onboard()` 删除 `check_docker_image()`。
- [ ] 从 `deeppresenter/cli/dependency.py` 删除 Docker 安装、镜像构建和检查函数。
- [ ] 删除 CLI 中相关 imports 和提示文案。
- [ ] 保留并验证 Node/npm、Playwright Chromium、Poppler 检查。
- [ ] Linux 本地模型服务不再通过 `deeppresenter/serve.sh` 的 Docker/SGLang 方案启动。
- [ ] 将 Linux 本地模型服务改为显式外部 OpenAI-compatible endpoint，或使用与 macOS 一致的 `llama-server` 可执行文件。
- [ ] 将 `serve` 命令与“启动 Slidex API”区分；本地模型服务命令应改名或标注为 model server。

## 1.6 清理仓库级 Docker 文件

- [ ] 在代码不再引用后删除或停止分发 `deeppresenter/docker/`。
- [ ] 删除或停止使用根目录 `docker-compose.yml`。
- [ ] 删除或停止使用 `.dockerignore`。
- [ ] 检查 `MANIFEST.in` 和 package data 不再包含 Docker 资源。
- [ ] 全仓运行 `rg -n "docker|Docker|sandbox container"`，逐项确认剩余引用是否仅为历史文档。

## 1.7 Docker-free 验证

- [ ] 在没有 Docker CLI、没有 Docker daemon 的环境执行配置加载。
- [ ] 启动 `AgentEnv` 并连接非 Docker MCP servers。
- [ ] 使用 local tools 生成一页 HTML。
- [ ] 完成 HTML → PPTX → PDF/image 转换。
- [ ] 运行 `pptagent generate` 兼容命令。
- [ ] 确认测试和运行日志中不存在 Docker probe。

### Phase 1 退出条件

- [ ] `pyproject.toml` 和运行时代码不依赖 Docker SDK。
- [ ] 默认 MCP 配置不执行 Docker。
- [ ] 无 Docker 环境可完成至少一页生成、检查和导出。
- [ ] 所有 agent 文件操作通过可记录的 local tools 完成。

---

# Phase 2：建立 Slidex 领域模型与 Artifact Store

## 2.1 定义版本化 schema

- [ ] 定义 `DefectClass`，至少包含 G1–G7、S1–S6，并允许未来扩展。
- [ ] 定义 `InspectionStatus = pass | fail | defer | not_applicable | error`。
- [ ] 定义 `EvidenceSource = declared_ir | computed_ir | render | clean_reference | deck_text`。
- [ ] 定义 `BoundingBox`，明确坐标系、单位、页面尺寸和序列化格式。
- [ ] 定义 `SlideElement`：稳定 ID、tag/type、semantic role、text、bbox、style、parent/children。
- [ ] 定义 `DeclaredSlideIR`：源文件声明的结构、容器、主题 token 和预期角色。
- [ ] 定义 `ComputedSlideIR`：浏览器实际 bbox、computed style、scroll size、visibility、stacking 和 font fallback。
- [ ] 定义 `RenderArtifact`：HTML render、PDF render、PPTX re-render 的路径、尺寸、哈希和 renderer 信息。
- [ ] 定义 `SlideArtifact`，聚合 source、declared IR、computed IR、renders 和 provenance。
- [ ] 定义 `InspectionResult`：class、status、severity、confidence、evidence、element IDs、repair hint、latency、cost、inspector version。
- [ ] 定义 `InspectionReport`：单页结果列表、summary、router/taxonomy version。
- [ ] 定义 `RewardBreakdown`：hard constraints、soft scores、cost penalty、aggregate 和 gating reason。
- [ ] 定义 `TrajectoryStep` 和 `EpisodeManifest`。
- [ ] 所有 mutable list/dict 使用 `default_factory`，清理现有 Pydantic mutable defaults。

## 2.2 稳定元素 ID

- [ ] 规定生成 HTML 中每个可检查元素必须有 `data-slidex-id`。
- [ ] Design prompt 要求 ID 在修订时保持稳定，不得每轮全部重编号。
- [ ] 浏览器 extractor 对缺失 ID 的元素生成 deterministic fallback ID，并发出 warning。
- [ ] ID 必须能跨 source HTML、DOM snapshot、render annotation、critic report 和 repair action 关联。
- [ ] 对重复 ID、空 ID 和层级变化增加 validation。

## 2.3 Artifact Store

- [ ] 为每次 episode 创建独立 workspace，不复用可变全局目录。
- [ ] 采用 `artifacts/<artifact_id>/` 保存 source、IR、renders、inspection 和 reward。
- [ ] artifact ID 使用内容哈希或 UUID + 内容哈希，避免仅靠文件名。
- [ ] 保存 `manifest.json`：父 artifact、创建 action、模型、sampling 参数、工具调用和版本。
- [ ] 对 HTML、CSS、图片、IR JSON、PNG、PDF、PPTX 分别计算 SHA-256。
- [ ] 记录 renderer 名称和版本，如 Chromium、html2pptx、LibreOffice。
- [ ] artifact 写入采用临时目录 + atomic rename，避免 API 并发读到半成品。
- [ ] 大文件不嵌入 trajectory JSONL，只记录 artifact URI 和 hash。
- [ ] 增加 workspace/artifact 配额和清理策略，但不得在活跃 episode 中自动删除。

## 2.4 配置模型

- [ ] 在 `DeepPresenterConfig` 中增加 `slidex` 子配置。
- [ ] 增加 `taxonomy_version`、`router_version`、`reward_version`。
- [ ] 增加安全边距、alignment tolerance、overlap tolerance、palette threshold。
- [ ] 增加 max repair rounds、max episode steps、command timeout。
- [ ] 增加 strict export、PPTX re-render 和 reference policy 开关。
- [ ] 增加独立 `critic_model` 和可选 `semantic_model`，不能默认与 policy history 共享状态。
- [ ] 为旧配置提供清晰迁移默认值；未知关键字段应报错而不是静默忽略。
- [ ] 更新 `deeppresenter/config.yaml.example` 展示 OpenAI-compatible outbound endpoint 和 critic 配置。

## 2.5 Schema 测试

- [ ] 为每个 Pydantic model 增加 round-trip JSON test。
- [ ] 测试旧版本 manifest 的显式拒绝或 migration。
- [ ] 测试坐标越界、负尺寸、重复 ID、未知 status。
- [ ] 生成一份单页 artifact fixture 并验证所有 hash 可复算。

### Phase 2 退出条件

- [ ] 一页 HTML 可以被表示为完整 `SlideArtifact`。
- [ ] source、computed IR、render 和 provenance 可通过稳定 ID 关联。
- [ ] schema 和 artifact manifest 可独立于 agent/LLM 测试。

---

# Phase 3：实现可信 Native IR 和浏览器观测

## 3.1 Source/declared IR 提取

- [ ] 从 HTML 提取 `data-slidex-id`、semantic role、容器关系、文本和资源引用。
- [ ] 从 `global.css` 和页面 style 提取设计 token：字体 scale、palette、safe area 和 grid hints。
- [ ] 明确 declared IR 是生成管线拥有的源结构，不使用像素 layout detector 替代。
- [ ] 对动态脚本、远程字体和交互依赖默认禁止或标记为不可复现。
- [ ] 对没有 declared semantic role 的元素标记 `unknown`，不猜测为 ground truth。

## 3.2 Playwright computed IR

- [ ] 重构 `PlaywrightConverter`，允许一次 page load 同时生成 DOM snapshot、截图和 PDF。
- [ ] 获取每个元素的 `getBoundingClientRect()`。
- [ ] 获取 `clientWidth/clientHeight`、`scrollWidth/scrollHeight`。
- [ ] 获取 text node range bounding rects，支持文本真实占用范围检测。
- [ ] 获取关键 computed styles：font family/size/weight、color、background、overflow、display、visibility、opacity、z-index、transform。
- [ ] 获取实际使用字体/字体 fallback；复用 html2pptx 已有 CDP font detection 思路。
- [ ] 获取图片 natural size、object-fit、clip 和 load status。
- [ ] 获取元素可见区域和页面交集，识别完全/部分出界。
- [ ] 等待 `document.fonts.ready`、图片加载和网络空闲后再采样。
- [ ] 固定 viewport、device scale factor、locale、timezone 和 browser flags，减少 reward 漂移。
- [ ] 将 JS console error、page error 和资源加载失败写入 artifact。

## 3.3 DOM 与 render 一致性

- [ ] 在 screenshot 上支持绘制 debug overlay，标出 element ID 和 bbox。
- [ ] 检查 CSS transform 后 bbox 与 source geometry 的差异。
- [ ] 检查 clipping、pseudo-element、shadow 和 SVG 等无法完整表达在 declared IR 中的内容。
- [ ] 记录每页 render readiness；未加载完成的页面不得进入 critic。
- [ ] 将当前 `inspect_slide` 的临时目录行为改为 artifact store 管理，避免无法回放。

## 3.4 浏览器观测测试

- [ ] fixture：正常文本框，验证 client/scroll 相等。
- [ ] fixture：文本 overflow，验证 scroll dimension 超出。
- [ ] fixture：hidden overflow，验证内容超出但像素被裁剪。
- [ ] fixture：absolute positioned child 越过 container。
- [ ] fixture：页面边界和 safe margin 违规。
- [ ] fixture：transform、pseudo-element 和 SVG。
- [ ] fixture：字体 fallback 导致换行变化。
- [ ] 同一 fixture 连续运行多次，确认 IR 和 reward 输入稳定。

### Phase 3 退出条件

- [ ] 单次浏览器加载可产出 computed IR、PNG 和 PDF。
- [ ] G2/G6/G7 所需几何证据无需 VLM 即可取得。
- [ ] 所有证据可定位到稳定 element ID。

---

# Phase 4：实现论文中的 Symbolic Linters

## 4.1 G2 元素重叠

- [ ] 基于 computed/native bbox 计算 pairwise intersection。
- [ ] 区分预期重叠与异常重叠：背景、装饰、容器子元素、overlay allowlist。
- [ ] 使用面积比例和最小像素阈值过滤抗锯齿级误差。
- [ ] 输出两个 element ID、intersection bbox、面积和严重度。
- [ ] clean fixtures 必须零误报。

## 4.2 G3 alignment offset

- [ ] 提取 left/right/center/top/bottom alignment candidates。
- [ ] 对重复列、卡片和文本基线进行 clustering。
- [ ] 使用配置化 tolerance，不将所有自由布局强制吸附到统一 grid。
- [ ] 只有存在足够 sibling evidence 时才判定 fail，否则 `not_applicable` 或 `defer`。
- [ ] 输出偏移方向、像素值、参考元素集合和 repair target。

## 4.3 G4 font-size inconsistency

- [ ] 建立 deck typography scale，而不是单页孤立阈值。
- [ ] 按 semantic role 比较 title/subtitle/body/caption/footer。
- [ ] 识别同角色的异常字号、字体和 weight。
- [ ] 对 intentional emphasis 使用 role/attribute allowlist。
- [ ] 输出 expected scale、actual value 和相关页面。

## 4.4 G5 brand-color violation

- [ ] 从 `global.css` / theme 配置读取可信 palette。
- [ ] 将 CSS colors 规范化到统一色彩空间。
- [ ] 使用 CIEDE2000，而不是 RGB 欧氏距离。
- [ ] 分开处理文本色、背景色、边框色和装饰色。
- [ ] 透明度合成后再判断最终可见颜色。
- [ ] 输出最近 palette color、Delta E 和修改建议。

## 4.5 G6 margin violation

- [ ] 按 aspect ratio 定义 safe area。
- [ ] 检查可见 bbox，而非仅 CSS 声明值。
- [ ] 背景 bleed 和装饰性 full-bleed 元素可显式豁免。
- [ ] 输出越界边、距离和允许边界。
- [ ] 对 poster 与 presentation 使用独立默认阈值。

## 4.6 S3 terminology inconsistency

- [ ] 从 deck 全文建立 occurrence table：term、slide、element、context。
- [ ] 先做确定性 normalization：大小写、全半角、连字符、复数和空白。
- [ ] 使用近重复聚类发现 subtle variants，但保留可解释 edit evidence。
- [ ] 允许用户提供 glossary 和 accepted aliases。
- [ ] 只输出候选冲突，不自动把所有近义词视为错误。
- [ ] 输出 canonical suggestion、所有 occurrences 和修改目标。

## 4.7 G1 与 G7 的工程化区分

- [ ] G1 定义为 declared text/container 约束违规；在 Slidex 源结构中可判定的部分走 symbolic。
- [ ] G7 定义为 declared container 合法、但 computed/rendered content 越界。
- [ ] 使用 scroll metrics、text ranges 和 child union 检测 DOM-level G7。
- [ ] 对 DOM 无法解释的 pixel-only anomaly 标记为 unresolved，交给 atomic render inspector。
- [ ] 不因为 Slidex 可直接检测 DOM overflow 就删除论文中的 render-gap 概念；必须保留 export 后再次检查。

## 4.8 统一 linter 输出

- [ ] 所有 linters 实现统一 `Inspector` protocol。
- [ ] 所有结果必须使用 `InspectionResult`，禁止返回自由文本字典。
- [ ] 每个 inspector 记录名称、版本、输入 hash 和耗时。
- [ ] checker 抛出的内部异常转换为 `error`，不能伪装成 `pass`。
- [ ] 无足够证据时返回 `defer/not_applicable`，不能用默认阈值硬判。

## 4.9 Linter 测试

- [ ] 每个 class 至少准备 clean/defective matched pair。
- [ ] 检查 defective 触发且 clean 静默，避免“全报错”获得高 recall。
- [ ] 测试阈值边界和 magnitude sweep。
- [ ] 测试真实自由布局，避免 checker 只适合合成网格。
- [ ] 对 clean suite 设定零误报或明确可接受上限。

### Phase 4 退出条件

- [ ] G2–G6 和 S3 可在无模型环境运行。
- [ ] G1/G7 的 declared/computed/render 边界被显式记录。
- [ ] 每个 fail 都能定位元素并给出机器可读 repair hint。

---

# Phase 5：实现 Atomic Neural Inspectors 与 Failure Attribution

## 5.1 独立 critic 模型配置

- [ ] 增加 `critic_model`，要求显式声明 `is_multimodal`，不要只根据模型名称猜测。
- [ ] 增加可选 `semantic_model` 处理 deck-level S2/S5。
- [ ] critic 调用使用独立、无生成 history 的 request。
- [ ] 固定 temperature、top_p、seed（服务支持时）和 response schema。
- [ ] 保存 endpoint identifier、model、sampling 参数、usage、latency 和 raw response。
- [ ] 对不支持 image、JSON schema 或 tool call 的 OpenAI-compatible provider 给出 capability error。

## 5.2 原子查询协议

- [ ] 每次 neural call 只检查一个 defect class。
- [ ] prompt 必须给出 operational definition，而不是抽象“是否美观”。
- [ ] 要求 `verdict = pass | fail | defer`。
- [ ] 要求 localization：element ID 或归一化 bbox。
- [ ] 要求 evidence：可观察事实，不接受只给总体分数。
- [ ] 要求 repair suggestion，但 repair suggestion 不参与 defect verdict。
- [ ] 使用 Pydantic structured output；soft parsing 失败时返回 `error`。
- [ ] 禁止在同一 prompt 中展示 ground-truth label 或 mutation metadata。

## 5.3 S1 title/body mismatch

- [ ] 从 IR 提供 title 和 body text，同时提供 render 作为补充证据。
- [ ] 原子询问正文是否与标题主题矛盾或明显不匹配。
- [ ] 区分“不完整”与“矛盾”，避免泛化为内容质量评分。
- [ ] 输出冲突文本片段和 element IDs。

## 5.4 S4 density violation

- [ ] 先计算 symbolic statistics：字符数、占用面积、字号、空白率、元素数。
- [ ] 明显超阈值情况可直接 deterministic fail。
- [ ] 边界情况由 semantic examiner 判断信息量是否与页面作用匹配。
- [ ] 区分 over-packed、under-packed 和 intentional minimal title slide。
- [ ] 输出统计证据和语义理由。

## 5.5 S6 image/text contradiction

- [ ] 原子检查单个 image-caption/claim pair，不一次检查整页所有语义关系。
- [ ] 从 IR 提供周围 claim/caption，并提供 crop 或整页 render。
- [ ] 单视图证据不足时返回 `defer`。
- [ ] 有 clean/reference 时使用 pairwise comparison。
- [ ] 增加顺序对调控制，记录 positional disagreement。

## 5.6 unresolved G7 render anomaly

- [ ] 仅对 DOM symbolic checker 无法判定的 render anomaly 调用 atomic VLM。
- [ ] query 明确询问“内容是否越过或被裁剪于指定容器”，并提供目标 bbox overlay。
- [ ] 要求定位目标元素，禁止 broad visual-quality rubric。
- [ ] DOM 已确定 fail 时不得重复调用 VLM 浪费预算。

## 5.7 Deck-level S2/S5

- [ ] S2 narrative-order break 使用 deck outline、每页标题和摘要，不依赖逐页截图堆叠。
- [ ] S5 missing logic section 根据任务/批准 outline 判断缺失步骤。
- [ ] 明确这是 deck-level semantic inspector，结果关联相关 slide IDs。
- [ ] 允许任务本身没有固定逻辑结构时返回 `not_applicable`。

## 5.8 Reference-assisted inspector

- [ ] `InspectionContext` 支持 clean/reference artifact ID。
- [ ] reference 必须与目标页在 aspect ratio、renderer 和页面角色上兼容。
- [ ] pairwise prompt 支持 `left | right | tie | defer`。
- [ ] 同一 pair 运行 AB/BA 两种顺序，并聚合为 order-controlled verdict。
- [ ] clean-vs-clean 控制用于检测 forced-choice 偏差。
- [ ] 无 reference 时返回 `defer: clean_reference_required`，不自动伪造 synthetic twin。

## 5.9 Failure attribution protocol

- [ ] 实现论文中的观察条件：A=image、B=trusted structured IR、C=image+IR、可选 reference。
- [ ] attribution 只用于开发/评测和未知 class 诊断，不要求生产每次都运行全部条件。
- [ ] 记录 `image_sufficient`、`structure_rescued`、`format_suppressed`、`reference_assisted`、`unresolved`。
- [ ] attribution 结论描述操作层面的证据需求，不声称揭示模型内部机制。
- [ ] 为同一模型/同一 item 保存 whole-rubric 与 atomic query 对照。
- [ ] 支持 repeated whole-rubric budget control，用于验证收益不是仅来自更多采样。

## 5.10 Neural inspector 测试

- [ ] 使用 fake OpenAI-compatible server 测试请求 payload 和 structured response。
- [ ] 测试 timeout、429、invalid JSON、missing verdict、provider 不支持 image。
- [ ] 对真实模型测试放入 `llm` marker，不阻塞无凭证 CI。
- [ ] 保存一组人工审核的 atomic-query fixtures，检查 prompt 版本变化。

### Phase 5 退出条件

- [ ] 每个 neural inspector 都是单类、可定位、可 defer 的结构化调用。
- [ ] reference-required 情况不会被错误判为 pass/fail。
- [ ] failure attribution 可重现实验性 A/B/C 对照。

---

# Phase 6：冻结 Hybrid Critic Router

## 6.1 Router v1 映射

- [ ] 定义默认 frozen mapping：G2/G3/G4/G5/G6 → symbolic native-IR inspectors。
- [ ] S3 → terminology linter。
- [ ] G7 → DOM render overflow；unresolved 时 atomic VLM。
- [ ] S1 → atomic semantic inspector。
- [ ] S4 → symbolic density statistics + semantic boundary inspector。
- [ ] S2/S5 → deck-level semantic inspector。
- [ ] S6 → atomic VLM；证据不足时 reference inspector 或 defer。
- [ ] G1 → source/computed checker；无法从单页确定的情形进入 reference policy。

## 6.2 Router 行为约束

- [ ] router 输入仅包含 defect class、available evidence、artifact trust level 和配置。
- [ ] router 不读取模型输出后再 post-hoc 改路线以提高分数。
- [ ] inspector `defer` 是终态之一；只有 frozen policy 明确允许时才进入下一 inspector。
- [ ] router 输出记录选择原因和缺失证据。
- [ ] router 配置保存为机器可读对象并计算 hash。
- [ ] taxonomy/router 变更必须提升版本，旧 trajectory 保留旧版本解释。

## 6.3 Trust policy

- [ ] `native_html` 和 Slidex 生成的 DOM 标为 trusted source IR。
- [ ] 第三方 PPTX 提取的原生 XML 标为 partial trusted，并声明缺少哪些 bookkeeping。
- [ ] 从 PNG/PDF layout detector 恢复的 boxes 标为 recovered/untrusted。
- [ ] symbolic linter 对 recovered structure 不宣称 native-IR 保证。
- [ ] open-world image-only 输入自动降级为 VLM-only，并在 report 中显式标注能力上限。

## 6.4 Critic 聚合

- [ ] 汇总 per-class inspection results，不用平均值掩盖 hard failure。
- [ ] summary 明确列出 fail、defer、error 和 not-applicable 数量。
- [ ] 对 inspector 冲突保留双方结果，不直接覆盖。
- [ ] 增加 deterministic priority 规则，但仅适用于 trusted native predicates。
- [ ] critic report 写入 artifact store，并可通过 API 获取。

## 6.5 与 `inspect_slide` 集成

- [ ] 将 `tools/reflect.py:inspect_slide` 改为调用 Slidex critic service。
- [ ] 返回结构化 JSON 文本或 artifact URI，不再仅返回 image block。
- [ ] 删除未启用 multimodal 时返回 `This slide is valid.` 的错误语义。
- [ ] 无 critic model 时仍运行 symbolic inspectors；神经项返回明确 `defer/unavailable`。
- [ ] 提供单独 `render_slide` 工具给 agent 请求视觉预览，避免检查与渲染职责混合。
- [ ] Design Agent 收到 report 后必须逐项处理 hard failures，并在下一 revision 关联前一 artifact。

### Phase 6 退出条件

- [ ] `inspect_slide` 在无 VLM 时仍能提供可信 symbolic report。
- [ ] router 版本、路线和 defer 行为均可回放。
- [ ] 当前“看图自省”已替换为论文式 hybrid critic，而不是在其旁边叠加另一个评分器。

---

# Phase 7：Repair Loop 与生成流程集成

## 7.1 机器可读 RepairAction

- [ ] 定义 `RepairAction`：operation、target IDs、constraints、source inspection IDs。
- [ ] 支持 `move_element`、`resize_container`、`reduce_text`、`change_font_size`、`replace_color`、`rename_term` 等操作类型。
- [ ] repair hint 是建议，不直接修改 source；执行后必须重新 inspect。
- [ ] 每个 action 保存 before/after artifact ID。
- [ ] 不可执行的自由文本建议标记为 `policy_edit`。

## 7.2 Policy repair

- [ ] 更新 Design prompt：先读结构化 report，再只修改被定位的元素。
- [ ] 禁止为了通过 checker 删除核心内容或把元素隐藏。
- [ ] 修订时保留稳定 element ID。
- [ ] 每页设置 max repair rounds，超过后返回带 unresolved defects 的终态。
- [ ] agent 不得在 inspector `error` 时假装页面通过。

## 7.3 Deterministic repair（可选工具）

- [ ] 实现安全边距 clamp。
- [ ] 实现 palette replacement。
- [ ] 实现 terminology canonical replacement。
- [ ] 对 alignment snap、font shrink 等可能改变设计意图的操作默认仅建议，不自动执行。
- [ ] deterministic repair 必须作为显式 action 写入 trajectory，不能后台静默修改。

## 7.4 Deck-level final inspection

- [ ] 所有单页通过后运行 S2/S3/S5 和 typography/palette deck consistency。
- [ ] deck-level repair 必须指明受影响页面，避免全 deck 无差别重生成。
- [ ] 修订某页后只重跑受影响的 page inspectors，加上必要 deck inspectors。
- [ ] finalization 前若存在 hard fail，默认阻止导出；允许显式 override，并记录原因。

## 7.5 防 reward hacking 检查

- [ ] 检查 opacity、visibility、off-screen positioning、zero-size 等隐藏内容行为。
- [ ] 检查重要文本被转为不可解析图片以绕过 terminology/semantic checker。
- [ ] 检查字体过小但无 overflow 的规避行为。
- [ ] 检查将所有元素标记为 decorative/allow-overlap 的滥用。
- [ ] 检查删除 manuscript 必需内容以降低 density。
- [ ] 将这些行为列为 hard policy violations。

### Phase 7 退出条件

- [ ] 生成 → inspect → localized repair → re-inspect 形成闭环。
- [ ] 每次修订有明确父 artifact 和 defect delta。
- [ ] hard failure 不能被 aesthetic/semantic 高分抵消。

---

# Phase 8：最终导出物验证与 Render Fidelity

## 8.1 Strict HTML → PPTX

- [ ] 将训练/API 默认 `soft_parsing` 改为 `False`。
- [ ] validation error 转成明确 invalid artifact 和 hard penalty。
- [ ] soft mode 只能由请求显式开启，并记录所有 ignored warnings。
- [ ] 将 html2pptx stdout/stderr、版本和命令参数写入 export manifest。

## 8.2 PPTX 重新渲染

- [ ] 确定首选 headless renderer（例如 LibreOffice）并检测版本。
- [ ] 将生成的 PPTX 渲染为 PDF/PNG。
- [ ] 每页输出稳定命名并关联 source slide ID。
- [ ] renderer 缺失时返回 capability error，不把 HTML render 冒充 PPTX render。
- [ ] 可选支持 PowerPoint render worker，但不作为本地必需依赖。

## 8.3 Render-fidelity gate

- [ ] 比较 HTML screenshot 与 PPTX re-render 的页面尺寸和页数。
- [ ] 计算像素差/感知差，但不把单一相似度当质量结论。
- [ ] 对关键元素比较 OCR/text presence 或可用的导出结构。
- [ ] 对 G1/G7、margin、missing image 重新检查最终 render。
- [ ] 检测 PPTX 中字体替换、换行和元素位置变化。
- [ ] render 差异超过阈值时标记 `export_fidelity_failure`。

## 8.4 Template snapping / mutation fidelity

- [ ] 对所有注入式训练数据保存 clean 和 defective 最终 render。
- [ ] 如果最终像素相同，则样本标记 `zero_signal` 并从 detection/reward 训练中排除。
- [ ] 统计每类 mutation 的 render survival rate。
- [ ] 标签来自最终可观察 artifact，而不是仅来自 IR mutation 操作。
- [ ] clean twin 也必须经过同一 renderer 和版本。

## 8.5 最终 artifact 状态

- [ ] 区分 `draft_html_valid`、`pptx_exported`、`pptx_render_validated`。
- [ ] API 默认只将 `pptx_render_validated` 标为成功终态。
- [ ] 导出失败时保留 HTML/PDF 调试 artifact，但不得把 PDF fallback 宣称为 PPTX 成功。
- [ ] `intermediate_output.json` 迁移到更明确的 artifact manifest，同时提供兼容字段。

### Phase 8 退出条件

- [ ] reward 和成功状态基于最终交付物，而非仅基于 HTML 草稿。
- [ ] template snapping/导出重排造成的标签失真可被自动检测。
- [ ] 任一最终 PPTX 可以追溯到 source、critic report 和 renderer。

---

# Phase 9：Reward System

## 9.1 Reward vector

- [ ] 定义 `validity_reward`：source、browser、export 和 re-render 是否成功。
- [ ] 定义 `geometry_reward`：G1–G7 的 per-class pass/fail/severity。
- [ ] 定义 `semantic_reward`：S1–S6 的结果，defer 与 error 单独记录。
- [ ] 定义 `fidelity_reward`：HTML 与最终 PPTX render 一致性。
- [ ] 定义 `task_reward`：页面数、outline、必需内容和用户约束。
- [ ] 定义 `efficiency_reward`：token、模型调用、工具调用、repair steps 和 latency。
- [ ] 定义 `policy_violation_penalty`：隐藏内容、路径逃逸、无效 action 等。

## 9.2 Hard-gated aggregation

- [ ] invalid export 触发 terminal hard negative。
- [ ] 存在严重 overflow、missing asset、页面出界时不发 aesthetic bonus。
- [ ] defer 不等于 pass；聚合时保留 coverage。
- [ ] inspector error 不直接算 defect miss，但触发 reliability penalty/episode invalidation。
- [ ] 所有 hard gate 的阈值进入 `reward_version` 配置。
- [ ] API 同时返回 reward vector 和 aggregate scalar。

## 9.3 Repair delta reward

- [ ] 计算修订前后 fail 数量和 severity delta。
- [ ] 奖励解决目标 defect，同时惩罚引入新 defect。
- [ ] 保留未变化、改善、恶化三类 per-class transition。
- [ ] 避免仅因多次微小编辑累积无限正奖励；采用 potential-based 或终态奖励约束。
- [ ] 对 deterministic tool repair 与 policy repair 分开统计。

## 9.4 Reward calibration

- [ ] 使用 matched clean/defective pairs 校准 hard checker。
- [ ] 报告 recall、specificity、balanced accuracy 和 localization，而非只报 accuracy。
- [ ] 神经 checker 按模型/provider 分开校准。
- [ ] 对 clean-vs-clean、AB/BA 和 repeated query 做偏差控制。
- [ ] reward 配置冻结后再跑 held-out evaluation。
- [ ] development 调参数据与最终评测数据严格分离。

## 9.5 Reward 输出与审计

- [ ] 每个 reward component 引用产生它的 inspection result ID。
- [ ] 保存聚合公式、权重和 gate reason。
- [ ] 提供 `explain_reward()`，输出机器可读解释。
- [ ] 支持离线根据相同 artifacts 重新计算 reward。
- [ ] reward 重算不能重新调用不确定模型，除非显式开始新的 evaluation run。

### Phase 9 退出条件

- [ ] 同一 artifact + 同一 critic/reward version 可得到可复算结果。
- [ ] hard defect 无法被 soft score抵消。
- [ ] reward 能用于单页 repair RL，也能汇总 deck generation。

---

# Phase 10：OpenAI-compatible API

## 10.1 API application 基础

- [ ] 新建 FastAPI app factory，避免 import 时加载模型或读取全局配置。
- [ ] 使用 lifespan 管理 Playwright browser、HTTP clients 和后台 task registry。
- [ ] 配置 host、port、API key、CORS、workspace base 和并发上限。
- [ ] 增加健康检查，但 OpenAI compatibility 核心保持在 `/v1`。
- [ ] 所有请求生成 request ID，写入日志、episode 和响应 header。
- [ ] 将内部异常映射为 OpenAI 风格 error object。

## 10.2 `GET /v1/models`

- [ ] 返回 OpenAI 风格 model list。
- [ ] 至少暴露 `slidex-generate`、`slidex-design`、`slidex-repair`、`slidex-critic`。
- [ ] 每个 model ID 映射到固定 application capability，不直接泄露底层 provider key。
- [ ] 返回稳定 owner 和 created 字段。
- [ ] 为未知 model 返回 404 `model_not_found`。

## 10.3 `POST /v1/chat/completions`

- [ ] 使用 OpenAI Chat Completions request schema 的核心字段：model、messages、stream、temperature、top_p、max_tokens、tools、tool_choice、response_format、metadata。
- [ ] 支持文本 prompt。
- [ ] 支持 image URL/data URL 输入，用于 critic 和 repair。
- [ ] 支持 tool calls 的 OpenAI-compatible serialization。
- [ ] 非流式响应包含 id、object、created、model、choices、usage。
- [ ] `finish_reason` 区分 stop、tool_calls、length、content_filter/error policy。
- [ ] artifact 不直接 base64 塞入 content；返回 artifact ID/URL 和结构化 summary。
- [ ] `slidex-generate` 将请求映射到 generation application service。
- [ ] `slidex-critic` 接收 artifact reference 或上传后的文件，返回 inspection report。
- [ ] `slidex-repair` 接收 artifact + target inspection IDs，产生 child artifact。

## 10.4 Streaming SSE

- [ ] 实现 `chat.completion.chunk`。
- [ ] 第一 chunk 发送 role，后续发送 content/tool-call deltas。
- [ ] 生成过程中将阶段事件转换为兼容文本 delta；详细事件通过 native API 获取。
- [ ] 正常结束发送 `[DONE]`。
- [ ] client disconnect 时取消未完成任务并安全关闭 episode，或转为后台任务并返回策略一致的状态。
- [ ] 测试 OpenAI Python SDK 的 sync/async streaming。

## 10.5 OpenAI 错误与 usage

- [ ] 实现 `invalid_request_error`、`authentication_error`、`rate_limit_error`、`model_not_found`、`server_error`。
- [ ] 对请求 schema、附件、模型 capability 和 artifact ownership 分别校验。
- [ ] usage 至少汇总底层 LLM prompt/completion tokens。
- [ ] 额外的 critic/tool/browser cost 放在 response metadata 或 native episode，不破坏兼容字段。
- [ ] 不在日志、manifest 或错误中泄漏 API key。

## 10.6 文件与 artifact API

- [ ] 提供 multipart upload，返回 file/artifact ID。
- [ ] 提供 artifact metadata endpoint。
- [ ] 提供带权限检查的下载 endpoint。
- [ ] 支持 Range 或合理的大文件传输策略。
- [ ] 文件类型、大小、页数和压缩包内容做限制。
- [ ] API 返回的本地路径不得直接暴露宿主机绝对路径。

## 10.7 CLI 接入

- [ ] 增加 `slidex serve-api` 命令。
- [ ] `serve-api` 与当前“启动本地模型”的 `serve` 分离，避免语义冲突。
- [ ] 增加 `--host`、`--port`、`--api-key`、`--config`、`--workers`。
- [ ] Playwright/browser singleton 不支持多进程共享时，默认单 worker，并通过任务并发控制扩展。
- [ ] 保留 `pptagent` CLI alias，新增 `slidex` 主入口。

## 10.8 Compatibility tests

- [ ] 使用官方 `openai` Python SDK 调用 `/v1/models`。
- [ ] 测试同步和异步 `chat.completions.create()`。
- [ ] 测试 stream、tools、response_format 和 image input。
- [ ] 使用 fake downstream OpenAI-compatible provider 做无外网端到端测试。
- [ ] 增加 curl smoke tests。
- [ ] 明确当前只兼容 Chat Completions，若暂不支持 Responses API，返回清晰 404 而非半兼容实现。

### Phase 10 退出条件

- [ ] 标准 OpenAI SDK 可发现 Slidex models 并发起非流式/流式请求。
- [ ] generation、critic、repair 三类能力共享同一 artifact/episode backend。
- [ ] OpenAI 兼容层不承担 RL 私有状态协议。

---

# Phase 11：Native Agentic RL API 与 Environment

## 11.1 Environment 边界

- [ ] 第一优先实现单页 repair environment，再扩展完整 deck generation。
- [ ] 定义 observation：source excerpt、IR、render URI、inspection report、step budget。
- [ ] 定义 action：policy text/tool calls、source patch、repair action、finalize。
- [ ] 定义 terminal：success、max_steps、invalid_action、export_failure、cancelled。
- [ ] 环境不隐藏自动修改；任何 source 变化都必须对应 action。
- [ ] 对 observation 做稳定序列化，支持离线 replay。

## 11.2 Episode API

- [ ] `POST /v1/slidex/episodes`：创建 episode，指定 task、mode、policy、critic/reward versions。
- [ ] `GET /v1/slidex/episodes/{id}`：获取状态、budget、当前 artifact 和累计 reward。
- [ ] `POST /v1/slidex/episodes/{id}/steps`：提交 action 并返回 observation/reward/done。
- [ ] `POST /v1/slidex/episodes/{id}/inspect`：显式触发 critic，不修改环境。
- [ ] `POST /v1/slidex/episodes/{id}/finalize`：执行最终 export/re-render gate。
- [ ] `GET /v1/slidex/episodes/{id}/trajectory`：下载完整轨迹 manifest/JSONL。
- [ ] `DELETE /v1/slidex/episodes/{id}`：取消活跃 episode，保留审计状态。
- [ ] API 操作使用 idempotency key，防止 trainer retry 重复执行 action。

## 11.3 Step 执行

- [ ] 每步记录 observation hash 和 action hash。
- [ ] 验证 action 基于当前 revision，拒绝 stale parent artifact。
- [ ] 执行 action 后生成 child artifact。
- [ ] 运行增量 critic 和 reward。
- [ ] 返回 reward vector、aggregate、done、termination reason。
- [ ] 超时、invalid patch 和工具失败作为结构化 step result。
- [ ] 并发 step 使用 episode lock，禁止同一 episode 分叉写入；显式 branch API 另行实现。

## 11.4 Trajectory

- [ ] 使用 append-only JSONL 保存 step envelope。
- [ ] 保存 policy request/response、tool calls、artifact IDs、critic report IDs、reward。
- [ ] 对大型二进制只保存 URI/hash。
- [ ] 保存所有随机 seed、provider/model、sampling parameters。
- [ ] 保存代码版本/commit、config hash、taxonomy/router/reward version。
- [ ] 轨迹支持脱敏导出，移除 API key 和用户敏感附件内容。

## 11.5 Replay

- [ ] `replay --verify` 验证 hashes、parent chain 和 stored rewards。
- [ ] deterministic checker 可离线重跑并与原结果比较。
- [ ] neural checker 默认读取缓存结果；显式 `--rejudge` 才发起新调用。
- [ ] renderer/version 不一致时报告 non-comparable，不静默覆盖。
- [ ] 支持从任意 artifact 创建 branch episode，用于 counterfactual repair。

## 11.6 RL adapters

- [ ] 提供轻量 Python client，不引入 trainer 框架依赖到主包。
- [ ] 提供 Gymnasium-like adapter 时放到 optional dependency 或独立模块。
- [ ] 支持 synchronous single-env baseline。
- [ ] 支持 async vectorized episodes，控制 browser/model 并发。
- [ ] 将 observation 中的 artifact URL 转换为 trainer 可访问的内容。
- [ ] 支持 external policy：trainer 通过 API 决定 action。
- [ ] 支持 internal policy：Slidex 调用配置的 OpenAI-compatible policy endpoint。

## 11.7 单页 repair benchmark

- [ ] 从 clean HTML 注入 G2–G7/S3 可控缺陷。
- [ ] 保留 clean twin、defective artifact 和 mutation manifest。
- [ ] 只保留最终 render 中真实可见/可测的 mutations。
- [ ] 评估 repair success、new-defect rate、steps、tokens 和 wall time。
- [ ] 划分 development/test，冻结 critic 后才跑 test。

### Phase 11 退出条件

- [ ] 外部 trainer 可通过 HTTP 完成 reset → step → reward → done。
- [ ] episode 可回放，reward 可审计。
- [ ] OpenAI-compatible policy endpoint 可作为 internal policy 插入同一环境。

---

# Phase 12：品牌、配置和兼容迁移

## 12.1 包与命令

- [ ] 将项目展示名称改为 Slidex。
- [ ] 在 `pyproject.toml` 增加 `slidex = "deeppresenter.cli:main"`。
- [ ] 暂时保留 `pptagent` 为兼容 alias，并输出 deprecation 提示策略。
- [ ] 将 API server 命令命名为 `slidex serve-api`。
- [ ] 保留 `pptagent-mcp` 仅服务 legacy backend；若新 MCP 需要入口，使用独立 `slidex-mcp`。
- [ ] 更新 package description、keywords 和默认 workspace 环境变量。

## 12.2 配置路径

- [ ] 引入 `SLIDEX_WORKSPACE_BASE`，兼容读取旧 `DEEPPRESENTER_WORKSPACE_BASE`。
- [ ] 引入 Slidex config directory，提供旧配置迁移而非静默丢失。
- [ ] 配置输出隐藏 API key。
- [ ] onboarding 默认生成 Docker-free MCP 配置。
- [ ] outbound policy/critic endpoints 均采用 OpenAI-compatible `base_url/model/api_key`。

## 12.3 日志与观测

- [ ] 日志名称从 DeepPresenter 迁移为 Slidex，同时兼容旧 history reader。
- [ ] 每个 request/episode/step/artifact 使用关联 ID。
- [ ] 记录 inspector 和 tool timing。
- [ ] 记录 LLM usage 和 estimated cost，但不将 provider 定价硬编码为 reward truth。
- [ ] API 日志不输出完整附件、base64 图片或 secret。

## 12.4 Legacy compatibility

- [ ] legacy `ConvertType.PPTAGENT` 仍能走原 template backend。
- [ ] legacy backend 输出也可包装为 artifact，并运行有限的 final critic。
- [ ] 明确 legacy PPTX 缺少完整 HTML native IR 时的 trust downgrade。
- [ ] 不为追求统一而重写 `pptagent/` 全部内部模块。

### Phase 12 退出条件

- [ ] 新用户看到和调用的是 Slidex。
- [ ] 旧 CLI/配置有可控兼容路径。
- [ ] legacy backend 不阻塞 Slidex IR/critic/RL 主线。

---

# Phase 13：测试矩阵、性能与安全

## 13.1 Unit tests

- [ ] schema、hash、artifact lineage。
- [ ] geometry/style/terminology linters。
- [ ] router 和 trust policy。
- [ ] reward gates 和 delta reward。
- [ ] local filesystem tools 和 path traversal。
- [ ] OpenAI request/response serialization。
- [ ] episode state machine 和 idempotency。

## 13.2 Browser/export tests

- [ ] DOM extraction fixtures。
- [ ] browser determinism 重复测试。
- [ ] HTML → PPTX strict validation。
- [ ] PPTX re-render 和 render-fidelity。
- [ ] missing font/image、JS error、network timeout。
- [ ] 不同 aspect ratio。

## 13.3 LLM tests

- [ ] fake server 覆盖所有错误和 structured response。
- [ ] 少量真实 provider smoke tests，使用 marker 和环境变量。
- [ ] atomic vs whole-rubric 对照。
- [ ] AB/BA reference order control。
- [ ] defer/abstain 行为。

## 13.4 API contract tests

- [ ] OpenAI Python SDK sync/async。
- [ ] SSE chunk 顺序与 `[DONE]`。
- [ ] auth、rate limit、unknown model、invalid artifact。
- [ ] multipart upload/download。
- [ ] client disconnect/cancel。
- [ ] 同一 episode 并发 step 冲突。

## 13.5 性能

- [ ] browser/context pooling，避免每个 inspector 重启 Chromium。
- [ ] 同一 artifact 的 IR/render/inspection 按 hash 缓存。
- [ ] symbolic inspectors 并行运行。
- [ ] neural calls 按 class 和 artifact hash 缓存。
- [ ] 限制并发模型调用、浏览器页面和导出进程。
- [ ] 记录 p50/p95 latency 和每 episode cost。
- [ ] 不通过跳过 hard inspection 来优化延迟。

## 13.6 本地源码执行安全

- [ ] 明确 Docker 移除后不是强隔离执行环境。
- [ ] API 默认不向不可信公网用户开放任意 `run_command`。
- [ ] internal agent tool 与 public API capability 分离。
- [ ] workspace path resolver 防止 `..`、symlink 和绝对路径逃逸。
- [ ] 命令 timeout、输出上限和进程组清理。
- [ ] 上传文件名清洗，压缩包防 zip-slip/zip-bomb。
- [ ] API key 使用 constant-time compare 或成熟 auth middleware。
- [ ] 对生产多租户场景预留外部 sandbox runner 接口，但主项目不依赖 Docker。

## 13.7 CI 分层

- [ ] PR 默认运行 unit + API fake-server tests。
- [ ] browser tests 在具备 Chromium 的 job 运行。
- [ ] export tests 在具备 Node/Poppler/LibreOffice 的 job 运行。
- [ ] real LLM tests 手动或定时运行，不进入普通 PR 门禁。
- [ ] benchmark 和 frozen evaluation 独立运行并保存版本化结果。

### Phase 13 退出条件

- [ ] 无凭证 CI 可验证绝大多数核心逻辑。
- [ ] browser/export/LLM 失败可以区分依赖缺失与代码回归。
- [ ] public API 不直接暴露任意本地命令能力。

---

# Phase 14：论文方法复现实验与发布门禁

## 14.1 方法复现

- [ ] 构建 matched clean/defective pairs。
- [ ] 对 A=image、B=IR、C=image+IR 条件运行 attribution。
- [ ] 对 whole-rubric、named whole-rubric、atomic query、reference pair 做对照。
- [ ] 对 repeated whole-rubric 做预算控制。
- [ ] 统计 detection、specificity、balanced accuracy、localization 和 defer rate。
- [ ] 明确 synthetic、real-layout、open-world image-only 三种输入信任级别。

## 14.2 Frozen critic evaluation

- [ ] 使用 development set 选择 inspector 和阈值。
- [ ] 写出 router v1 executable specification。
- [ ] 冻结 taxonomy/router/prompt/checker/reward versions。
- [ ] 在 disjoint test set 运行一次正式结果。
- [ ] 神经 transfer 失败和 reference unresolved 按原样报告，不 post-hoc reroute。
- [ ] 单独报告 trusted native-IR classes 与 neural classes。

## 14.3 Agentic RL readiness gate

- [ ] reward 不依赖隐藏标签泄漏。
- [ ] mutation 通过 final render fidelity 检查。
- [ ] critic 输出对 policy 不泄漏 clean answer，除非任务明确允许 reference。
- [ ] 环境支持固定 seed、版本和 replay。
- [ ] observation/action/reward schema 冻结为 v1。
- [ ] 单页 repair baseline 能稳定训练/评估。
- [ ] 报告 reward hacking probes。

## 14.4 发布验收

- [ ] 全仓无运行时 Docker 依赖。
- [ ] Slidex CLI 可完成 generate、inspect、repair、serve-api。
- [ ] OpenAI SDK 可调用 generate/critic/repair。
- [ ] native episode API 可完成单页 repair episode。
- [ ] 最终 PPTX 经过 re-render validation。
- [ ] 每个输出有完整 artifact manifest、inspection report 和 reward breakdown。
- [ ] legacy `pptagent` 路径仍有明确兼容状态。

---

# 15. 推荐实施批次与依赖关系

## Batch A：可运行基础（Phase 0–1）

- [ ] 基线测试。
- [ ] Docker SDK、Docker MCP、Docker onboarding 清除。
- [ ] local filesystem tools。
- [ ] Docker-free 单页生成和导出。

**阻塞关系：** 未完成 Batch A，不开始 API/RL server；否则 server 会继承不可控 Docker 生命周期。

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

## Batch E：服务化和 RL（Phase 10–11）

- [ ] OpenAI-compatible API。
- [ ] Native episode API。
- [ ] trajectory/replay。
- [ ] single-slide repair environment。

**里程碑：** 外部 OpenAI SDK 和 RL trainer 均能调用同一 Slidex backend。

## Batch F：迁移、性能与发布（Phase 12–14）

- [ ] 品牌和兼容迁移。
- [ ] CI、性能、安全。
- [ ] frozen benchmark 和发布门禁。

---

# 16. 第一轮可直接执行的文件级 TODO

## `pyproject.toml`

- [ ] 删除 `docker` dependency。
- [ ] 增加 `slidex` console script。
- [ ] 增加 API/RL 新包的 package data（仅确有非 Python 资源时）。
- [ ] 增加/整理 pytest markers：browser、export、api、rl。

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
- [ ] 分离 model server 与 Slidex API server 命令。
- [ ] 后续增加 inspect、repair、serve-api 命令。

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

# 17. 明确不做的事情

- [ ] 第一版不训练 learned critic router。
- [ ] 第一版不把所有 aesthetic quality 压成单一 reward model。
- [ ] 第一版不从 PNG 恢复 boxes 后宣称等价于 native IR。
- [ ] 第一版不重写整个 legacy `pptagent/`。
- [ ] 第一版不同时实现 Chat Completions 和 Responses API 的半兼容版本；先把前者做完整。
- [ ] 第一版不提供公网多租户任意 shell execution。
- [ ] 第一版不把 Docker 改成“可选但默认探测”；运行时应彻底不触碰 Docker。
- [ ] 第一版不在 export 失败后把 PDF fallback 伪装成 PPTX 成功。
- [ ] 第一版不将 inspector 的 `defer` 当作零缺陷。

---

# 18. Definition of Done

只有同时满足以下条件，完整改造才算完成：

- [ ] **Docker-free**：安装、onboarding、generation、inspection、export、API 和 RL episode 均不需要 Docker CLI/daemon/image。
- [ ] **Source-aware**：Slidex 保存 declared IR、computed IR 和最终 render，并明确其证据边界。
- [ ] **Paper-grounded critic**：实现 symbolic、atomic neural、semantic、reference-assisted 四类检查，并由冻结 router 分工。
- [ ] **Structured diagnosis**：每个 defect 有 class、status、evidence、localization、severity 和 repair hint。
- [ ] **Final-artifact validation**：PPTX 重新渲染，mutation/render fidelity 被检查。
- [ ] **Repair loop**：生成和修订形成可审计的 parent-child artifact 链。
- [ ] **Verifiable reward**：reward vector、hard gates、delta reward 可离线复算。
- [ ] **OpenAI compatibility**：官方 OpenAI Python SDK 可调用 models、chat completions、streaming 和 tool calls。
- [ ] **Agentic RL ready**：native episode API 支持 reset/step/reward/done、trajectory 和 replay。
- [ ] **Reproducible evaluation**：critic/router/reward 版本冻结，开发集与测试集分离，defer 和 transfer failure 如实保留。
