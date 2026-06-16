## Why

InfoLens 现在的生成路径（Chat / Causal Flow）只呈现模型**最终走的那一条**贪心续写。但生成式模型在每一步其实都手握一个 top-k 概率分布——每一步都是"无数条没走的路"分叉出去。学习笔记和 canvas 里都点出这层遗憾：用户**看不到模型本来可以走向哪些未来**。把这条遗憾做成可触摸的探索器，是"可交互机制探针"主线（消融 → Logit Lens → 分叉树）的第三阶段，也是最适合对外演示的一环。

生成分叉树让用户在每个生成位置点开任意 top-k 候选、从那里开出一条新分支继续生成，把"模型本来可以走向哪些未来"变成一棵可点开、可续写的概率树。

## What Changes

- 新增后端能力：给定一段已确定的前缀（prompt + 之前选定的候选 token 序列），返回**下一个位置**的 top-k 候选 token 及其概率；分叉 = 把"前缀 + 选中的候选 token"当作新前缀，再调一次本能力。这是分叉树的原子操作。
- 新增 HTTP 端点 `POST /api/branch-next`：请求体含 `prefix`、`model`、`source_page`、可选 `top_k`，返回该位置 top-k 候选的 token 文本 / token id / 概率，以及 prefix 的 token 数与是否触上下文上限。复用推理锁、超长校验（`PromptTooLongError`）、`source_page` 校验等基础设施。
- 实现采用**单步前向、无生成循环**：编码 prefix → 一次 `forward(use_cache=False)` → 末位 logits → softmax → top-k。不调用 `model.generate`、不开生成循环、不反传，单次请求即一次前向，轻量且快。
- 前端在现有 Causal Flow 页（`client/src/pages/causal_flow/index.ts` 的 d3 DAG 基础上）新增"分叉树"模式：点一个 token 节点即展开其 top-k 候选，点候选即开新分支续写，形成树。复用现有 DAG 的节点/连线渲染，但用**树形布局**（dagre / d3 hierarchy）替代 DAG 布局。

## Capabilities

### New Capabilities
- `generation-branch-tree`: 对一段已确定前缀做单步 next-token top-k 展开，作为生成分叉树的原子操作——前端把多次调用串成"从任意点开岔路、续写"的概率树。

### Modified Capabilities
<!-- 无：现有 Causal Flow / completion 行为不变；前端在 causal_flow 页加新模式属实现细节，无既有 spec。 -->

## Impact

- **新增后端代码**：`backend/core/branch_next.py`（单步 top-k 算法）、`backend/api/branch_next.py`（handler）。
- **接线**：`server.py` 导入 handler；`server.yaml` 增加 `/branch-next` 路径（以 `/prediction-attribute` 段为模板，响应含 top-k 数组）。
- **复用**：`backend/core/completion_generator.py` 的 `ensure_slot_ready` / 上下文上限校验（`PromptTooLongError`）；`backend/core/next_token_topk.py` 的 top-k 解码；`backend/models/model_manager` 的槽位与推理锁；`backend/platform` 的 OOM / 日志 / `DeviceManager.clear_cache`。
- **前端**：`client/src/pages/causal_flow/index.ts`（新增分叉树模式 + 树形布局）、`shared/api/GLTR_API.ts`（新增 `branchNext` 客户端方法）、对应 i18n。
- **性能/内存**：单次请求 = 一次前向（无生成循环、无反传、无 scores 缓存），比现有 completion 轻得多；prefix 长度受上下文上限约束（沿用 `PromptTooLongError`）。无新增第三方依赖；前端树深/分支数需设上限以防爆炸（见 design）。
