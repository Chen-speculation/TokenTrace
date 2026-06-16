## Context

现有生成路径 `backend/core/completion_generator.py` 的 `core_generate_from_text` 是续写唯一入口：它用 `model.generate(return_dict_in_generate=True, output_scores=True)` **走完一整段**贪心/低温续写，返回文本 + 每步 `pred_topk`（在 `_build_generated_bpe_strings` 里拼装）。也就是说模型**每一步的 top-k 候选其实已经算出来了**，只是被绑死在"一条线性续写"里，且 `model.generate` 一旦启动就自己走完，无法中途插嘴"换个候选"。

Causal Flow 页（`client/src/pages/causal_flow/index.ts`，2785 行）已有完整的 d3 DAG 渲染：节点、连线、逐步回放、`dagHandle`。但它是 DAG（同 token 可被多前驱复用），布局是图布局；分叉树是严格的树（每个节点唯一父节点），需要换树形布局。

约束：单机 Qwen3-0.6B，CPU / MPS / CUDA；单用户串行（共享 `inference_lock`）；prefix 长度受模型上下文上限约束。

## Goals / Non-Goals

**Goals:**
- 提供"单步 next-token top-k"原子操作：输入已确定 prefix，输出下一个位置的 top-k 候选（token 文本 / token id / 概率）+ prefix token 数。
- 让分叉 = 重复调用该原子操作（prefix + 选中的候选 token → 新 prefix），由前端串成树。后端无状态、不存树。
- 复用现有模型槽位、推理锁、上下文上限校验（`PromptTooLongError`）、`source_page` 校验。
- 单次请求 = 一次前向，无生成循环、无反传、无 scores 缓存。

**Non-Goals:**
- **服务端树状态管理 / 树持久化**——v1 后端无状态，树完全由前端构造与缓存；session 持久化留后续。
- **采样续写（do_sample）**——v1 每步只返回分布 top-k，"选哪个"由用户点，不涉及随机采样。
- **多 token 联合候选 / beam search**——v1 每次只展开单个位置。
- **自动展开整棵树**——v1 由用户逐个点开分支；自动展开 N 层留后续（防爆炸）。
- **在 Chat 页接入分叉**——v1 落 Causal Flow 页；Chat 页不动。
- **替换现有 `core_generate_from_text`**——completion 链路不变，分叉树走独立端点。

## Decisions

### D1：算法 = 单步前向，不调 model.generate
对 prefix 编码 → 一次 `forward(use_cache=False, output_attentions=False)` → 取末位 logits → softmax → `torch.topk(logits, k)`。返回 top-k 的 token 文本 / token id / 概率。
- **为什么**：`model.generate` 会自己走完一整段且把 top-k 绑在线性序列里，无法"只展开一步"。直接前向取末位 logits 是最小、最快、最可控的做法——一次 kernel launch、无 scores 元组、无反传。
- **复用**：top-k 解码复用 `next_token_topk.decode_topk_ids_to_strings_and_rounded_probs`（现有归因/消融已用）。
- **Alternatives**：① 用 `core_generate_from_text(max_tokens=1)` 复用——但它内部仍走 `model.generate` 全套（streamer / stopping criteria / scores 拼装），开销大且语义错位（它是"续写一步"，不是"展开候选"）；② 在 generate 里 hook 取每步分布——过度工程。直接前向最优。

### D2：上下文上限校验复用 PromptTooLongError
prefix token 数 + 1（要展开的下一位置）不得超过模型上下文上限。复用 `completion_generator` 的 `completion_max_token_length` / `_model_context_token_limit` 与 `PromptTooLongError`。
- **为什么**：现有 completion 已有成熟的上下文上限处理与错误类型，前端也认这个错误；分叉树是 completion 的"单步版"，语义一致。
- **边界**：当 prefix 已占满上下文（无剩余位置）时，返回明确错误（`PromptTooLongError`），前端提示"已到上下文末尾，无法再展开"。

### D3：新增独立端点 `POST /api/branch-next`
请求体：`prefix`（string）、`model`（base/instruct）、`source_page`、可选 `top_k`（默认 `DEFAULT_NEXT_TOKEN_TOPK=10`，上限硬封顶如 50 防巨型响应）。
响应：
```
{
  "success": true,
  "model": str,
  "prefix_tokens": int,
  "candidates": [
    { "token": str, "token_id": int, "prob": float }, ...
  ],
  "is_context_full": bool   // prefix 是否已占满上下文（前端据此禁用展开）
}
```
- **为什么独立端点**：与 completion（整段续写）、归因（输入→输出影响）语义都不同；独立端点比给旧端点加 mode 清晰。
- **接线**：`server.py` 导入 + `server.yaml` 仿 `/prediction-attribute` 段；`backend/api/branch_next.py` 复用 `prediction_attribute.py` 的参数校验与日志骨架。
- **为什么返回 token_id**：前端选中某候选后，新 prefix = 旧 prefix + 该候选的**文本**（用户可读），token_id 仅用于展示与调试。重新 tokenize 由后端在下一次请求做（与 completion 的"解码→重编码"一致）。

### D4：前端落 Causal Flow 页，加"分叉树"模式
现有页有 DAG 模式。新增**分叉树模式**（模式切换，不与 DAG 冲突）：
- 用户输入初始 prefix → 调 `/api/branch-next` → 渲染根节点 + 其 top-k 候选为可点叶子。
- 点某候选 → 新 prefix = 旧 prefix + 候选文本 → 再调 `/api/branch-next` → 渲染新子节点。
- **布局**：树形布局（d3 hierarchy / dagre tree），左到右或上到下，与 DAG 的图布局区分。
- **复用**：节点/连线的 d3 渲染、tooltip、颜色编码（概率→色阶）复用现有 DAG 代码；新增树形数据结构（`TreeNode`）与逐节点懒加载。
- **防爆炸**：硬上限——树深 ≤ N（如 12）、每节点展示 top-k ≤ K（如 5，可配）、总节点数 ≤ M（如 200）；超限前端拒绝展开并提示。

### D5：前端缓存与取消
每展开一个节点就一次请求；用户快速点击会触发多次。处理：
- 用 `AbortController` 取消进行中的请求（与现有归因页一致）。
- 节点级缓存：同一 prefix 的展开结果缓存（LRU），避免重复点同一节点重复请求。

## Risks / Trade-offs

- [请求放大：深树 = 多次串行请求] → 后端单次前向轻量（~数十毫秒 CPU），串行点开可接受；自动展开整棵树（Non-Goal）若后续做，需后端批量化或限流。
- [解码→重编码不一致：候选文本重新 tokenize 可能不等于原 token_id] → 与现有 completion 一致的已知行为；前端只把候选**文本**拼进新 prefix，token_id 不参与拼接，避免 id 错位。
- [树深爆炸 / DOM 爆炸] → D4 硬上限（深/宽/总节点数）+ 懒加载（只渲染可见区域或已展开节点）。
- [prefix 过长导致每次前向慢] → 受上下文上限约束（D2）；前端展示当前 prefix_tokens，接近上限时提示。
- [BPE 候选文本含前导空格/特殊字符，拼接后语义变化] → UI 在候选展示与拼接时保留原始 token 文本（`skip_special_tokens=False`），与 completion 一致。
- [半精度数值差异] → 与现有归因一致：CPU float32 精确，MPS/CUDA 半精度 top-k 排序可能有边界扰动，可接受。

## Migration Plan

- 纯新增端点 + 前端新模式，无对现有功能的破坏性改动，无需迁移。
- 部署顺序：先后端（`/api/branch-next` 可独立验证）→ 再前端（可先 mock 契约开发，后端就绪后联调）。
- 回滚：删除新端点与前端模式即可，不影响 completion / Causal Flow DAG。

## Open Questions

- `top_k` 上限封顶值定多少？（v1 暂定默认 10、上限 50）
- 树深/总节点数硬上限定多少？（v1 暂定深 12、总节点 200，可配）
- 候选拼接 prefix 时，是否需要在前端做"前导空格归一化"以匹配模型期望？（v1 暂不做，保留原始 token 文本）
