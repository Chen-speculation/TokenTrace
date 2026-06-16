## Context

现有 `backend/core/prediction_attributor.py` 对 next-token 预测做梯度归因：编码 `context` → 取 embedding 为可微输入 → 前向取末位 logits → `logits[target].backward()` → 输入各 token embedding 梯度的 L2 范数为归因分。它衡量的是局部敏感度（相关性）。

本设计新增一个**因果对照**通道：不再求导，而是对每个输入 token 做遮挡（occlusion），重算目标概率，量出实际效果 ΔP。两者在同一组 token 坐标上并排，构成"相关性 vs 实际扰动"的教学对照。约束：单机 Qwen3-0.6B，CPU / MPS / CUDA 皆可；单用户串行（共享 `inference_lock`）；归因输入有 token 上限。

## Goals / Non-Goals

**Goals:**
- 对给定 `context` 与目标 token，输出逐输入 token 的因果影响量 ΔP（baseline 目标概率 − 遮挡该 token 后的目标概率），可选 Δlogit。
- token 坐标与梯度归因 1:1 对齐（同一 offset 体系），支持前端同屏对照。
- 复用现有目标选择（top-1 / `target_prediction` / `target_token_id`）、模型槽位、推理锁、超长校验、`source_page` 校验。
- 单次请求仅一轮推理量级（批处理），无需反向传播。

**Non-Goals:**
- 真删除（length-changing）遮挡——v1 用定长 embedding 遮挡（见 Decisions），删除模式列为后续扩展。
- 指定替换词的反事实（"中国→日本"式具体换词）——v1 用中性基线，定向换词为后续扩展。
- 多 token 联合遮挡 / 交互效应（Shapley 类）。
- 其他归因方法（Integrated Gradients 等）——属"归因方法擂台"后续 change。
- Causal Flow 的多步消融——本期只做单次 next-token 预测。

## Decisions

### D1：遮挡方式 = 定长 embedding 遮挡（occlusion），而非删除
对每个输入位置 i，复制 baseline 的 `inputs_embeds`，把第 i 行替换为基线向量，序列长度与位置编码不变。
- **为什么**：删除会移动后续 token 的位置、改变长度，使 offset 与梯度归因无法对齐，且 RoPE 位置变化引入额外混杂；定长遮挡保证与梯度视图同坐标、可并排。
- **Alternatives**：真删除（最贴近"反事实"直觉，但坐标不对齐、解释更混杂）→ 列为 Non-Goal/后续；token 替换为特定词 → 后续定向反事实。

### D2：基线向量 = 全序列 embedding 均值（mean-embedding baseline）
被遮挡位置替换为当前输入各 token embedding 的均值向量。
- **为什么**：零向量在分布外、易给出夸大且不稳定的 ΔP；均值向量更"中性、在分布内"，是遮挡类归因常见基线。基线策略集中在一处常量，便于后续做成可配置项。
- **Alternatives**：零向量（最简单但 OOD）；pad/eos token 的 embedding（语义偏置强）；mask token（Qwen 无原生 mask）。均值为 v1 默认，其余留作可配置扩展。

### D3：批处理一次前向
构造 batch = baseline(1) + 遮挡变体(N)，形状 `[N+1, seq, hidden]` 的 `inputs_embeds`，一次 `forward(use_cache=False, output_attentions=False)`，取各样本末位 logits 的目标概率。
- **为什么**：等价 N+1 次"留一"前向但只一次 kernel launch；无需反传，显存峰值低于梯度归因。
- **分批**：当 `(N+1) × seq` 超过内存阈值时按行分批多次前向累积；阈值与现有 chunk 配置风格一致，放运行时配置。
- **目标 token id 解析**：先用 baseline 那一行的末位 logits 按现有逻辑解析 target（top-1/encode/显式 id），再对所有行取同一 target id 的概率。

### D4：新增独立端点 `POST /api/ablation-attribute`
请求体与 `/prediction-attribute` 同构：`context`、`model`（base/instruct）、`target_prediction?`、`target_token_id?`、`source_page`、`flow_id?`、`flow_step?`。响应：`success`、`model`、`target_token`、`target_prob`（baseline）、`token_attribution: [{offset,raw,score}]`（score=ΔP，可正可负）、可选 `delta_logit`、`debug_info{topk_tokens,topk_probs}`、`is_eos`。
- **为什么**：与梯度归因语义不同（score 含义、可为负），独立端点比给旧端点加 mode 更清晰，且前端可并行请求两条做对照。
- **接线**：`server.py` 导入 + `server.yaml` 仿 `/prediction-attribute` 段；`backend/api/ablation_attribute.py` 复用 `prediction_attribute.py` 的参数校验与日志骨架。

### D5：前端落在现有 Attribution 页
`attribution inspector` 增加方法切换（Gradient / Ablation / Both）。Both 模式并排两套 token 配色，并显示一致性读数（如两方法归因排名的 Spearman 相关、Top-k 重合度）。score 可正可负，Ablation 用发散色阶（正=该 token 支撑目标，负=抑制目标）。
- **为什么**：reviewer 明确建议落在现有页、复用 `pages/attribution/index.ts` 与 inspector，改动最小、对照最直观。

## Risks / Trade-offs

- [遮挡 ≠ 真删除/真反事实] → 文档与 UI tooltip 注明"定长遮挡，基线=均值 embedding"，把 ΔP 解释为"该位置被中性化后的效果"，不宣称等于真实因果机制；定向换词/删除作为后续。
- [均值基线仍可能 OOD，长文里被遮挡位贡献小导致 ΔP 噪声] → baseline 集中可配置，design 给出零向量/pad 备选；UI 显示 baseline 目标概率帮助判读。
- [长上下文 batch 显存] → 自动分批 + 沿用归因 token 上限；`use_cache=False`、不反传，峰值可控。
- [SDPA/量化下的数值差异] → 与现有归因一致：CPU float32 精确，半精度（MPS/CUDA）允许约 1% 量级差；INT8 不支持时按现有方式报错/降级。
- [概率饱和时 ΔP 偏小] → 同屏展示 Δlogit 作为补充，缓解 softmax 饱和导致的判读困难。

## Open Questions

- 基线默认值最终选"均值 embedding"还是"零向量"？（v1 暂定均值，可配置）
- 一致性读数用 Spearman 还是 Top-k 重合度，或两者都给？
- `target_prediction` 多 token 时仅取首 token（与现有一致）是否够用，还是需支持整段目标概率？
