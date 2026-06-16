## Context

现有 `backend/core/prediction_attributor.py` 与刚归档的 `ablation_attributor.py` 都只看"输入 embedding → 末位最终 logits"两个端点。**中间 L 层（Qwen3-0.6B 约 28 层）的 hidden states 完全没用上**——用户看不到"答案是怎么一层层成形的"。

Logit Lens（nostalgebraist 2020, nnostalgebraist)的经典做法：把第 `ℓ` 层残差流的末位向量，套上模型**最后一层之后**的 final norm（RMSNorm）和 lm_head，得到"如果模型此刻就停在第 ℓ 层、直接输出，它会预测什么"。Qwen3 是标准 HF `Qwen3ForCausalLM`，结构为 `model: Qwen3Model`（含 `norm: RMSNorm`）+ `lm_head: Linear`，与 Logit Lens 假设完全契合。

约束：单机 Qwen3-0.6B，CPU / MPS / CUDA 皆可；单用户串行（共享 `inference_lock`）；归因输入有 token 上限（沿用 `ATTRIBUTION_MAX_TOKEN_LENGTH=500`）。

## Goals / Non-Goals

**Goals:**
- 对给定 `context`，输出每一层（含 embedding 层 ℓ=0 与最终层 ℓ=L）末位 hidden state 过 norm+lm_head 后的 top-k 候选 token 与概率。
- 对目标 token（top-1 / `target_prediction` 首 token / `target_token_id`），输出它在各层的概率轨迹 `[ℓ=0..L]`，构成"答案成形曲线"。
- 复用现有目标选择、模型槽位、推理锁、超长校验、`source_page` 校验。
- 单次请求仅一轮前向，**无需反向传播**。

**Non-Goals:**
- **Tuned Lens / Depth-wise affine transform**——Logit Lens 不学习任何校准层；加 transform 属后续扩展。
- **非末位 token 的逐层轨迹**（如整个序列每层的预测）——v1 只看 next-token 预测的最后一个位置，与现有归因口径一致。
- **Decoder-block 中间投影**（每个 transformer block 内部）——v1 只在每层残差流出口投影。
- **Causal Flow 的多步 Logit Lens**——本期只做单次 next-token 预测，不做生成序列的逐层轨迹。
- **多头 / per-head 视图**——v1 只做每层一个聚合投影。

## Decisions

### D1：投影方式 = 每层末位 hidden → model.norm → lm_head
对 `output_hidden_states` 返回的 `hidden_states: tuple[Tensor]`（长度 L+1，下标 0 是 embedding 输出、L 是最终层输出），取每个的 `[:, -1, :]`（末位、shape `[hidden]`），依次过 `hf_model.model.norm`（Qwen3 的 RMSNorm）与 `hf_model.get_output_embeddings()`（即 `lm_head`）。
- **为什么**：这是 Logit Lens 的标准定义；`model.norm + lm_head` 是 Qwen3 完成最终层之后本就要做的事，把它"提前套到中间层"得到的 logits 即"此刻截断输出会预测什么"。
- **属性访问**：norm 经 `hf_model.model.norm`（Qwen3ForCausalLM → Qwen3Model.norm），lm_head 经标准 HF 接口 `hf_model.get_output_embeddings()`。两者在实现时各 `hasattr` 兜底（万一不同 HF 版本属性名差异），找不到则报清晰错误。
- **Alternatives**：Tuned Lens（学一个 per-layer 校准层，更准但需训练、引入参数）→ Non-Goal；直接对 raw hidden 取 argmax（不经 norm+lm_head，得到的是无意义隐藏维 argmax）→ 错误做法，不采用。

### D2：target token 在各层的概率 = softmax(逐层 logits)[target_id]
对每层投影出的 logits 做一次 softmax（全词表），取目标 token id 的概率。最终层（ℓ=L）的概率必须与现有 `/prediction-attribute` 返回的 `target_prob` 在数值上一致（同一次前向、同一位置、同一投影），作为正确性自检。
- **为什么**：用 softmax 概率而非 raw logit，便于跨层比较与前端折线可视化（概率有界 [0,1]）。
- **Alternatives**：报 raw logit（无界、跨层难比较）→ 作为 debug 字段可选附，主信号用概率。

### D3：内存控制 = 逐层投影、即时取 top-k、不物化 L×vocab
关键风险是 `L × vocab` 的中间张量（28 层 × ~150k 词表）。控制手段：
1. `forward(output_hidden_states=True, use_cache=False)`，拿到 `hidden_states` tuple（`L+1` 个 `[1, seq, hidden]`）。
2. **逐层循环**：取 `hidden_states[ℓ][:, -1, :]`（`[1, hidden]`）→ norm → `lm_head` → `[1, vocab]` → `softmax` → `topk(k)` → 只保留 top-k 的 token id / prob / 字符串 + 目标 token 概率，**立即 `.cpu().tolist()` 后丢弃该层 logits 张量**。
3. 全程不 `torch.cat` 出 `[L, vocab]`；最终内存里只有 `L × k` 个标量。
- **为什么**：`output_hidden_states` 本身只多存 hidden（`L × seq × hidden`，0.6B 量级 ~28×500×1024 ≈ 14M floats ≈ 56MB float32，可接受），真正会爆的是 `[L, vocab]`，逐层处理即可规避。
- **分批/截断**：v1 不对 seq 分批（500 token 上限下 hidden 内存可控）；若未来放开上限，按现有 chunk 风格分批。

### D4：新增独立端点 `POST /api/logit-lens`
请求体与 `/prediction-attribute` / `/ablation-attribute` 同构：`context`、`model`、`target_prediction?`、`target_token_id?`、`source_page`、`flow_id?`、`flow_step?`。响应：
```
{
  "success": true,
  "model": str,
  "target_token": str,
  "n_layers": int,                    # L（不含 embedding 层）
  "final_target_prob": float,         # = 最终层目标概率，与 /prediction-attribute 一致（自检锚点）
  "layers": [                         # 长度 L+1，下标 0=embedding 层，L=最终层
    {
      "layer": 0..L,
      "is_embedding": bool,           # ℓ==0
      "topk_tokens": [str, ...],      # 该层投影后 top-k
      "topk_probs":  [float, ...],
      "target_prob": float            # 目标 token 在该层的概率（可为 0）
    }, ...
  ],
  "debug_info": {"topk_tokens": [...], "topk_probs": [...]},  # 最终层 top-k（与现有归因同形，便于复用 UI）
  "is_eos": bool
}
```
- **为什么独立端点**：响应形状（逐层结构）与梯度/消融归因（逐输入 token）正交，独立端点比给旧端点加 mode 更清晰。
- **接线**：`server.py` 导入 + `server.yaml` 仿 `/prediction-attribute` 段；`backend/api/logit_lens.py` 复用 `prediction_attribute.py` 的参数校验与日志骨架。
- **k 值**：top-k 默认沿用 `DEFAULT_NEXT_TOKEN_TOPK`（10）；不做请求参数（v1 收敛口径）。

### D5：前端落在现有 Attribution 页，作为第三个面板
现有页已有 Gradient / Ablation / Both 三态。Logit Lens 新增一个**独立面板**（不与三态互斥，可同屏并存，因为它看的是"层"维度而非"输入 token"维度）：
- **逐层 top-k 热力图**：行=层（0..L），列=top-k 槽位，单元格=token 字符串 + 概率，颜色编码概率。
- **目标 token 概率折线**：横轴=层，纵轴=目标 token 概率，标出"首次进入 top-1 的层"。
- 复用 inspector 的 token tooltip 体系。
- **为什么落现有页**：reviewer 在 canvas 里明确建议复用 `pages/attribution/index.ts` 与 inspector；与梯度/消融归因同上下文（同一 context、同一 target），对照最直观。

## Risks / Trade-offs

- [Logit Lens ≠ 真实逐层决策] → 文档与 UI tooltip 注明"这是把 final norm+lm_head 提前套到中间层的**投影读数**，是一种解释性工具，不等于模型真实的逐层计算意图"；不宣称等于内部机制。
- [浅层投影常落到无意义高频词] → 正常现象（Logit Lens 已知特性），UI 给出"首次目标进入 top-k 的层"作为可读摘要，避免被浅层噪声淹没。
- [norm 在 lm_head 之前 vs Qwen3 的具体 norm 位置] → 已核对 Qwen3 结构：`Qwen3ForCausalLM` forward 内部是 `hidden = model(input_ids); hidden = model.norm(hidden); logits = lm_head(hidden)`，故对每层 hidden 套 `model.norm` 再 `lm_head` 与最终层口径一致。实现时加自检：最终层投影出的 top-1 必须等于 `output.logits[:, -1, :].argmax()`。
- [hidden states 在 MPS/CUDA 的内存] → 500 token × 28 层 × 1024 hidden ≈ 56MB float32，可控；`use_cache=False`、逐层处理避免 `[L,vocab]` 物化（D3）。
- [半精度数值差异] → 与现有归因一致：CPU float32 精确，MPS/CUDA 半精度允许约 1% 量级差；最终层自检在半精度下用 `torch.isclose` 容差而非严格相等。
- [vocab 很大（~150k），每层 lm_head 全词表乘法] → 单层是 `[1,hidden] @ [hidden, vocab]`，0.6B 下约 1024×150k ≈ 1.5亿次乘法 ×28 层，CPU 上约百毫秒级，可接受；如太慢，未来可只投影目标 token + top-k 区域（需两趟）。

## Open Questions

- top-k 默认 10 是否够？层很多时热力图列数是否需要可配？（v1 暂定 10，可配置留后续）
- 目标 token 折线的"关键层"摘要，用"首次进 top-1"还是"首次进 top-3"或"概率超过 0.5 的首层"？（v1 暂给"首次进 top-1"，其余作 tooltip）
- 是否需要把 Logit Lens 也接到 Causal Flow 的每一步？（明确 Non-Goal，留后续）
