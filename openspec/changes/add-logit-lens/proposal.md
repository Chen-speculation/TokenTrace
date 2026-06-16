## Why

学习笔记结尾那句遗憾——"看不到内部神经元、看不到答案为什么成形"——目前 InfoLens 还没回答。前两个能力（梯度归因、消融归因）都停在"输入 ↔ 最终输出"两端，**中间 L 层完全是黑盒**：用户不知道"北京"这个答案是从第几层开始浮现的，也不知道浅层模型在猜什么。

**Logit Lens** 正面回应这一点：取出每一层最后位置的 hidden state，过模型的 final norm + lm_head，得到**逐层的"此刻模型会预测什么"**。于是能看到一条预测轨迹——浅层还在猜高频虚词（"的""是"），中层开始往正确答案靠拢，深层锁定"北京"。这是 InfoLens 第一次真正读**中间表示**，是从"输入输出测量"往"内部过程观测"迈的一步。

它是 canvas 上候选里"开盒价值"和"演示惊艳度"双 5、且复用现有前向骨架最干净的一个，因此排在主线第二阶段（第一阶段消融归因已完成并归档）。

## What Changes

- 新增后端能力：给定 `context`，对 base / instruct 模型做一次 `forward(output_hidden_states=True)`，对 `[0..L]` 每层 hidden state 的末位向量过 `model.norm + lm_head`，得到逐层 logits → softmax → top-k，并标记目标 token（沿用现有 top-1 / `target_prediction` / `target_token_id` 三选一逻辑）在各层的概率。
- 新增 HTTP 端点 `POST /api/logit-lens`，请求形状对齐现有 `POST /prediction-attribute`（`context` / `model` / `source_page` / `target_prediction?` / `target_token_id?`），响应返回逐层 top-k + 目标 token 逐层概率轨迹。复用推理锁、超长校验、`source_page` 校验等基础设施。
- 实现要点：**只前向、不反传**；hidden states 默认在 GPU 上按层投影、逐层取 top-k 后即释放，避免把 `L × seq × vocab` 的张量物化到内存（见 design D3 内存控制）。
- 前端在现有 Attribution 页（`client/src/pages/attribution/index.ts` + inspector）增加一个"逐层预测轨迹"面板：层 × 候选词热力图 + 目标 token 概率的逐层折线，复用现有方法切换（Gradient / Ablation / Both）之外的第三个面板。

## Capabilities

### New Capabilities
- `logit-lens`: 对下一 token 预测做逐层 logit lens——取出每层最后位置的 hidden state，过 final norm + lm_head，返回每层的 top-k 候选与目标 token 的逐层概率，让用户看到答案在第几层成形。

### Modified Capabilities
<!-- 无：现有 prediction-attribution / ablation-attribution 行为不变，新增能力为独立端点；前端 Attribution 页改动属实现细节，无既有 spec。 -->

## Impact

- **新增后端代码**：`backend/core/logit_lens.py`（算法）、`backend/api/logit_lens.py`（handler）。
- **接线**：`server.py` 导入 handler；`server.yaml` 增加 `/logit-lens` 路径（以 `/prediction-attribute` 段为模板）。
- **复用**：`backend/core/prediction_attributor.py` 的模型加载、目标 token 选择、offset 过滤、`ATTRIBUTION_MAX_TOKEN_LENGTH`、`next_token_topk` 解码；`backend/models/model_manager` 的槽位与推理锁；`backend/platform` 的 OOM / 日志 / `DeviceManager.clear_cache`。
- **前端**：`client/src/pages/attribution/index.ts`、attribution inspector 组件、`shared/api/GLTR_API.ts`（新增 `logitLens` 客户端方法）、对应 i18n。
- **性能/内存**：一次前向（`output_hidden_states=True`）+ L 层小矩阵乘（`hidden[-1] @ lm_head.weight.T`），无需反传；内存峰值主要来自 hidden states 缓存（`L × seq × hidden`，0.6B 约 28 层 × hidden 1024，量级可控），逐层投影后即释放。无新增第三方依赖。
