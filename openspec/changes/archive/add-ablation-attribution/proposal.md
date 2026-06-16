## Why

InfoLens 现有的 Context Attribution 用的是**梯度归因**（`logit[target].backward()` 取输入 embedding 的梯度 L2 范数），它衡量的是"输入无穷小扰动 → 输出变多少"，本质是**相关性**，不是因果。学习笔记里最关键的认知正是这一点："梯度是事后算的相关性分析，不是真实因果机制"。

消融归因把这句话变成一个可交互的实验：真的把某个输入 token 替换成中性基线、重新前向，量出目标 token 概率的**实际变化 ΔP**。把"梯度（相关性）"和"消融（实际效果）"并排展示，用户能一眼看到两种解释在哪里一致、哪里打架——这是 InfoLens 从"相关性测量仪"升级为"可交互机制探针"路线图的第一阶段（后续为 Logit Lens、生成分叉树）。

## What Changes

- 新增后端能力：给定 `context` + 归因目标 token（沿用现有 top-1 / `target_prediction` / `target_token_id` 三选一的选择逻辑），对每个输入 token 计算"消融该 token 后目标概率的变化量"，返回逐 token 的 ΔP（及可选 Δlogit）与 baseline 目标概率。
- 新增 HTTP 端点 `POST /api/ablation-attribute`，请求/响应形状对齐现有 `POST /prediction-attribute`，复用推理锁、超长校验、`source_page` 校验等基础设施。
- 实现采用 **embedding 级遮挡（occlusion）**：保持序列长度不变，逐 token 把 embedding 替换为基线向量，N 个变体 + 1 个 baseline 拼成一个 batch 一次前向（无需反向传播），与梯度归因的 token 坐标 1:1 对齐，便于并排。
- 前端在现有 Attribution 页（`client/src/pages/attribution/index.ts` + attribution inspector）增加方法切换（Gradient / Ablation / Both）与第二套配色，并给出两种方法的一致性读数（如排名相关性）。

## Capabilities

### New Capabilities
- `ablation-attribution`: 对下一 token 预测做基于遮挡的因果归因——逐输入 token 替换为基线并重算目标概率，返回每个 token 的实际影响量，供与梯度归因并排对照。

### Modified Capabilities
<!-- 无：现有 attribution 行为不变，新增能力为独立端点；前端 Attribution 页改动属实现细节，无既有 spec。 -->

## Impact

- **新增后端代码**：`backend/core/ablation_attributor.py`（算法）、`backend/api/ablation_attribute.py`（handler）。
- **接线**：`server.py` 导入 handler；`server.yaml` 增加 `/ablation-attribute` 路径（以 `/prediction-attribute` 段为模板）。
- **复用**：`backend/core/prediction_attributor.py` 的模型加载、目标 token 选择、offset 过滤、`next_token_topk` 解码；`backend/models/model_manager` 的槽位与推理锁；`backend/platform` 的 OOM / 日志。
- **前端**：`client/src/pages/attribution/index.ts`、attribution inspector 组件、`shared/api/GLTR_API.ts`（新增 `ablationAttribute` 客户端方法）、对应 i18n。
- **性能/内存**：batch = (token 数 + 1) 的一次前向，长上下文需分批；归因长度上限沿用现有约束。无新增第三方依赖。
