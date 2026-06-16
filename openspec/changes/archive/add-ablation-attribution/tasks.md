# 说明：任务分两条独立轨道

- **Track A（后端 & 算法）**：组 1–4。可独立 apply、独立验证（脚本/单测打到 `/api/ablation-attribute`）。
- **Track B（前端）**：组 5–8。依赖 Track A 的端点契约（见 `design.md` D4 的请求/响应形状）；可对契约先行用 mock 数据开发，待 Track A 就绪后联调。
- 两轨唯一耦合点是 HTTP 契约；请勿跨轨改对方代码。

## 1. 算法核心（Track A · 后端）

- [x] 1.1 新建 `backend/core/ablation_attributor.py`，定义 `analyze_ablation_attribution(context, target_prediction=None, *, model, target_token_id=None)`，签名与返回结构对齐 `prediction_attributor.analyze_prediction_attribution`（复用其模型加载、目标 token 选择、offset 过滤、`next_token_topk` 解码工具）
- [x] 1.2 实现 baseline 前向：编码 `context`，取 `inputs_embeds`，前向得末位 logits，按 top-1 / `target_prediction` / `target_token_id` 解析 `target_token_id` 与 baseline `target_prob`（沿用现有互斥与超长校验）
- [x] 1.3 实现 mean-embedding 基线向量（context token embedding 均值，见 design D2），并把 baseline 选择收敛到单一常量便于后续配置化
- [x] 1.4 构造遮挡 batch：`[N+1, seq, hidden]` 的 `inputs_embeds`（baseline + 每个非特殊 token 各一行替换为基线向量），单次 `forward(use_cache=False, output_attentions=False)`，对同一 target id 取各行末位概率
- [x] 1.5 计算逐 token `score = baseline_target_prob − occluded_target_prob`（保留正负号），按现有 offset 规则过滤零宽特殊 token，组装 `token_attribution: [{offset, raw, score}]`；可选附 `delta_logit`
- [x] 1.6 内存分批：当 `(N+1) × seq` 超阈值时按行分批多次前向累积；阈值放运行时配置，复用 `DeviceManager.clear_cache`、设备同步与 NaN/Inf→0 处理
- [x] 1.7 返回 `{model, target_token, target_prob, token_attribution, debug_info{topk_tokens,topk_probs}, is_eos}`，并按设备类型做半精度数值差异/INT8 不支持的既有降级与报错

## 2. API 端点（Track A · 后端）

- [x] 2.1 新建 `backend/api/ablation_attribute.py`，复用 `prediction_attribute.py` 的参数校验（context/model/source_page/target 互斥/flow_id/flow_step）、推理锁获取与超时、日志与 OOM 处理骨架
- [x] 2.2 在 `server.py` 导入 `ablation_attribute` handler（与现有 `prediction_attribute` 同处）
- [x] 2.3 在 `server.yaml` 仿 `/prediction-attribute` 段新增 `/ablation-attribute` 路径：请求体（context/model/target_prediction?/target_token_id?/source_page/flow_id?/flow_step?）与响应 schema（含 `score` 可为负、可选 `delta_logit`），并补 400/500/503 描述

## 3. 后端验证（Track A · 后端）

- [x] 3.1 加单测覆盖：top-1 默认、显式 `target_token_id`、`target_prediction`、目标互斥 400、缺 context/非法 model/非法 source_page 400、超长报错（对应 spec 场景）
- [x] 3.2 加测试断言核心语义：`score = baseline − occluded`、offset 满足 `start<end` 且 `raw==context[start:end]`、特殊 token 被排除（对应「Coordinate alignment」需求）
- [x] 3.3 冒烟脚本：本地起服务对 `/api/ablation-attribute` 发若干样例（含中文「中国的首都是」类）核对 ΔP 方向合理

## 4. 契约固化（Track A · 交接物）

- [x] 4.1 产出端点请求/响应样例（成功 + 各 400）作为 Track B 的 mock fixture 依据，确认与 `server.yaml`、`design.md` D4 完全一致

## 5. API 客户端（Track B · 前端）

- [x] 5.1 在 `client/src/shared/api/GLTR_API.ts` 新增 `ablationAttribute(...)` 方法，参数/返回类型对齐端点契约（含 `score` 可为负、可选 `delta_logit`），错误处理与现有归因请求一致
- [x] 5.2 更新/扩展前端类型（如 `generatedSchemas`/相关 type）以容纳 ablation 响应

## 6. 方法切换与状态（Track B · 前端）

- [x] 6.1 在 `client/src/pages/attribution/index.ts` 增加方法选择状态：Gradient / Ablation / Both，持久化方式与该页现有设置一致
- [x] 6.2 Both 模式下并行发起梯度与消融两次请求，统一 loading / 错误 / 取消（abort）处理

## 7. 可视化与读数（Track B · 前端）

- [x] 7.1 attribution inspector 渲染消融 score：因 score 有正负，使用发散色阶（正=支撑目标 / 负=抑制目标）
- [x] 7.2 Both 模式同屏并排两套 token 高亮（梯度 vs 消融），坐标对齐到同一 token
- [x] 7.3 显示至少一项一致性读数（Spearman 排名相关或 Top-k 重合度）于 inspector
- [x] 7.4 tooltip/说明注明「定长遮挡 + 均值基线，ΔP 为该位置被中性化后的效果，非真实因果机制」（对应 design 风险项）

## 8. 收尾（Track B · 前端）

- [x] 8.1 补充方法切换、色阶图例、一致性读数等文案的 i18n（中/英）
- [x] 8.2 `cd client/src && npm run build` 通过，手测 Gradient/Ablation/Both 三态在示例文本上的渲染与对照
