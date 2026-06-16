# 说明：任务分两条独立轨道

- **Track A（后端 & 算法）**：组 1–4。可独立 apply、独立验证（脚本/单测打到 `/api/logit-lens`）。
- **Track B（前端）**：组 5–8。依赖 Track A 的端点契约（见 `design.md` D4 的请求/响应形状）；可对契约先行用 mock 数据开发，待 Track A 就绪后联调。
- 两轨唯一耦合点是 HTTP 契约；请勿跨轨改对方代码。

## 1. 算法核心（Track A · 后端）

- [x] 1.1 新建 `backend/core/logit_lens.py`，定义 `analyze_logit_lens(context, target_prediction=None, *, model, target_token_id=None)`，签名与返回结构对齐 `prediction_attributor.analyze_prediction_attribution`（复用其模型加载、目标 token 选择、超长校验）
- [x] 1.2 实现 `forward(output_hidden_states=True, use_cache=False)`，拿到 `hidden_states: tuple`（长度 L+1）；先做一次最终层 logits 自检：argmax 必须等于标准 `outputs.logits[:, -1, :].argmax()`（半精度用 `torch.isclose` 容差）
- [x] 1.3 取得 `norm = hf_model.model.norm`（`hasattr` 兜底，找不到报清晰错误）与 `lm_head = hf_model.get_output_embeddings()`；对每层末位 hidden 过 `norm → lm_head`，验证最终层投影与标准 logits 一致（D1 自检）
- [x] 1.4 逐层投影（D3 内存控制）：循环 `ℓ in 0..L`，取 `hidden_states[ℓ][:, -1, :]` → norm → lm_head → softmax → topk(k=DEFAULT_NEXT_TOKEN_TOPK) → 立即 `.cpu().tolist()` 保留 top-k 与目标 token 概率，**丢弃该层 `[1,vocab]` 张量**；严禁 `torch.cat` 出 `[L,vocab]`
- [x] 1.5 解析目标 token（top-1 / `target_prediction` 首 token / `target_token_id`，三者互斥），记录其在各层的概率，构成 `target_trajectory`；`final_target_prob` 必须等于现有归因端点对该 context 的 `target_prob`（跨端点自检）
- [x] 1.6 组装响应 `{model, target_token, n_layers, final_target_prob, layers:[{layer,is_embedding,topk_tokens,topk_probs,target_prob}], debug_info{topk_tokens,topk_probs}, is_eos}`；NaN/Inf→0；按设备做半精度降级

## 2. API 端点（Track A · 后端）

- [x] 2.1 新建 `backend/api/logit_lens.py`，复用 `prediction_attribute.py` 的参数校验（context/model/source_page/target 互斥/flow_id/flow_step）、推理锁获取与超时、日志与 OOM 处理骨架
- [x] 2.2 在 `server.py` 导入 `logit_lens` handler（与现有 `prediction_attribute` / `ablation_attribute` 同处）
- [x] 2.3 在 `server.yaml` 仿 `/prediction-attribute` 段新增 `/logit-lens` 路径：请求体（context/model/target_prediction?/target_token_id?/source_page/flow_id?/flow_step?）与响应 schema（含 `layers[]` 逐层结构、`n_layers`、`final_target_prob`），并补 400/500/503 描述

## 3. 后端验证（Track A · 后端）

- [x] 3.1 加单测覆盖：top-1 默认、显式 `target_token_id`、`target_prediction`、目标互斥 400、缺 context/非法 model/非法 source_page 400、超长报错（对应 spec 场景，mock core 无需模型权重）
- [x] 3.2 加测试断言核心语义（需模型权重，`@skipUnless`）：`layers` 长度 = `n_layers+1`、下标 0 `is_embedding==True`、最终层 `target_prob == final_target_prob`、最终层投影 top-1 等于标准 logits argmax（D1 自检）
- [x] 3.3 冒烟脚本：本地起服务对 `/api/logit-lens` 发若干样例（含中文「中国的首都是」类），核对目标 token 概率随层单调上升的趋势合理、浅层常落高频词

## 4. 契约固化（Track A · 交接物）

- [x] 4.1 产出端点请求/响应样例（成功 + 各 400）作为 Track B 的 mock fixture 依据，确认与 `server.yaml`、`design.md` D4 完全一致

## 5. API 客户端（Track B · 前端）

- [x] 5.1 在 `client/src/shared/api/GLTR_API.ts` 新增 `logitLens(...)` 方法，参数/返回类型对齐端点契约（含 `layers[]`、`n_layers`、`final_target_prob`），错误处理与现有归因请求一致
- [x] 5.2 更新/扩展前端类型（如 `generatedSchemas`/相关 type）以容纳 logit-lens 响应

## 6. 面板与状态（Track B · 前端）

- [x] 6.1 在 `client/src/pages/attribution/index.ts` 增加 Logit Lens 面板状态与请求触发（与 Gradient/Ablation/Both 三态并存，因维度不同不互斥），统一 loading / 错误 / 取消（abort）处理
- [x] 6.2 持久化 Logit Lens 面板的开关与设置，方式与该页现有设置一致

## 7. 可视化与读数（Track B · 前端）

- [x] 7.1 渲染逐层 top-k 热力图：行=层（0..L），列=top-k 槽位，单元格=token 字符串 + 概率，颜色编码概率
- [x] 7.2 渲染目标 token 概率逐层折线，标出"首次进入 top-1 的层"（找不到则显式说明"目标从未进入 top-1"）
- [x] 7.3 tooltip/说明注明「逐层投影读数（final norm + lm_head 套到中间层），非模型逐层真实计算」（对应 design 风险项 / spec 的 disclaimer 场景）

## 8. 收尾（Track B · 前端）

- [x] 8.1 补充面板标题、热力图图例、折线标注等文案的 i18n（中/英）
- [x] 8.2 `cd client/src && npm run build` 通过，手测 Logit Lens 面板在示例文本（「中国的首都是」/英文）上的渲染与逐层轨迹
