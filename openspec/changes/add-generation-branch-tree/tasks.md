# 说明：任务分两条独立轨道

- **Track A（后端 & 算法）**：组 1–4。可独立 apply、独立验证（脚本/单测打到 `/api/branch-next`）。
- **Track B（前端）**：组 5–8。依赖 Track A 的端点契约（见 `design.md` D3 的请求/响应形状）；可对契约先行用 mock 数据开发，待 Track A 就绪后联调。
- 两轨唯一耦合点是 HTTP 契约；请勿跨轨改对方代码。

---

## Track A · 后端

## 1. 算法核心（Track A · 后端）

- [x] 1.1 新建 `backend/core/branch_next.py`，定义 `expand_branch_next(prefix, *, model, top_k=DEFAULT_NEXT_TOKEN_TOPK)`，复用 `completion_generator.ensure_slot_ready` 的槽位加载与 `next_token_topk.decode_topk_ids_to_strings_and_rounded_probs`
- [x] 1.2 实现单步前向（D1）：编码 prefix → 一次 `forward(use_cache=False, output_attentions=False)` → 取末位 logits → softmax → `torch.topk(logits, k)`；**禁止调用 `model.generate`、禁止反传**
- [x] 1.3 计算各候选的 token 文本（`skip_special_tokens=False`）、token_id、概率，按 prob 降序组装 `candidates: [{token, token_id, prob}]`；记录 `prefix_tokens` 与 `is_context_full`
- [x] 1.4 `top_k` 边界处理：默认 `DEFAULT_NEXT_TOKEN_TOPK`（10）；请求值 > 上限（暂定 50）则 clamp 到上限、< 1 则报错；NaN/Inf→0；按设备做半精度降级与 `DeviceManager.clear_cache`

## 2. API 端点（Track A · 后端）

- [x] 2.1 新建 `backend/api/branch_next.py`，复用 `prediction_attribute.py` 的参数校验（prefix/model/source_page）、推理锁获取与超时、日志与 OOM 处理骨架
- [x] 2.2 在 `server.py` 导入 `branch_next` handler（与现有 `prediction_attribute` / `ablation_attribute` 同处）
- [x] 2.3 在 `server.yaml` 仿 `/prediction-attribute` 段新增 `/branch-next` 路径：请求体（prefix/model/source_page/top_k?）与响应 schema（含 `candidates[]`、`prefix_tokens`、`is_context_full`），并补 400/500/503 描述

## 3. 后端验证（Track A · 后端）

- [x] 3.1 加单测覆盖（mock core，无需模型权重）：缺 prefix/非法 model/非法 source_page 400、`top_k` 越界（<1 报错、>上限 clamp）、正常 top-1 默认、自定义 top_k、服务繁忙 503
- [x] 3.2 加算法语义测试（需模型权重，`@skipUnless`）：top-1 的 `token_id` 等于末位 logits argmax、候选按 prob 降序、`prefix_tokens` 正确、单次前向（断言 `model.generate` 未被调用，可 mock 计数）
- [x] 3.3 冒烟脚本：本地起服务对 `/api/branch-next` 发若干样例（中文「中国的首都」/英文），核对 top-1 合理（"是"/" Beijing" 之类），并验证 prefix 接近上下文上限时 `is_context_full` 正确翻转

## 4. 契约固化（Track A · 交接物）

- [x] 4.1 产出端点请求/响应样例（成功 + 各 400 + context_full）作为 Track B 的 mock fixture 依据，确认与 `server.yaml`、`design.md` D3 完全一致

---

## Track B · 前端

## 5. API 客户端（Track B · 前端）

- [x] 5.1 在 `client/src/shared/api/GLTR_API.ts` 新增 `branchNext(...)` 方法，参数/返回类型对齐端点契约（含 `candidates[]`、`prefix_tokens`、`is_context_full`），错误处理与现有归因请求一致
- [x] 5.2 更新/扩展前端类型（如 `generatedSchemas`/相关 type）以容纳 branch-next 响应

## 6. 模式切换与状态（Track B · 前端）

- [x] 6.1 在 `client/src/pages/causal_flow/index.ts` 增加"DAG 模式 / 分叉树模式"切换，持久化方式与该页现有设置一致；分叉树模式使用独立的输入区与画布
- [x] 6.2 实现 `TreeNode` 数据结构（节点 = prefix 文本 + 候选列表 + 子节点 + 元数据），严格树形（每节点唯一父节点）；维护根到当前节点的 prefix 路径

## 7. 可视化与交互（Track B · 前端）

- [x] 7.1 渲染分叉树（D4）：树形布局（d3 hierarchy / dagre tree），节点/连线/概率色阶复用现有 DAG 代码；根节点 + 其 top-k 候选为可点叶子
- [x] 7.2 点候选 → 新 prefix = 旧 prefix + 候选文本 → 调 `branchNext` → 渲染新子节点；选中态、路径高亮
- [x] 7.3 防爆炸（D4 硬上限）：树深 ≤ 12、每节点展示 top-k ≤ 5（可配）、总节点数 ≤ 200；超限拒绝展开并提示具体命中哪条限制
- [x] 7.4 请求管理（D5）：`AbortController` 取消进行中请求；节点级 LRU 缓存，重复点同一节点不重复请求；`is_context_full` 为 true 时禁用该节点展开并提示

## 8. 收尾（Track B · 前端）

- [x] 8.1 补充分叉树模式标题、限制提示、候选 tooltip、context_full 提示等文案的 i18n（中/英）
- [x] 8.2 `cd client/src && npm run build` 通过，手测分叉树模式：根展开、开新分支、命中深度/节点上限、取消请求、缓存命中、context_full 禁用
