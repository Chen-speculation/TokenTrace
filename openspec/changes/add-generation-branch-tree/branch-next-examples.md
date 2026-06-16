# `POST /api/branch-next` — 请求/响应样例

供 Track B（前端）作为 mock fixture 依据。

---

## 成功：默认 top-10

**请求**
```json
POST /api/branch-next
{
  "prefix": "中国的首都",
  "model": "base",
  "source_page": "causal_flow"
}
```

**响应 200**
```json
{
  "success": true,
  "model": "Qwen3-0.6B-Base",
  "prefix_tokens": 5,
  "candidates": [
    { "token": "是", "token_id": 374, "prob": 0.612 },
    { "token": "北", "token_id": 117, "prob": 0.152 },
    { "token": "城", "token_id": 223, "prob": 0.071 },
    { "token": "的", "token_id": 89,  "prob": 0.034 },
    { "token": "，", "token_id": 11,  "prob": 0.021 }
  ],
  "is_context_full": false
}
```

> `candidates` 按 `prob` 降序，长度 = `top_k`（默认 10）  
> 前端追加分支：新 prefix = 旧 prefix + `candidate.token`

---

## 成功：自定义 top_k

**请求**
```json
{
  "prefix": "The capital of France is",
  "model": "base",
  "source_page": "causal_flow",
  "top_k": 5
}
```

**响应 200**（同上结构，candidates 长度 = 5）

---

## 成功：is_context_full = true

当 `prefix_tokens + 1 >= model.config.max_position_embeddings` 时：

```json
{
  "success": true,
  "model": "Qwen3-0.6B-Base",
  "prefix_tokens": 32767,
  "candidates": [...],
  "is_context_full": true
}
```

> 前端应禁用该节点的进一步展开并提示"已到上下文末尾"

---

## 400：prefix 超出上下文上限

```json
{
  "success": false,
  "message": "Prefix length (32768 tokens) has reached the model context limit (32768); cannot expand further."
}
```

## 400：缺少 prefix

```json
{ "success": false, "message": "Missing required field: prefix" }
```

## 400：非法 model

```json
{ "success": false, "message": "model must be \"base\" or \"instruct\"" }
```

## 400：top_k < 1

```json
{ "success": false, "message": "top_k must be >= 1" }
```

> `top_k > 50` 时自动 clamp 到 50，不返回 400

## 503：服务繁忙

```json
{ "success": false, "message": "Queue wait exceeded 30 seconds; server is busy, please try again later." }
```

---

## 字段说明

| 字段 | 说明 |
|------|------|
| `prefix_tokens` | prefix 的 token 数，前端可用于展示进度条 |
| `candidates[].token` | token 原始文本（含前导空格/特殊字符，`skip_special_tokens=False`） |
| `candidates[].token_id` | token 在词表中的 id（仅供展示/调试，不参与前端 prefix 拼接） |
| `is_context_full` | `true` 时前端禁用该节点展开 |
