# `POST /api/ablation-attribute` — 请求/响应样例

供 Track B（前端）作为 mock fixture 依据。与 `server.yaml` `/ablation-attribute` 段及 `design.md` D4 完全一致。

---

## 成功：top-1 默认目标

**请求**
```json
POST /api/ablation-attribute
{
  "context": "中国的首都是",
  "model": "base",
  "source_page": "attribution"
}
```

**响应 200**
```json
{
  "success": true,
  "model": "Qwen3-0.6B",
  "target_token": "北",
  "target_prob": 0.812,
  "token_attribution": [
    { "offset": [0, 3], "raw": "中国", "score": 0.134, "delta_logit": 0.21 },
    { "offset": [3, 4], "raw": "的",  "score": 0.012, "delta_logit": 0.02 },
    { "offset": [4, 6], "raw": "首都", "score": 0.089, "delta_logit": 0.14 },
    { "offset": [6, 7], "raw": "是",  "score": -0.005, "delta_logit": -0.01 }
  ],
  "debug_info": {
    "topk_tokens": ["北", "京", "上", "南", "武"],
    "topk_probs":  [0.812, 0.063, 0.021, 0.018, 0.011]
  },
  "is_eos": false
}
```

> `score = baseline_target_prob − occluded_target_prob`，正值表示该 token 支撑目标，负值表示抑制。
> `delta_logit` 为辅助读数（baseline_logit − occluded_logit），缓解 softmax 饱和时 ΔP 偏小的问题。

---

## 成功：显式 `target_prediction`

**请求**
```json
{
  "context": "The capital of France is",
  "model": "base",
  "source_page": "attribution",
  "target_prediction": "Paris"
}
```

**响应 200**（结构同上，target_token 为 "Paris" 的首个 subword）

---

## 成功：显式 `target_token_id`

**请求**
```json
{
  "context": "中国的首都是",
  "model": "base",
  "source_page": "attribution",
  "target_token_id": 15946
}
```

**响应 200**（结构同上）

---

## 400：target 互斥

**请求**
```json
{
  "context": "hello",
  "model": "base",
  "source_page": "attribution",
  "target_prediction": "world",
  "target_token_id": 5
}
```

**响应 400**
```json
{
  "success": false,
  "message": "target_prediction and target_token_id are mutually exclusive"
}
```

---

## 400：缺少 context

**请求**
```json
{
  "model": "base",
  "source_page": "attribution"
}
```

**响应 400**
```json
{
  "success": false,
  "message": "Missing required field: context"
}
```

---

## 400：非法 model

**请求**
```json
{
  "context": "hello",
  "model": "gpt4",
  "source_page": "attribution"
}
```

**响应 400**
```json
{
  "success": false,
  "message": "model must be \"base\" or \"instruct\""
}
```

---

## 400：非法 source_page

**请求**
```json
{
  "context": "hello",
  "model": "base",
  "source_page": "nowhere"
}
```

**响应 400**
```json
{
  "success": false,
  "message": "source_page must be one of: analysis, attribution, causal_flow, chat (legacy *.html and gen_attribute accepted)"
}
```

---

## 400：context 超长

**请求**（tokenize 后超过 500 tokens）
```json
{
  "context": "<very long text...>",
  "model": "base",
  "source_page": "attribution"
}
```

**响应 400**
```json
{
  "success": false,
  "message": "Context exceeds attribution length limit (500 tokens); current length is 612 tokens."
}
```

---

## 503：服务繁忙

**响应 503**
```json
{
  "success": false,
  "message": "Queue wait exceeded 30 seconds; server is busy, please try again later."
}
```

---

## 字段说明（Track B 对照）

| 字段 | 类型 | 说明 |
|------|------|------|
| `target_prob` | float | baseline 目标概率（所有 score 的参考基准） |
| `token_attribution[].score` | float | ΔP，**可为负**；正=支撑目标，负=抑制目标 |
| `token_attribution[].delta_logit` | float | Δlogit，辅助读数（软饱和时比 ΔP 更显著） |
| `token_attribution[].offset` | [int, int] | 与梯度归因 **完全对齐** 的字符偏移，BOS/EOS 等特殊 token 已排除 |
| `is_eos` | bool | 目标 token 是否为 EOS（与梯度归因端点语义一致） |
