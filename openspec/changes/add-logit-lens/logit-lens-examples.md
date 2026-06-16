# `POST /api/logit-lens` — 请求/响应样例

供 Track B（前端）作为 mock fixture 依据。

---

## 成功：top-1 默认

**请求**
```json
POST /api/logit-lens
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
  "model": "Qwen3-0.6B-Base",
  "target_token": "北",
  "n_layers": 28,
  "final_target_prob": 0.812,
  "layers": [
    { "layer": 0, "is_embedding": true,  "topk_tokens": ["的","是","，","在","了"], "topk_probs": [0.12,0.09,0.08,0.07,0.06], "target_prob": 0.0001 },
    { "layer": 1, "is_embedding": false, "topk_tokens": ["北","是","的","京","在"], "topk_probs": [0.15,0.11,0.09,0.08,0.07], "target_prob": 0.003 },
    "...(省略中间层)...",
    { "layer": 28, "is_embedding": false, "topk_tokens": ["北","京","上","南","武"], "topk_probs": [0.812,0.063,0.021,0.018,0.011], "target_prob": 0.812 }
  ],
  "debug_info": {
    "topk_tokens": ["北","京","上","南","武"],
    "topk_probs":  [0.812,0.063,0.021,0.018,0.011]
  },
  "is_eos": false
}
```

> `layers` 长度 = `n_layers + 1`（0 = embedding 层，28 = 最终层）  
> `final_target_prob` == `layers[-1].target_prob`（自检锚点）  
> `debug_info` 与 `/prediction-attribute` 同形，便于 UI 复用

---

## 400：target 互斥

```json
{ "success": false, "message": "target_prediction and target_token_id are mutually exclusive" }
```

## 400：缺少 context

```json
{ "success": false, "message": "Missing required field: context" }
```

## 400：非法 model

```json
{ "success": false, "message": "model must be \"base\" or \"instruct\"" }
```

## 400：context 超长

```json
{ "success": false, "message": "Context exceeds attribution length limit (500 tokens); current length is 612 tokens." }
```

## 503：服务繁忙

```json
{ "success": false, "message": "Queue wait exceeded 30 seconds; server is busy, please try again later." }
```

---

## 字段说明

| 字段 | 说明 |
|------|------|
| `n_layers` | Transformer 层数 L（不含 embedding 层 0） |
| `final_target_prob` | 最终层目标概率，与 `/prediction-attribute` 的 `target_prob` 数值一致 |
| `layers[0].is_embedding` | `true`（embedding 层投影读数） |
| `layers[i].target_prob` | 目标 token 在第 i 层的概率，可为接近 0 |
| `debug_info` | 最终层 top-k，与梯度归因 debug_info 同形 |

> **UI 注意**：Logit Lens 是把 `final norm + lm_head` 提前套到中间层的**投影读数**，非模型真实逐层计算意图。
