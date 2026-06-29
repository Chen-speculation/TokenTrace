# InfoLens / INNERSCOPE — 统一工作流重设计 Prompt

> **目标受众**：专精 UI/UX 设计的 Agent  
> **任务**：理解现有前端架构和各功能模块，设计一个统一的工作流 demo，将当前孤立的 7 个功能（含新训练的 Tiny-NLA）有机融合为一个流畅的「输入文本 → 多维度探索模型内部」体验。  
> **输出形式**：给出初步的 HTML/CSS/JS demo（可以是静态原型或可交互 demo），不需要对接真实后端 API。  
> **重要的**：这是概念验证 demo，不是产品实现。你可以自由选择任何设计方向、技术栈、视觉风格。不要被现有代码的样式束缚。

---

## 一、项目是什么

**InfoLens（品牌名 INNERSCOPE）** 是一个「透视大语言模型内部」的可视化工具箱。用户输入一段文本，通过不同的分析维度观察模型在想什么。

- **服务端**：Python Flask/Connexion，跑在本地 M4 Pro Mac 上  
- **模型**：Qwen3-0.6B (base + instruct)，本地 CPU/MPS 推理  
- **用户画像**：AI 研究者 / 可解释性爱好者，希望直观看到模型内部状态  

---

## 二、现有前端代码结构（供参考，不需要遵循）

```
client/src/
├── index.html                    # 首页（导航网格）
├── analysis.html                 # Info Highlight 页
├── chat.html                     # Raw Chat 页
├── attribution.html              # Attribution 页
├── causal_flow.html              # Causal Flow 页
├── logit_lens.html               # Logit Lens 页
├── branch_tree.html              # Branch Tree 页
│
├── webpack.config.js             # 多入口打包
│
├── pages/                        # 每个页面的 TS 入口
│   ├── home/index.ts
│   ├── analysis/index.ts
│   ├── chat/index.ts             # ~800 行，最复杂
│   ├── attribution/index.ts
│   ├── causal_flow/index.ts
│   ├── logit_lens/index.ts
│   └── branch_tree/index.ts
│
├── shared/                       # 跨页面共享基础设施
│   ├── api/GLTR_API.ts           # TextAnalysisAPI 类（所有后端通信）
│   ├── core/                     # URL 解析、事件总线、工具函数
│   ├── ui/                       # toast, dialog, 主题/语言切换, 面板布局
│   ├── vis/                      # D3 可视化组件（Token着色、Tooltip、直方图、散点图）
│   └── lang/translations.ts      # i18n 翻译
│
├── partials/                     # HTML 片段（被 include 到页面中）
│
├── features/                     # 按功能域拆分
│   ├── analysis/                 # Info Highlight 流程
│   ├── chat/                     # 对话/续写逻辑
│   ├── causal_flow/              # Causal Flow 逻辑
│   └── demo/                     # Demo 存储
│
└── css/pages/                    # 每页独立的 SCSS
```

---

## 三、7 个功能模块详解（含完整 API Schema）

### 3.1 Info Highlight — Token 级别信息密度分析

**做什么**：展示模型对文本中每个 token 的「意外程度」——哪些词模型觉得理所当然，哪些词让它意外。

**API：`POST /api/analyze`**
```
请求：
{
  "model": "base" | "instruct",
  "text": "water freezes into solid ice",
  "stream": false              // 可选，true 时 SSE 流式返回
}

响应 200：
{
  "request": { "text": "water freezes into solid ice" },
  "result": {
    "model": "qwen3-0.6b-base",
    "bpe_strings": [
      {
        "offset": [0, 5],        // 字符偏移 [start, end)
        "raw": "water",           // token 原文
        "real_topk": [156, 0.023],  // [模型排序名次, softmax 概率]
        "pred_topk": [            // 该位置 top-N 候选
          ["ice", 0.34],
          ["liquid", 0.18],
          ...
        ]
      },
      ...
    ]
  }
}

错误：400 (缺参数) / 404 (模型未注册) / 500 (推理失败) / 503 (模型加载失败)
```

**附加子功能 — 语义搜索：`POST /api/analyze-semantic`**
```
请求：
{
  "query": "frozen",            // 查询主题
  "text": "...",                // 原文
  "stream": false,              // 可选 SSE 流式
  "submode": "count",           // 可选: count / fill_blank
  "debug_info": true            // 可选，返回 top10 预测
}

响应 200：
{
  "success": true,
  "model": "qwen3-0.6b-instruct",
  "token_attention": [
    {
      "offset": [13, 19],
      "raw": "solid",
      "score": 0.87             // 对 query 的语义关注度
    },
    ...
  ],
  "debug_info": {               // 仅 debug_info=true 时
    "abbrev": "...",
    "topk_tokens": [...],
    "topk_probs": [...]
  }
}
```

---

### 3.2 Raw Chat — 对话 & 续写

**做什么**：和模型对话，可视化每次续写中每个生成 token 的细节。支持 Raw 模式（直接给 prompt）和 Chat 模式（system + user prompt + chat template）。

**API 链（3 步）：**

**Step 1 — 组装 prompt：`POST /v1/completions/prompt`**
```
请求：
{
  "model": "qwen3-0.6b-instruct",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is water?" }
  ],
  "tools": [...],               // 可选 OpenAI tools schema
  "enable_thinking": false      // 可选，启用 Qwen3 thinking 模式
}

响应 200：
{
  "prompt_used": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\nWhat is water?<|im_end|>\n<|im_start|>assistant\n"
}
```

**Step 2 — 续写（SSE 流式）：`POST /v1/completions`**
```
请求：
{
  "model": "qwen3-0.6b-instruct",
  "prompt": "<|im_start|>system\n...",  // 从上一步或直接构造
  "max_tokens": 256,                      // 可选，正整数
  "temperature": 0.7,                     // 可选，0-2
  "top_p": 0.9,                           // 可选，0-1
  "stop": ["\n\n"]                        // 可选，停止序列（最多4个）
}
// 响应恒为 text/event-stream (SSE)
// type=delta → 增量 token
// type=result → 末条，data 同 OpenAICompletionsResponse

SSE 末条 data (type=result)：
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1719000000,
  "model": "qwen3-0.6b-instruct",
  "choices": [{
    "text": "Water is a chemical substance...",
    "index": 0,
    "finish_reason": "stop"     // stop / length / content_filter
  }],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 45,
    "total_tokens": 69
  },
  "info_radar": {               // 续写 token 级分析
    "bpe_strings": [
      { "offset": [0, 5], "raw": "Water", "real_topk": [1, 0.92], "pred_topk": [...] },
      ...
    ]
  }
}
```

**Step 3 — 停止生成：`POST /v1/completions/stop`**
```
请求：(无 body)
响应 200：{ "ok": true }
```

**高级功能**：Tool Calling 多轮模拟（通过 `/v1/completions/prompt-incremental` 计算 tool response 增量后缀）、Teacher Forcing（强制续写特定文本）、multi-turn 对话缓存与恢复。

---

### 3.3 Attribution — 输入对预测的因果归因

**做什么**：给定上下文和目标预测，分析「是哪些输入词驱动了这个预测」。提供两种方法：梯度归因（gradient-based）和消融归因（ablation/occlusion-based）。

**方法 A — 梯度归因：`POST /api/prediction-attribute`**
```
请求：
{
  "context": "water freezes and turns into solid",   // 必填，token 数 ≤ 2000
  "target_prediction": " ice",                        // 可选，缺省用 top-1
  "model": "base" | "instruct",                      // 必填
  "source_page": "attribution",                       // 必填
  "flow_id": null,                                    // 可选，连续归因会话 ID
  "flow_step": null                                   // 可选，连续归因步骤
}

响应 200：
{
  "success": true,
  "model": "qwen3-0.6b-base",
  "target_token": "ice",           // 归因目标 token
  "target_prob": 0.34,             // 预测概率
  "token_attribution": [
    {
      "offset": [0, 5],            // 字符偏移 [start, end]
      "raw": "water",
      "score": 0.42                // 梯度 L2 范数归因分（正值=支撑，越低越无关）
    },
    { "offset": [30, 35], "raw": "solid", "score": 0.78 },
    ...
  ],
  "debug_info": {                  // 下一 token 的 top-10
    "topk_tokens": ["ice", "liquid", "crystal", ...],
    "topk_probs": [0.34, 0.18, 0.12, ...]
  },
  "is_eos": false                  // target 是否为 EOS token
}
```

**方法 B — 消融归因：`POST /api/ablation-attribute`**
```
请求：
{
  "context": "water freezes and turns into solid",   // 必填，token 数 ≤ 500
  "target_prediction": " ice",                        // 可选，与 target_token_id 互斥
  "target_token_id": null,                            // 可选，直接指定 token id
  "model": "base" | "instruct",
  "source_page": "attribution"
}

响应 200：
{
  "success": true,
  "target_token": "ice",
  "target_prob": 0.34,             // baseline 概率
  "token_attribution": [
    {
      "offset": [0, 5],
      "raw": "water",
      "score": 0.12,               // ΔP = baseline_prob − occluded_prob（可为负）
      "delta_logit": 0.38          // Δlogit
    },
    ...
  ],
  "is_eos": false
}

错误：400 (缺字段/model非法/超长) / 500 (推理失败) / 503 (繁忙)
```

---

### 3.4 Logit Lens — 逐层解码轨迹

**做什么**：看模型在每一层的「中间想法」——把每层 hidden state 投影到词表（final norm + lm_head），展示每层 top-k 候选词和目标词概率如何逐层演化。

**API：`POST /api/logit-lens`**
```
请求：
{
  "context": "The sun rises in the east and sets in the",  // 必填，token 数 ≤ 500
  "target_prediction": " west",                             // 可选，与 target_token_id 互斥
  "target_token_id": null,                                  // 可选
  "model": "base" | "instruct",
  "source_page": "logit_lens"
}

响应 200：
{
  "success": true,
  "model": "qwen3-0.6b-base",
  "target_token": "west",
  "n_layers": 28,                    // Transformer 层数（不含 embedding）
  "final_target_prob": 0.72,         // 最终层目标 token 概率
  "layers": [
    {
      "layer": 0,
      "is_embedding": true,          // 第 0 层为 embedding 层
      "topk_tokens": ["the", "a", "in", "and", ...],
      "topk_probs": [0.12, 0.09, 0.07, 0.06, ...],
      "target_prob": 0.001           // 目标词在该层的概率
    },
    {
      "layer": 1,
      "is_embedding": false,
      "topk_tokens": ["in", "the", "to", "of", ...],
      "topk_probs": [0.15, 0.11, 0.08, 0.07, ...],
      "target_prob": 0.003
    },
    ...                               // 共 n_layers+1 层（含 embedding）
    {
      "layer": 28,
      "is_embedding": false,
      "topk_tokens": ["west", "east", "north", ...],  // 最终层
      "topk_probs": [0.72, 0.08, 0.03, ...],
      "target_prob": 0.72
    }
  ],
  "is_eos": false
}
```

**关键概念 — Eureka 层**：目标词首次跃至 top-1 的那一层。例如 layer 18 时 `west` 概率首次超过所有其他候选，这就是「模型想通了这是 west 的那一层」。

---

### 3.5 Branch Tree — 续写分支展开

**做什么**：输入前缀，展开模型所有可能的续写路径（top-k tree），递归探索「模型会怎么往下写」。

**API：`POST /api/branch-next`**
```
请求：
{
  "prefix": "The sun rises in the",   // 必填
  "model": "base" | "instruct",       // 必填
  "source_page": "branch_tree",       // 必填
  "top_k": 10                         // 可选，候选数（默认10，上限50）
}

响应 200：
{
  "success": true,
  "model": "qwen3-0.6b-base",
  "prefix_tokens": 5,                  // prefix 的 token 数
  "candidates": [
    { "token": "east",  "token_id": 8423,  "prob": 0.48 },
    { "token": "west",  "token_id": 2534,  "prob": 0.22 },
    { "token": "sky",   "token_id": 6712,  "prob": 0.08 },
    ...
  ],
  "is_context_full": false             // true 时前端应禁用展开
}

错误：400 (缺参数/model非法/超长) / 500 / 503
```

**递归模式**：选择一个候选 → 拼接进 prefix → 再次调用 branch-next → 形成树。前端负责树的构建和渲染逻辑。

---

### 3.6 Causal Flow — 因果信息流

**做什么**：追踪信息如何在模型层间流动，观察归因信号沿生成步骤的传播。核心是「文本自回归生成 + 每一步的输入 token 对输出 token 的梯度归因」的串行管道。

**依赖的 API**（Causal Flow 是客户端编排的复合流程）：
1. `POST /api/tokenize` — 分词（见下方 3.6.1）
2. `POST /api/branch-next` — 获取 top-k 候选（同 3.5）
3. `POST /api/prediction-attribute` — 每一步的梯度归因（同 3.3），带 `flow_id` + `flow_step` 连续归因
4. `POST /v1/completions/prompt` — 组装 prompt（同 3.2 Step 1）
5. `POST /v1/completions` — 续写生成下一个 token

**客户端流程**：输入 context → tokenize → 自动循环：用当前 context 调用 branch-next，选 top-1 → 调用 prediction-attribute 归因当前步 → 拼接 token → 继续下一步（或手动选择分支方向）

**预计算 Demo 数据**：可加载预计算的完整归因 DAG，以 D3 力导向图或传播动画播放，无需实时推理。

---

### 3.6.1 Tokenize — 文本分词（通用工具端点）

**API：`POST /api/tokenize`**
```
请求：
{
  "context": "Hello, world!",          // 必填
  "model": "base" | "instruct"         // 必填
}

响应 200：
{
  "success": true,
  "spans": [
    { "offset": [0, 5],  "raw": "Hello" },
    { "offset": [5, 6],  "raw": "," },
    { "offset": [6, 7],  "raw": " " },
    { "offset": [7, 12], "raw": "world" },
    { "offset": [12, 13], "raw": "!" }
  ]
}

错误：400 (缺字段/model非法)
```

**特点**：不持有推理锁，不做前向/梯度计算，响应极快。是构建统一工作流的「第一步」。

---

### 3.7 🆕 Tiny-NLA Activation Explainer（新训练的模型）

**做什么**：给定一个 token 位置的激活向量（从模型 layer 19 残差流提取的 1024 维向量），用 LoRA 微调的小模型解释「这个激活向量代表了什么概念」。支持两种输入模式：传入文本让后端自动提取激活，或直接传入向量。

**模型细节**：
- **架构**：Qwen3-0.6B-Base + LoRA adapter (r=8, alpha=16) @ layer 19 + AR head（线性层 1024→1024）
- **推理精度**：float32 + eager（独立加载，不复用 base 槽位的 float16）
- **训练数据**：284 条 teacher labeled 激活-解释对
- **设备**：Apple M4 Pro MPS
- **核心机制**：用特殊 token `㈎`（id=149705）作为激活注入锚点，注入归一化后的激活向量（injection_scale=126.223），AV 模型 generate 出自然语言解释，AR head 重建后计算 cosine 评估可信度

**API：`POST /api/activation-explain`**
```
请求模式 A — 文本模式（自动提取激活）：
{
  "model": "base" | "instruct",        // 必填
  "source_page": "logit_lens",         // 必填
  "text": "water freezes into solid ice",  // 输入上下文
  "token_index": 4                     // 目标 token 位置（text 模式必填，必须 ≥ 0 且在范围内）
}

请求模式 B — 向量模式（直接传入）：
{
  "model": "base",
  "source_page": "workspace",
  "vector": [0.12, -0.34, 0.56, ...]  // 恰好 1024 个 float（与 text 互斥）
}

响应 200：
{
  "success": true,
  "concept": "",                       // 概念标签（当前为空，可后续扩展）
  "explanation": "This activation vector encodes the concept of water turning into a solid state through freezing, capturing the physical phase transition from liquid to ice.",
  "roundtrip_cosine": 0.6631,          // 重建可信度分数 (0-1)
  "vector_dim": 1024,                  // 向量维度
  "note": ""                           // 备注或状态说明
}

错误：
  400 — 缺字段、model 非法、text/vector 均未提供、token_index 越界、向量维度不为 1024
  500 — Tiny-NLA 推理失败
  503 — 推理锁等待超时（30s）
```

**可信度解读**（roundtrip_cosine）：
- **≥ 0.70** — 优秀（Excellent）：解释高度忠实于激活
- **0.60–0.70** — 良好（Good）：解释与激活有较强关联
- **0.50–0.60** — 可用（Fair）：解释方向大致正确
- **< 0.50** — 保留（Low）：解释与激活关联较弱，仅供参考

**模型性能基线**：
- AV（Activation Vector 模型）：best_val_loss = 0.64
- AR（Auto-Regressive head）：best_val_cosine = 0.6631，mean baseline cosine = 0.5962，shuffled baseline = 0.5252
- 相比随机 baseline 提升 0.07（统计显著）

## 四、核心问题

```
现状：6 个功能 = 6 个独立页面
  ❌ 同一段文本要在不同页面反复输入
  ❌ 结果之间没有关联（Logit Lens 的结果不能顺手看 Attribution）
  ❌ 新模型（Tiny-NLA）完全没接入前端
  ❌ 用户体验割裂——每个功能像独立的工具而非一个整体
```

---

## 五、设计目标

> **输入一次文本，在一个页面上以多种「透镜」/「视角」探索模型内部。**

期望的用户旅程：
1. 用户输入一段文本 → tokenize 为可交互的 token 序列
2. 用户选中某个 token → 可以从多个维度探索：
   - 📊 **Info Highlight**：看 surprisal
   - 🎯 **Attribution**：看哪些前文驱动了这个 token
   - 🔍 **Logit Lens**：看逐层解码轨迹
   - 🌳 **Branch Tree**：展开可能的续写
   - 🧠 **Activation Explainer**：解释激活向量代表什么概念
   - 🔗 **Causal Flow**：看信息流向
3. 多个分析视图可以同时打开，共享同一份输入上下文

**你可以完全自由地设计交互模式和视觉风格**，不需要考虑现有前端代码的任何约束。

---

## 六、Mock 数据 & 接口设计自由度

### 现有接口参考

所有后端的完整请求/响应 schema 见**第三章各小节**。demo 中构造 mock 数据时，请直接参考对应功能的 API 响应结构。

### 🆕 可以设计新接口

**你不必拘泥于现有接口。** 如果统一工作流的交互需要新的后端接口来达到更好的效果，完全可以提出新的 API 设计。

**前提条件**：新接口必须在我们后端能力范围内。我们后端的能力边界是：

- **运行时**：Python Flask/Connexion，单个 `server.py` 注册函数即自动生成路由
- **模型能力**：
  - Qwen3-0.6B (base + instruct) — 前向推理、logits 获取、hidden states 提取、tokenize
  - Tiny-NLA (LoRA + AR head) — 激活解释（已有）
- **可做的计算**：
  - 单个 token 的前向传播及 hidden states
  - 任意层的 hidden state 投影到词表（logit lens 的核心操作）
  - 梯度计算（需 backward）
  - 两个向量的 cosine 相似度、L2 距离
  - token 级别的概率获取
- **做不到的**：
  - 大规模 batch 训练（但我们的训练代码在 `experiments/tiny_nla/` 中，可以离线跑）
  - 需要额外预训练模型的计算（除非把模型文件放到 `artifacts/` 下）

**如果你设计了新接口，在 demo 中用 mock 数据即可。我们在后续实现阶段会在后端补上。**

关键端点速查：
- **Info Highlight** → 3.1 节 `POST /api/analyze` 响应
- **Info Highlight 语义搜索** → 3.1 节 `POST /api/analyze-semantic` 响应
- **Raw Chat** → 3.2 节 `POST /v1/completions` SSE 末条响应
- **Attribution 梯度** → 3.3 节 `POST /api/prediction-attribute` 响应
- **Attribution 消融** → 3.3 节 `POST /api/ablation-attribute` 响应
- **Logit Lens** → 3.4 节 `POST /api/logit-lens` 响应（注意 layers 数组结构）
- **Branch Tree** → 3.5 节 `POST /api/branch-next` 响应
- **Causal Flow** → 3.6 节复合流程（tokenize + branch-next + prediction-attribute 串行）
- **Tokenize** → 3.6.1 节 `POST /api/tokenize` 响应
- **🆕 Activation Explainer** → 3.7 节 `POST /api/activation-explain` 响应

---

## 七、设计资源 & 参考

### 设计灵感网站（自由参考）
- **https://21st.dev/** — 现代 UI 组件灵感和设计模式
- **https://cuicui.day/application-ui** — 应用 UI 设计参考
- **https://ui.aceternity.com/** — 创新 UI 组件和动画效果

### 本地设计仓库（内含设计规范）

**[impeccable](https://github.com/pbakaus/impeccable)** — Anti-Design-Slop 检测器
- `AGENTS.md` — 仓库规范和工作流
- `DESIGN.md` — Neo Kinpaku 设计系统（暗色漆器 + 金箔 + 铜绿），包含完整的色彩、排版、圆角、间距 token 和组件规范（button、input、card、nav-link 等）
- `CLAUDE.md` — Claude 专用的详细行为规范和交互模式
- `docs/STYLE.md` — 风格指南
- **核心哲学**：避免 AI 生成设计中的常见反模式（AI-purple gradients、三列等大 Feature Cards、过度 glassmorphism、无限循环微动画等）

**[taste-skill](https://github.com/Leonxlnx/taste-skill)** — Anti-Slop 前端 Skill
- `skills/taste-skill/SKILL.md` — 核心设计 Skill（85KB），包含：
  - 「Read the Room」——先推断设计意图再动手
  - 「三旋钮」系统（DESIGN_VARIANCE / MOTION_INTENSITY / VISUAL_DENSITY）从 1-10 控制视觉风格
  - 设计系统映射表（何时用 Fluent UI / Material / Carbon / Primer / govuk-frontend / shadcn 等）
  - 反默认纪律（避免 LLM 默认审美）
- `skills/minimalist-skill/SKILL.md` — 极简风格
- `skills/brutalist-skill/SKILL.md` — 粗野主义风格
- `skills/soft-skill/SKILL.md` — 柔和风格
- `skills/redesign-skill/SKILL.md` — 重设计方法论
- `skills/stitch-skill/SKILL.md` — 设计缝合参考

**建议在开始设计前，至少阅读 `impeccable/DESIGN.md` 和 `taste-skill/skills/taste-skill/SKILL.md` 以了解高质量设计的核心原则。**

---

## 八、交付要求

### 必须包含
1. **统一输入区**：一个文本框接受用户输入，tokenize 后展示为可交互的 token 序列
2. **至少 3 种分析入口**：包括至少 Logit Lens、Attribution、Tiny-NLA Activation Explainer
3. **多视图共存**：可以同时看到至少 2 种分析的输出（不互斥）
4. **Token 选择交互**：点击 token 触发对应分析
5. **Mock 数据**：用静态数据代替真实 API 响应

### 不需要做的
- 不需要对接真实后端 API
- 不需要实现完整的 i18n
- 不需要考虑 webpack 集成（独立 HTML 即可）
- 不需要实现 Admin / Settings 功能

### 交付物
一个或多个独立的 HTML/CSS/JS 文件，可以在浏览器中直接打开运行。

---

## 九、自由发挥空间

以下内容**完全由你决定**，不受任何限制：

- 🎨 **视觉风格**：暗色/亮色、极简/丰富、任何配色方案
- 🖼️ **布局方式**：面板/分屏/Tab/卡片网格/画布/任何创意布局
- 🎬 **动效设计**：过渡/微交互/动画，完全自由
- 📐 **交互模式**：点击/悬停/拖拽/快捷键/手势
- 🏗️ **技术选型**：原生 CSS / Tailwind / 任何 CSS 框架 / 任何 JS 库
- 📝 **文案风格**：技术性 / 产品化 / 趣味化，任意

**唯一的要求是：看起来不像"又一个 AI 生成的 demo"，而是像一个真正有设计思考的产品。**
