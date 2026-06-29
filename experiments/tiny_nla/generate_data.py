#!/usr/bin/env python3
"""
Phase 3: Data Generation

Two-pass process:
  1. Load BASE model → extract activations for token positions in diverse texts
  2. Load INSTRUCT model → generate teacher explanations

Output: artifacts/tiny_nla/dataset.jsonl + artifacts/tiny_nla/dataset_stats.json
"""

import os, sys, json, math, yaml, time, random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = Path(__file__).resolve().parent / "nla_meta.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
INSTRUCT_MODEL = "Qwen/Qwen3-0.6B"

# ── device ─────────────────────────────────────────────
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dtype(device):
    if device.type == "cuda":
        return torch.float16
    return torch.float32  # MPS/CPU safest


# ── diverse Chinese texts ──────────────────────────────
TEXTS = [
    # Technology & AI
    "人工智能正在深刻改变各行各业的发展模式，从自动驾驶到医疗诊断，都在不断突破。",
    "深度学习模型通过多层神经网络学习数据的层次化特征表示，在自然语言处理领域取得了显著成果。",
    "量子计算是一种利用量子力学原理处理信息的新型计算范式。",
    
    # Daily life
    "今天天气真好，阳光明媚，我们一起去公园散步吧，感受大自然的美丽。",
    "这家餐厅的菜非常好吃，特别是他们的招牌菜红烧肉，味道鲜美极了。",
    "昨天晚上我看了一部很感人的电影，讲述了一个关于友情和梦想的故事。",
    
    # Science & education
    "在学习编程的过程中，最重要的是掌握基本的逻辑思维和解决问题的能力。",
    "数学是自然科学的基础，它帮助我们理解世界的规律和结构。",
    "历史上许多伟大的科学家都经历了无数次失败才取得了突破性的发现。",
    
    # News & society
    "近年来，全球气候变化问题日益受到各国政府的高度关注。",
    "数字经济正在成为推动全球经济增长的新引擎。",
    "教育公平是社会公平的重要基础，需要全社会共同努力。",
    
    # Literature & culture
    "读书是一种与作者对话的方式，通过阅读我们可以获得知识和智慧。",
    "中国传统文化的魅力在于其深厚的历史积淀和独特的哲学思想。",
    "音乐是人类共同的语言，它能够跨越国界传达情感。",
    
    # Economy & business
    "市场经济的核心在于供需关系的动态平衡。",
    "创新是企业保持竞争力的关键因素。",
    "投资理财需要长期规划和理性决策。",
    
    # Short phrases
    "请帮我解释一下这个概念。",
    "我想了解更多关于这个话题的信息。",
    "这是一个非常重要的发现。",
    "我们需要认真对待这个问题。",
]


# ── Pass 1: Extract activations ─────────────────────────
def extract_activations(model, tokenizer, texts, layer_idx, device, dtype):
    """
    For each text, extract residual stream activation at layer_idx
    for EVERY token position (not just last). Returns list of records.
    """
    print("=" * 60)
    print("📥 Pass 1: Extracting Activations")
    print("=" * 60)
    records = []
    total_tokens = 0
    
    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]  # [1, seq_len]
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
        
        # Get logits for last token position for top-k info
        logits = outputs.logits[0]  # [seq_len, vocab]
        
        # Get hidden states for our target layer
        hidden = outputs.hidden_states[layer_idx]  # [1, seq_len, d_model]
        
        for pos in range(seq_len):
            token_id = input_ids[0, pos].item()
            token_text = tokenizer.decode([token_id])
            activation = hidden[0, pos, :].cpu()  # [d_model]
            
            # Top-k at this position (from logits at this position predicting NEXT token)
            if pos < seq_len - 1:
                next_logits = logits[pos]  # logits for predicting next token after pos
                topk_vals, topk_idxs = torch.topk(next_logits, k=10)
                top_tokens = [tokenizer.decode([t]) for t in topk_idxs]
            else:
                # Last position has no "next token" prediction in base model
                # But we still extract the activation for it
                next_logits = logits[pos]
                topk_vals, topk_idxs = torch.topk(next_logits, k=10)
                top_tokens = [tokenizer.decode([t]) for t in topk_idxs]
            
            records.append({
                "text_idx": idx,
                "text": text,
                "pos": pos,
                "token_id": token_id,
                "token_text": token_text,
                "activation_vector": activation.tolist(),
                "top_tokens": top_tokens,
                "activation_norm": activation.norm().item(),
            })
            total_tokens += 1
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(texts)} texts ({total_tokens} tokens so far)")
    
    print(f"  ✅ Extracted {total_tokens} activations from {len(texts)} texts")
    return records


# ── Pass 2: Generate teacher explanations ──────────────
def generate_teacher_explanations(
    model, tokenizer, records, device, dtype,
    max_samples=500, batch_prompt=False,
):
    """
    For each record, generate a teacher explanation using the instruct model.
    
    Teacher prompt (instruct format):
      Given a context and token position, explain what the model might be focusing on.
    """
    print("=" * 60)
    print("📝 Pass 2: Generating Teacher Explanations")
    print("=" * 60)
    
    # Determine instruct chat template format
    # Qwen3 instruct uses: <|im_start|>system...<|im_end|> etc
    # But for base model style, just use a simple prompt
    
    system_msg = "你是一个模型可解释性专家。给定一段文本和其中的某个位置，用1-2句中文简短解释模型在该位置可能关注什么语义信息。不要长篇分析。"
    
    # Sample records, ensuring diversity across texts
    random.seed(42)
    n_texts = len(set(r["text_idx"] for r in records))
    samples_per_text = max(1, max_samples // n_texts)
    
    # Stratified sample: take at least 15 tokens per text (or all if fewer available)
    texts_grouped = {}
    for r in records:
        texts_grouped.setdefault(r["text_idx"], []).append(r)
    
    sampled_records = []
    for tidx in sorted(texts_grouped.keys()):
        group = texts_grouped[tidx]
        # Take evenly spaced positions for diversity
        n = min(len(group), max(15, samples_per_text))
        indices = sorted(random.sample(range(len(group)), min(n, len(group))))
        for i in indices:
            sampled_records.append(group[i])
    
    # If we have fewer than target, take more
    if len(sampled_records) < max_samples and len(sampled_records) < len(records):
        existing_ids = set(id(r) for r in sampled_records)
        more = [r for r in records if id(r) not in existing_ids]
        remaining = max_samples - len(sampled_records)
        sampled_records.extend(random.sample(more, min(remaining, len(more))))
    
    print(f"  Target: {max_samples} samples, selected {len(sampled_records)} for labeling")
    
    results = []
    teacher_failures = 0
    
    for i, rec in enumerate(sampled_records):
        text = rec["text"]
        pos = rec["pos"]
        token_text = rec["token_text"]
        top_tokens = rec["top_tokens"]
        
        # Build instruction prompt
        user_msg = (
            f"文本：{text}\n"
            f"位置：第{pos}个token（文本位置），该token文本是「{token_text}」\n"
            f"该位置模型预测的下一个token候选：{', '.join(top_tokens[:5])}\n\n"
            f"请用1-2句中文解释：模型在这个位置可能关注什么？"
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            # Extract the generated part (skip input prompt)
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            explanation = tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            if not explanation or len(explanation) < 5:
                explanation = "[空输出]"
                teacher_failures += 1
                
        except Exception as e:
            explanation = f"[生成失败: {e}]"
            teacher_failures += 1
        
        rec["teacher_explanation"] = explanation
        results.append(rec)
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(sampled_records)}] token={token_text!r} -> {explanation[:60]}...")
    
    print(f"\n  ✅ Generated {len(results)} explanations")
    print(f"  ⚠️  Failures/empty: {teacher_failures}/{len(results)}")
    return results


# ── Save dataset ────────────────────────────────────────
def save_dataset(records, nla_meta: dict):
    """Save dataset as JSONL + stats."""
    # Strip bulky activation vectors for JSONL (keep as separate array file)
    jsonl_path = ARTIFACTS_DIR / "dataset.jsonl"
    # Save activations as a separate .pt file for efficiency
    act_path = ARTIFACTS_DIR / "activations.pt"
    
    activation_tensors = []
    json_records = []
    
    for r in records:
        act = r.pop("activation_vector")
        activation_tensors.append(torch.tensor(act))
        json_records.append(r)
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in json_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    torch.save(torch.stack(activation_tensors), act_path)
    
    print(f"\n  💾 Dataset saved:")
    print(f"     {jsonl_path} ({len(json_records)} records)")
    print(f"     {act_path} ({activation_tensors[0].shape})")
    
    # Stats
    n_texts = len(set(r["text_idx"] for r in json_records))
    has_explanation = sum(1 for r in json_records if r.get("teacher_explanation"))
    empty_explanation = sum(1 for r in json_records if r.get("teacher_explanation") in ("[空输出]", None, ""))
    
    stats = {
        "total_records": len(json_records),
        "texts_used": n_texts,
        "records_with_explanation": has_explanation,
        "records_empty_explanation": empty_explanation,
        "d_model": activation_tensors[0].shape[0],
    }
    
    stats_path = ARTIFACTS_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"     {stats_path}")
    
    # Update nla_meta
    nla_meta["training"]["dataset_size"] = len(json_records)
    with open(SIDECAR_PATH, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(nla_meta, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"     sidecar updated with dataset_size={len(json_records)}")
    
    return json_records


# ── main ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("📦 TINY-NLA DATA GENERATION")
    print("=" * 60)
    print()
    
    device = detect_device()
    dtype = get_dtype(device)
    print(f"  Device: {device}, dtype: {dtype}")
    
    # Load sidecar
    with open(SIDECAR_PATH, "r") as f:
        import yaml
        nla_meta = yaml.safe_load(f)
    
    layer_idx = nla_meta["layer_index"]
    max_samples = 500
    
    # ── Pass 1: Extract activations ──
    print(f"\n  Loading BASE model for activation extraction...")
    t0 = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    base_model.eval()
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"  Base model loaded in {time.perf_counter() - t0:.1f}s")
    
    all_records = extract_activations(base_model, base_tokenizer, TEXTS, layer_idx, device, dtype)
    
    # Clean up base model to free memory
    del base_model
    if device.type == "mps":
        torch.mps.empty_cache()
    print("  Base model freed from memory\n")
    
    # ── Pass 2: Generate teacher explanations ──
    print(f"  Loading INSTRUCT model for teacher...")
    t0 = time.perf_counter()
    instruct_model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    instruct_model.eval()
    instruct_tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=True)
    print(f"  Instruct model loaded in {time.perf_counter() - t0:.1f}s")
    
    results = generate_teacher_explanations(
        instruct_model, instruct_tokenizer, all_records,
        device, dtype, max_samples=max_samples,
    )
    
    # Clean up
    del instruct_model
    if device.type == "mps":
        torch.mps.empty_cache()
    
    # ── Save ──
    json_records = save_dataset(results, nla_meta)
    
    # ── Quality check ──
    print()
    print("=" * 60)
    print("🔍 Quick Quality Check")
    print("=" * 60)
    sample = random.Random(42).sample(json_records, min(10, len(json_records)))
    for r in sample:
        exp = r.get("teacher_explanation", "")
        print(f"  [{r['token_text']!r:10}] {exp[:80]}")
    
    print()
    print("✅ Data generation complete!")
    print(f"  Dataset: {ARTIFACTS_DIR}/dataset.jsonl")
    print(f"  Activations: {ARTIFACTS_DIR}/activations.pt")
    print(f"  Stats: {ARTIFACTS_DIR}/dataset_stats.json")
    print(f"  Sidecar: {SIDECAR_PATH}")


if __name__ == "__main__":
    main()
