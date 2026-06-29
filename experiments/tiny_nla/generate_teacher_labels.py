#!/usr/bin/env python3
"""
Re-generate teacher explanations with correct parameters.
Qwen3-0.6B instruct uses <think> blocks; we:
  1. Use max_new_tokens=200 (enough for CoT + answer)
  2. Add explicit instruction to suppress CoT
  3. Post-process to strip any remaining <think> blocks
"""

import json, re, os, time, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "tiny_nla"
INSTRUCT_MODEL = "Qwen/Qwen3-0.6B"

def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def clean_explanation(text: str) -> str:
    """Strip <think> blocks and normalize."""
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Remove any remaining XML-like tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned).strip()
    return cleaned

def generate_explanations(records, batch_size=1, max_new_tokens=200):
    device = detect_device()
    dtype = torch.float32
    
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Generating {len(records)} explanations with max_new_tokens={max_new_tokens}")
    
    # Load instruct model
    print("  Loading Qwen3-0.6B instruct...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=True)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s\n")
    
    # Strong instruction to avoid reasoning blocks
    system_msg = "你是一个模型可解释性专家。用1-2句中文简短回答，不要分析过程，不要推理步骤，直接给出解释。"
    
    results = []
    failures = 0
    
    for i, rec in enumerate(records):
        text = rec.get("text", "")
        pos = rec.get("pos", 0)
        token_text = rec.get("token_text", "")
        top_tokens = rec.get("top_tokens", [])
        
        user_msg = (
            f"文本：{text}\n"
            f"位置：第{pos}个token，该token文本是「{token_text}」\n"
            f"预测的下一个token候选：{', '.join(top_tokens[:5])}\n\n"
            f"用1-2句中文解释模型在该位置关注什么语义信息。直接回答，不要思考过程。"
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        raw_explanation = ""
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy for speed
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_explanation = tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            explanation = clean_explanation(raw_explanation)
            if not explanation or len(explanation) < 5:
                explanation = "[空输出]"
                failures += 1
                
        except Exception as e:
            explanation = f"[生成失败: {e}]"
            failures += 1
        
        rec["teacher_explanation_raw"] = raw_explanation
        rec["teacher_explanation"] = explanation
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(records)}] failures={failures}", flush=True)
        if i < 3:
            print(f"  [{i+1}] {token_text!r:10} -> {explanation[:80]}")
    
    print(f"\n  ✅ Generated {len(records)} explanations, failures={failures}")
    
    # Clean up
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    
    return records


def main():
    print("=" * 60)
    print("🔄 Re-Generating Teacher Explanations")
    print("=" * 60)
    
    # Load existing dataset
    jsonl_path = ARTIFACTS_DIR / "dataset.jsonl"
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {jsonl_path}")
    
    # Regenerate explanations
    records = generate_explanations(records, max_new_tokens=200)
    
    # Quality stats
    valid = [r for r in records if r.get("teacher_explanation") and r["teacher_explanation"] not in ("[空输出]", "")]
    empty = len(records) - len(valid)
    has_think = sum(1 for r in records if "<think>" in (r.get("teacher_explanation", "") or ""))
    
    print(f"\n  Quality: {len(valid)} valid, {empty} empty, {has_think} still has think tag")
    
    # Save
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✅ Saved to {jsonl_path}")
    
    # Save AV training data
    av_data = [{
        "text": r["text"],
        "token_text": r["token_text"],
        "pos": r["pos"],
        "teacher_explanation": r["teacher_explanation"],
        "top_tokens": r["top_tokens"],
    } for r in records if r.get("teacher_explanation") and r["teacher_explanation"] not in ("[空输出]", "")]
    
    av_path = ARTIFACTS_DIR / "av_training_data.json"
    with open(av_path, "w", encoding="utf-8") as f:
        json.dump(av_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ AV training data: {av_path} ({len(av_data)} samples)")
    
    # Sample
    print("\n  Samples:")
    for r in random.Random(42).sample(av_data, min(5, len(av_data))):
        print(f"    [{r['token_text']!r:10}] {r['teacher_explanation'][:80]}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
