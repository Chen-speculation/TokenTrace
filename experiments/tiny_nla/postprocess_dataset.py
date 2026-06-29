#!/usr/bin/env python3
"""
Post-process dataset: clean teacher explanations by stripping <think> blocks.
Qwen3 instruct outputs reasoning in <think>...</think> tags.
We want only the text after </think>.

Also generates an improved dataset with augmented prompt templates.
"""

import json, re, random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
SIDECAR_PATH = Path(__file__).resolve().parent / "nla_meta.yaml"

JSONL_PATH = ARTIFACTS_DIR / "dataset.jsonl"
ACT_PATH = ARTIFACTS_DIR / "activations.pt"


def clean_explanation(text: str) -> str:
    """Strip <think> blocks, extract only the answer text."""
    if not text:
        return ""
    # Remove <think>...</think> blocks (possibly with newlines)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Also strip leading/trailing whitespace per line
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
    # If after stripping think there's still nothing, keep minimal
    if not cleaned:
        # Try to salvage something from within think
        inner = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if inner:
            cleaned = inner.group(1).strip()[:100]
    return cleaned


def is_empty_or_useless(text: str) -> bool:
    """Check if explanation is empty or just boilerplate."""
    if not text or len(text) < 5:
        return True
    useless = ["[空输出]", "[生成失败", "<think>"]
    if any(u in text for u in useless):
        return True
    return False


def main():
    print("=" * 60)
    print("🧹 Post-Processing Dataset")
    print("=" * 60)
    
    # Load dataset
    records = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {JSONL_PATH}")
    
    # Check stats before
    empty_before = sum(1 for r in records if is_empty_or_useless(r.get("teacher_explanation", "")))
    has_think = sum(1 for r in records if "<think>" in r.get("teacher_explanation", ""))
    print(f"  Before cleaning:")
    print(f"    Empty/useless: {empty_before}")
    print(f"    Contains <think>: {has_think}")
    
    # Clean explanations
    for r in records:
        raw = r.get("teacher_explanation", "")
        cleaned = clean_explanation(raw)
        r["teacher_explanation_raw"] = raw  # keep original
        r["teacher_explanation"] = cleaned
    
    empty_after = sum(1 for r in records if is_empty_or_useless(r.get("teacher_explanation", "")))
    print(f"  After cleaning:")
    print(f"    Empty/useless: {empty_after}")
    
    # Check quality
    print("\n  Sample cleaned explanations:")
    sample = random.Random(42).sample(records, 8)
    for r in sample:
        print(f"    [{r['token_text']!r:10}] {r['teacher_explanation'][:80]}")
    
    # Save cleaned dataset
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  ✅ Saved cleaned dataset to {JSONL_PATH}")
    
    # Also save a separate "av ready" format (for AV training directly)
    av_data = []
    for r in records:
        if not is_empty_or_useless(r.get("teacher_explanation", "")):
            av_data.append({
                "text": r["text"],
                "token_text": r["token_text"],
                "pos": r["pos"],
                "teacher_explanation": r["teacher_explanation"],
                "top_tokens": r["top_tokens"],
            })
    
    av_path = ARTIFACTS_DIR / "av_training_data.json"
    with open(av_path, "w", encoding="utf-8") as f:
        json.dump(av_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Saved AV-ready data ({len(av_data)} samples) to {av_path}")
    
    print("\n✅ Post-processing complete!")


if __name__ == "__main__":
    main()
