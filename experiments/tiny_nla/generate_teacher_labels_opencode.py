#!/usr/bin/env python3
"""
Generate high-quality teacher labels via opencode CLI (or any LLM).

Usage:
  python generate_teacher_labels_opencode.py

This script:
1. Reads dataset.jsonl (activation records with context/token/top_tokens)
2. For each record, prints a prompt for the LLM (opencode/Claude/etc.)
3. Reads LLM output and saves to teacher_labels_hq.json

The output file can then be merged into av_training_data.json for retraining.

Run mode options:
  --dry-run       Print prompts only (for review), no LLM calls
  --local         Use local Qwen3-0.6B instruct as fallback teacher
  --opencode-cmd  Path/name of opencode CLI (default: "opencode")
"""

import json, sys, subprocess, argparse, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
DATASET = ARTIFACTS_DIR / "dataset.jsonl"
OUTPUT = ARTIFACTS_DIR / "teacher_labels_hq.json"


TEACHER_PROMPT = """\
你是一个语言模型内部机制分析专家。我需要你用 1-2 句简洁中文，描述一个 Transformer 语言模型（Qwen3-0.6B）在处理特定 token 时，该位置的 residual stream activation 所编码的语义信息。

---
上下文句子：{context}
当前分析的 token：「{token}」（位置 {pos}，共 {seq_len} 个 token）
模型在此位置预测的下一个词（概率最高的候选）：{top_tokens}
---

请根据上面信息，用 1-2 句中文描述：
- 模型在这个 token 的位置正在编码什么语义信息
- 这个位置的激活值如何帮助预测后续内容

要求：
- 简洁具体，不要泛泛而谈
- 联系「{token}」在句子中的实际作用
- 联系模型预测的下一个词来推断编码内容
- 不要以"这个位置"或"激活值"开头，而是直接描述语义
- 不超过 60 字

只输出解释本身，不要输出分析过程。"""


def make_prompt(rec):
    top5 = "、".join(rec["top_tokens"][:5])
    seq_len = len(rec["text"])  # approximate
    return TEACHER_PROMPT.format(
        context=rec["text"],
        token=rec["token_text"],
        pos=rec["pos"],
        seq_len=seq_len,
        top_tokens=top5,
    )


def call_opencode(prompt_text, opencode_cmd="opencode", model="opencode/deepseek-v4-flash-free"):
    """Call opencode CLI with a prompt, return response text."""
    result = subprocess.run(
        [opencode_cmd, "run", "--model", model, prompt_text],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"opencode error: {result.stderr[:200]}")
    # Strip ANSI escape codes and header lines
    output = result.stdout
    lines = output.splitlines()
    # Skip lines starting with ANSI/control chars or "> orchestrator"
    content_lines = [
        l for l in lines
        if l.strip() and not l.strip().startswith("\x1b") and not l.strip().startswith("> orchestrator")
    ]
    return "\n".join(content_lines).strip()


def call_local_qwen(prompt_text):
    """Fallback: use local Qwen3-0.6B instruct."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hasattr(call_local_qwen, "_model"):
        print("  Loading local Qwen3-0.6B instruct...")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", trust_remote_code=True, dtype=torch.float32,
            low_cpu_mem_usage=True, attn_implementation="eager",
        )
        mdl.eval()
        call_local_qwen._tok = tok
        call_local_qwen._model = mdl

    tok, mdl = call_local_qwen._tok, call_local_qwen._model
    msgs = [{"role": "user", "content": prompt_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(
            **inp, max_new_tokens=128, do_sample=False,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
    gen = out[0][inp["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print prompts, no LLM calls")
    parser.add_argument("--local", action="store_true", help="Use local Qwen instruct")
    parser.add_argument("--opencode-cmd", default="opencode", help="opencode CLI command")
    parser.add_argument("--model", default="opencode/deepseek-v4-flash-free", help="opencode model")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip if output exists")
    args = parser.parse_args()

    with open(DATASET) as f:
        records = [json.loads(l) for l in f]

    if args.limit:
        records = records[:args.limit]

    # Filter out OOD first-token records (norm > 2000) — they degrade training
    records = [r for r in records if r["activation_norm"] < 2000]
    print(f"Filtered to {len(records)} in-distribution records (norm < 2000)")

    # Load existing results if any
    existing = {}
    if OUTPUT.exists() and args.skip_existing:
        with open(OUTPUT) as f:
            existing_list = json.load(f)
        existing = {(r["text_idx"], r["pos"]): r for r in existing_list}
        print(f"Loaded {len(existing)} existing labels")

    results = list(existing.values())
    new_count = 0

    for i, rec in enumerate(records):
        key = (rec["text_idx"], rec["pos"])
        if key in existing:
            continue

        prompt = make_prompt(rec)

        if args.dry_run:
            print(f"\n=== Record {i}: [{rec['token_text']}] pos={rec['pos']} ===")
            print(prompt)
            print("---")
            continue

        try:
            if args.local:
                explanation = call_local_qwen(prompt)
            else:
                explanation = call_opencode(prompt, args.opencode_cmd, args.model)

            # Clean up: strip think blocks if any
            if "<think>" in explanation:
                if "</think>" in explanation:
                    explanation = explanation.split("</think>")[-1].strip()
                else:
                    explanation = explanation.split("<think>")[0].strip()

            result = {
                "text_idx": rec["text_idx"],
                "pos": rec["pos"],
                "text": rec["text"],
                "token_text": rec["token_text"],
                "top_tokens": rec["top_tokens"][:5],
                "activation_norm": rec["activation_norm"],
                "teacher_explanation": explanation,
                "teacher_source": "local_qwen" if args.local else "opencode",
            }
            results.append(result)
            new_count += 1

            if new_count % 10 == 0 or i == 0:
                # Save checkpoint
                with open(OUTPUT, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  [{i+1}/{len(records)}] saved {len(results)} labels")
                print(f"    [{rec['token_text']}] → {explanation[:60]}")

            time.sleep(0.3)  # gentle rate limit

        except Exception as e:
            print(f"  Error on record {i}: {e}", file=sys.stderr)
            time.sleep(2)

    if not args.dry_run:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDone. {len(results)} total labels saved to {OUTPUT}")
    else:
        print(f"\nDry run complete. {len(records)} prompts shown.")


if __name__ == "__main__":
    main()
