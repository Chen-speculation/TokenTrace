#!/usr/bin/env python3
"""
Batch teacher label generation via opencode (parallel workers).
Usage: python batch_teacher_labels.py [--workers 3]
"""
import json, subprocess, sys, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "tiny_nla"
RECORDS_FILE = ARTIFACTS / "expanded_records.jsonl"
OUTPUT = ARTIFACTS / "expanded_teacher_labels.json"
MODEL = "opencode/deepseek-v4-flash-free"

PROMPT = """\
你是语言模型内部机制分析专家。用1-2句简洁中文描述Qwen3-0.6B在处理下面这个token时，该位置residual stream激活值编码的语义信息。

上下文：{context}
当前token：「{token}」（位置{pos}/{seq_len}）
模型预测下一个词的候选：{top_tokens}

要求：简洁具体，联系token在句中的实际句法/语义角色，联系预测候选推断编码内容，不超过55字。只输出解释本身。"""


def call_opencode(record: dict) -> str:
    prompt = PROMPT.format(
        context=record["text"],
        token=record["token_text"],
        pos=record["pos"],
        seq_len=record["seq_len"],
        top_tokens="、".join(record["top_tokens"][:5]),
    )
    r = subprocess.run(
        ["opencode", "run", "--model", MODEL, prompt],
        capture_output=True, text=True, timeout=90,
    )
    lines = r.stdout.splitlines()
    content = [l.strip() for l in lines
               if l.strip()
               and not l.strip().startswith("\x1b")
               and "orchestrator" not in l
               and not l.strip().startswith("{")
               and not l.strip().startswith('"message"')]
    return "\n".join(content).strip()


def process_record(idx_rec):
    idx, rec = idx_rec
    try:
        expl = call_opencode(rec)
        # Strip think blocks
        if "<think>" in expl and "</think>" in expl:
            expl = expl.split("</think>")[-1].strip()
        return idx, {**rec, "teacher_explanation": expl, "teacher_source": "opencode/deepseek-v4-flash-free"}
    except Exception as e:
        return idx, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(RECORDS_FILE) as f:
        records = [json.loads(l) for l in f]

    if args.limit:
        records = records[:args.limit]

    # Load existing
    existing = {}
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for item in json.load(f):
                existing[(item["text_idx"], item["pos"])] = item
    print(f"Records: {len(records)}, existing labels: {len(existing)}")

    todo = [(i, r) for i, r in enumerate(records)
            if (r["text_idx"], r["pos"]) not in existing]
    print(f"To process: {len(todo)}")

    results = dict(existing)
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_record, item): item for item in todo}
        for fut in as_completed(futures):
            idx, result = fut.result()
            if result:
                results[(result["text_idx"], result["pos"])] = result
            done += 1
            if done % 50 == 0 or done == len(todo):
                # Checkpoint save
                with open(OUTPUT, "w", encoding="utf-8") as f:
                    json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
                print(f"  [{done}/{len(todo)}] saved {len(results)} labels")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)
    print(f"\nDone: {len(results)} teacher labels → {OUTPUT}")


if __name__ == "__main__":
    main()
