#!/usr/bin/env python3
"""opencode-only teacher label generation - runs in background."""
import json, subprocess, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "tiny_nla"
RECORDS_FILE = ARTIFACTS / "records_v2.jsonl"
OUTPUT_JSONL = ARTIFACTS / "teacher_labels_v2.jsonl"

PROMPT = """\
你是语言模型内部机制分析专家。用1-2句简洁中文描述Qwen3-0.6B在处理下面这个token时，该位置residual stream激活值编码的语义信息。

上下文：{context}
当前token：「{token}」（位置{pos}/{seq_len}）
模型预测下一个词的候选：{top_tokens}

要求：简洁具体，联系token在句中的实际句法/语义角色，联系预测候选推断编码内容，不超过55字。只输出解释本身。"""


def make_prompt(r):
    return PROMPT.format(context=r["text"], token=r["token_text"],
                         pos=r["pos"], seq_len=r["seq_len"],
                         top_tokens="、".join(r["top_tokens"][:5]))


def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def call_opencode(r):
    res = subprocess.run(
        ["opencode", "run", "--model", "opencode/deepseek-v4-flash-free", make_prompt(r)],
        capture_output=True, text=True, timeout=240,
    )
    lines = [l.strip() for l in res.stdout.splitlines()
             if l.strip() and not l.startswith("\x1b") and "orchestrator" not in l
             and not l.startswith("{") and not l.startswith('"')]
    return strip_think("\n".join(lines))


def load_done():
    done = set()
    try:
        for r in json.load(open(ARTIFACTS / "teacher_labels_v2.json")):
            done.add((r["text_idx"], r["pos"]))
    except Exception:
        pass
    if OUTPUT_JSONL.exists():
        for line in open(OUTPUT_JSONL):
            try:
                r = json.loads(line)
                done.add((r["text_idx"], r["pos"]))
            except Exception:
                pass
    return done


def process(idx_rec):
    idx, rec = idx_rec
    try:
        expl = call_opencode(rec)
        if expl and len(expl) >= 5:
            return {**rec, "teacher_explanation": expl, "teacher_source": "opencode"}
    except Exception as e:
        print(f"  err [{idx}]: {e}", flush=True)
    return None


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    records = [json.loads(l) for l in open(RECORDS_FILE)]
    done = load_done()
    todo = [(i, r) for i, r in enumerate(records)
            if (r["text_idx"], r["pos"]) not in done]
    print(f"Remaining: {len(todo)}/9810, workers: {workers}", flush=True)

    completed = 0
    t0 = time.time()
    out_f = open(OUTPUT_JSONL, "a", encoding="utf-8", buffering=1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, item): item for item in todo}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 20 == 0:
                    rate = completed / (time.time() - t0) * 60
                    eta_h = (len(todo) - completed) / (rate / 60) / 3600
                    print(f"  [{len(done)+completed}/{9810}] {rate:.0f}/min ETA={eta_h:.1f}h", flush=True)

    out_f.close()
    print(f"Done: {completed} new labels written", flush=True)


if __name__ == "__main__":
    main()
