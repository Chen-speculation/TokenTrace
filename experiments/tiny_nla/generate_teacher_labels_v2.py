#!/usr/bin/env python3
"""
Three-channel parallel teacher label generation.
Output: append-mode JSONL (one record per line) - no corruption possible.
Channels: NVIDIA API + OpenRouter (gpt-oss-120b, nex-n2-pro) + opencode CLI
Usage: python generate_teacher_labels_v2.py [--workers N]
"""
import json, os, subprocess, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "tiny_nla"
RECORDS_FILE = ARTIFACTS / "records_v2.jsonl"
OUTPUT_JSONL = ARTIFACTS / "teacher_labels_v2.jsonl"  # append-mode, safe for concurrent writes
OUTPUT_JSON  = ARTIFACTS / "teacher_labels_v2.json"   # final merged output

NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
OR_KEY     = os.environ.get("OPENROUTER_API_KEY", "")
DS_KEY     = os.environ.get("DEEPSEEK_API_KEY", "")
FW_KEY     = os.environ.get("FIREWORKS_API_KEY", "")

nvidia_client  = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_KEY)
or_client      = OpenAI(base_url="https://openrouter.ai/api/v1",         api_key=OR_KEY)
ds_client      = OpenAI(base_url="https://api.deepseek.com",              api_key=DS_KEY)
fw_client      = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=FW_KEY)

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


def call_nvidia(r):
    resp = nvidia_client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-pro",
        messages=[{"role": "user", "content": make_prompt(r)}], max_tokens=256)
    return strip_think(resp.choices[0].message.content or ""), "nvidia"


def call_gpt_oss(r):
    resp = or_client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[{"role": "user", "content": make_prompt(r)}], max_tokens=256)
    return strip_think(resp.choices[0].message.content or ""), "or-gpt-oss"


def call_nex(r):
    resp = or_client.chat.completions.create(
        model="nex-agi/nex-n2-pro:free",
        messages=[{"role": "user", "content": make_prompt(r)}], max_tokens=256)
    return strip_think(resp.choices[0].message.content or ""), "or-nex"


def call_deepseek(r):
    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": make_prompt(r)}], max_tokens=256)
    return strip_think(resp.choices[0].message.content or ""), "deepseek"


def call_fireworks(r):
    resp = fw_client.chat.completions.create(
        model="accounts/fireworks/models/gpt-oss-120b",
        messages=[{"role": "user", "content": make_prompt(r)}], max_tokens=256)
    return strip_think(resp.choices[0].message.content or ""), "fireworks"


# DeepSeek + Fireworks as primary (both confirmed working), others as fallback
CHANNELS = [call_deepseek, call_fireworks, call_deepseek, call_fireworks,
            call_deepseek, call_fireworks, call_nvidia, call_gpt_oss]


def process(args):
    idx, rec, ch_idx = args
    order = [CHANNELS[(ch_idx + i) % len(CHANNELS)] for i in range(len(CHANNELS))]
    for fn in order:
        try:
            expl, src = fn(rec)
            if expl and len(expl) >= 5:
                return {**rec, "teacher_explanation": expl, "teacher_source": src}
        except Exception:
            continue
    return None


def load_done():
    """Load all completed keys from existing JSON and JSONL."""
    done = {}
    if OUTPUT_JSON.exists():
        try:
            for r in json.load(open(OUTPUT_JSON)):
                done[(r["text_idx"], r["pos"])] = True
        except Exception:
            pass
    if OUTPUT_JSONL.exists():
        for line in open(OUTPUT_JSONL):
            try:
                r = json.loads(line)
                done[(r["text_idx"], r["pos"])] = True
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(RECORDS_FILE)]
    done = load_done()
    todo = [(i, r, i) for i, r in enumerate(records)
            if (r["text_idx"], r["pos"]) not in done]

    print(f"Records: {len(records)}, done: {len(done)}, todo: {len(todo)}, workers: {args.workers}")
    print(f"Channels: nvidia / gpt-oss / nex-n2-pro / opencode (round-robin)")

    completed = 0
    errors = 0
    t0 = time.time()

    # Open jsonl in append mode - each write is atomic (small line, OS guarantees)
    out_f = open(OUTPUT_JSONL, "a", encoding="utf-8", buffering=1)  # line-buffered

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, item): item for item in todo}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
            else:
                errors += 1

            n = completed + errors
            if n % 200 == 0:
                rate = n / (time.time() - t0) * 60
                remaining = len(todo) - n
                eta_min = remaining / (rate / 60) if rate > 0 else 0
                print(f"  [{len(done)+completed}/{len(records)}] errors={errors} "
                      f"rate={rate:.0f}/min ETA={eta_min/60:.1f}h")

    out_f.close()

    # Merge jsonl → final json
    all_done = {}
    if OUTPUT_JSON.exists():
        try:
            for r in json.load(open(OUTPUT_JSON)):
                all_done[(r["text_idx"], r["pos"])] = r
        except Exception:
            pass
    for line in open(OUTPUT_JSONL):
        try:
            r = json.loads(line)
            all_done[(r["text_idx"], r["pos"])] = r
        except Exception:
            pass
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(list(all_done.values()), f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(all_done)} labels → {OUTPUT_JSON}")

    # Sample
    items = list(all_done.values())[-5:]
    for item in items:
        print(f"  [{item['teacher_source']}] {item['token_text']} → {item['teacher_explanation'][:65]}")


if __name__ == "__main__":
    main()
