#!/usr/bin/env python3
"""Phase 0.1: expand Chinese text corpus to 500+ sentences via opencode (parallel)."""
import json, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "tiny_nla"
OUT = ARTIFACTS / "expanded_texts.json"
MODEL = "opencode/deepseek-v4-flash-free"

# Additional domains to push past 500
DOMAINS = [
    ("编程技术", 25), ("数学逻辑", 20), ("影视娱乐", 20), ("宗教信仰", 15),
    ("语言学", 20), ("艺术美学", 20), ("交通出行", 20), ("宠物动物", 20),
    ("天气气候", 15), ("金融投资", 20), ("职场沟通", 20), ("心理学", 20),
    ("社交媒体", 20), ("游戏娱乐", 15), ("亲子教育", 15),
]


def gen_sentences(domain_n):
    domain, n = domain_n
    # Request a mix of short/medium/long
    prompt = (
        f"请生成{n}条关于「{domain}」的中文句子。"
        "要求：包含8-15字短句、15-30字中等句、30-50字长句各占约三分之一；"
        "句式多样（陈述、疑问、复合）；语言自然，含丰富实词和虚词。"
        "不要编号，每行一句，只输出句子。"
    )
    r = subprocess.run(["opencode", "run", "--model", MODEL, prompt],
                       capture_output=True, text=True, timeout=120)
    lines = r.stdout.splitlines()
    sents = [l.strip() for l in lines
             if l.strip() and 5 <= len(l.strip()) <= 80
             and not l.strip().startswith("\x1b")
             and "orchestrator" not in l
             and not l.strip().startswith("{")]
    return domain, sents


def main():
    with open(OUT) as f:
        existing = json.load(f)
    existing_set = set(existing)
    print(f"Starting with {len(existing)} sentences")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(gen_sentences, d): d for d in DOMAINS}
        for fut in as_completed(futures):
            domain, sents = fut.result()
            new = [s for s in sents if s not in existing_set]
            existing.extend(new)
            existing_set.update(new)
            print(f"  [{domain}] +{len(new)} → total {len(existing)}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nFinal corpus: {len(existing)} sentences → {OUT}")


if __name__ == "__main__":
    main()
