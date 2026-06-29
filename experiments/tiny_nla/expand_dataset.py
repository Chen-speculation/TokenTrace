#!/usr/bin/env python3
"""
Generate diverse Chinese texts via opencode, then extract activations and teacher labels.
Run: python expand_dataset.py
"""
import json, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
OUT_TEXTS = ARTIFACTS_DIR / "expanded_texts.json"
MODEL = "opencode/deepseek-v4-flash-free"

DOMAINS = [
    ("科技AI", 20), ("日常生活", 20), ("自然科学", 20), ("社会新闻", 20),
    ("文学文化", 20), ("经济商业", 20), ("历史哲学", 20), ("医学健康", 15),
    ("体育运动", 15), ("环境生态", 15), ("教育学习", 15), ("人际关系", 15),
    ("法律政治", 10), ("饮食美食", 10), ("旅游地理", 10),
]

PROMPT_TEMPLATE = """\
请生成 {n} 条不同的中文句子，主题是「{domain}」。

要求：
- 每条 15-50 个字，完整句子
- 内容多样，涵盖该主题的不同角度
- 语言自然，包含实词（名词、动词、形容词）和虚词（介词、连词、语气词）
- 句式多样：陈述句、疑问句、复合句都可以
- 不要编号，每行一句，直接输出句子

只输出句子，不要其他解释。"""


def call_opencode(prompt: str) -> str:
    r = subprocess.run(
        ["opencode", "run", "--model", MODEL, prompt],
        capture_output=True, text=True, timeout=120,
    )
    lines = r.stdout.splitlines()
    content = [l for l in lines if l.strip()
               and not l.strip().startswith("\x1b")
               and "> orchestrator" not in l
               and not l.strip().startswith("{")
               and not l.strip().startswith('"')]
    return "\n".join(content).strip()


def main():
    existing = []
    if OUT_TEXTS.exists():
        with open(OUT_TEXTS) as f:
            existing = json.load(f)
    print(f"Existing texts: {len(existing)}")

    existing_set = set(existing)
    all_texts = list(existing)

    for domain, n in DOMAINS:
        prompt = PROMPT_TEMPLATE.format(domain=domain, n=n)
        print(f"\n  [{domain}] requesting {n} sentences...")
        try:
            out = call_opencode(prompt)
            sentences = [l.strip() for l in out.splitlines()
                        if l.strip() and 8 <= len(l.strip()) <= 80
                        and l.strip() not in existing_set]
            print(f"    Got {len(sentences)} new sentences")
            for s in sentences[:5]:
                print(f"      {s[:50]}")
            all_texts.extend(sentences)
            existing_set.update(sentences)
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)

    # Save
    with open(OUT_TEXTS, "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)
    print(f"\nTotal texts: {len(all_texts)} → {OUT_TEXTS}")


if __name__ == "__main__":
    main()
