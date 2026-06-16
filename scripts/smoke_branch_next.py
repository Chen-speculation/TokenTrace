#!/usr/bin/env python
"""
分叉树 branch-next 冒烟脚本。

用法:
    python scripts/smoke_branch_next.py
    python scripts/smoke_branch_next.py --base-url http://localhost:7860
"""
import argparse
import sys

try:
    import requests
except ImportError:
    print("requests not installed; run: pip install requests")
    sys.exit(1)

_DEFAULT_URL = "http://localhost:7860"


def post(base_url, payload):
    return requests.post(f"{base_url}/api/branch-next", json=payload, timeout=120)


def run(base_url):
    failures = 0

    print("\n[1] 正常 top-10 — 中国的首都")
    r = post(base_url, {"prefix": "中国的首都", "model": "base", "source_page": "causal_flow"})
    if r.status_code != 200:
        print(f"  ✗ {r.status_code}: {r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        print(f"  ✓ prefix_tokens={body['prefix_tokens']}, is_context_full={body['is_context_full']}")
        print(f"  top-3: {[(c['token'], c['prob']) for c in body['candidates'][:3]]}")
        # 概率降序
        probs = [c["prob"] for c in body["candidates"]]
        assert probs == sorted(probs, reverse=True), "candidates not sorted by prob"
        print("  ✓ 候选按概率降序")
        top1 = body["candidates"][0]["token"]
        print(f"  top-1 token: {top1!r}")

    print("\n[2] 自定义 top_k=3")
    r = post(base_url, {"prefix": "The capital of France is", "model": "base", "source_page": "causal_flow", "top_k": 3})
    if r.status_code != 200:
        print(f"  ✗ {r.status_code}: {r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        assert len(body["candidates"]) == 3
        print(f"  ✓ 3 candidates: {[c['token'] for c in body['candidates']]}")

    print("\n[3] top_k 超上限自动 clamp")
    r = post(base_url, {"prefix": "hello", "model": "base", "source_page": "causal_flow", "top_k": 999})
    if r.status_code != 200:
        print(f"  ✗ {r.status_code}: {r.text[:200]}")
        failures += 1
    else:
        from backend.core.branch_next import BRANCH_NEXT_TOP_K_MAX
        body = r.json()
        assert len(body["candidates"]) <= BRANCH_NEXT_TOP_K_MAX
        print(f"  ✓ clamped to {len(body['candidates'])} candidates")

    print("\n[4] 缺 prefix — 预期 400")
    r = post(base_url, {"model": "base", "source_page": "causal_flow"})
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    print("\n[5] 非法 model — 预期 400")
    r = post(base_url, {"prefix": "hello", "model": "gpt4", "source_page": "causal_flow"})
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    print(f"\n{'='*40}")
    print("✅ All smoke tests passed." if failures == 0 else f"❌ {failures} test(s) failed.")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=_DEFAULT_URL)
    sys.exit(run(parser.parse_args().base_url))


if __name__ == "__main__":
    main()
