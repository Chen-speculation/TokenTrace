#!/usr/bin/env python
"""
消融归因冒烟脚本。

用法（本地服务已启动）:
    python scripts/smoke_ablation_attribute.py
    python scripts/smoke_ablation_attribute.py --base-url http://localhost:7860

验证：
- 正常请求返回 200，score 方向合理（高影响 token 有正 score）
- 各 400 错误场景返回预期状态码
"""
import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("requests not installed; run: pip install requests")
    sys.exit(1)

_DEFAULT_URL = "http://localhost:7860"


def post(base_url: str, payload: dict) -> requests.Response:
    return requests.post(f"{base_url}/api/ablation-attribute", json=payload, timeout=120)


def run(base_url: str) -> int:
    failures = 0

    # ------------------------------------------------------------------ #
    # Case 1: 正常 top-1（中文，ΔP 方向核对）
    # ------------------------------------------------------------------ #
    print("\n[1] 正常 top-1 — 中国的首都是")
    r = post(base_url, {"context": "中国的首都是", "model": "base", "source_page": "attribution"})
    if r.status_code != 200:
        print(f"  ✗ status={r.status_code}: {r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        assert body["success"], body
        ta = body["token_attribution"]
        print(f"  ✓ target={body['target_token']!r}, baseline_prob={body['target_prob']:.4f}")
        for t in ta:
            bar = "+" * max(0, int(t["score"] * 50)) if t["score"] > 0 else "-" * max(0, int(-t["score"] * 50))
            print(f"    {t['raw']!r:8s} score={t['score']:+.4f}  {bar}")
        positive_scores = sum(1 for t in ta if t["score"] > 0)
        if positive_scores == 0:
            print("  ⚠ 所有 score 均 ≤ 0，结果可疑")
            failures += 1
        else:
            print(f"  ✓ {positive_scores}/{len(ta)} token 有正向贡献")

    # ------------------------------------------------------------------ #
    # Case 2: 显式 target_prediction
    # ------------------------------------------------------------------ #
    print("\n[2] 显式 target_prediction=北京")
    r = post(base_url, {
        "context": "中国的首都是",
        "model": "base",
        "source_page": "attribution",
        "target_prediction": "北京",
    })
    if r.status_code != 200:
        print(f"  ✗ status={r.status_code}: {r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        print(f"  ✓ target={body['target_token']!r}, baseline_prob={body['target_prob']:.4f}")

    # ------------------------------------------------------------------ #
    # Case 3: 互斥目标 → 400
    # ------------------------------------------------------------------ #
    print("\n[3] 互斥 target — 预期 400")
    r = post(base_url, {
        "context": "hello",
        "model": "base",
        "source_page": "attribution",
        "target_prediction": "world",
        "target_token_id": 5,
    })
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    # ------------------------------------------------------------------ #
    # Case 4: 非法 model → 400
    # ------------------------------------------------------------------ #
    print("\n[4] 非法 model — 预期 400")
    r = post(base_url, {"context": "hello", "model": "gpt4", "source_page": "attribution"})
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    # ------------------------------------------------------------------ #
    # Case 5: 缺 context → 400
    # ------------------------------------------------------------------ #
    print("\n[5] 缺 context — 预期 400")
    r = post(base_url, {"model": "base", "source_page": "attribution"})
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    # ------------------------------------------------------------------ #
    # Case 6: 非法 source_page → 400
    # ------------------------------------------------------------------ #
    print("\n[6] 非法 source_page — 预期 400")
    r = post(base_url, {"context": "hello", "model": "base", "source_page": "nowhere"})
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*40}")
    if failures == 0:
        print("✅ All smoke tests passed.")
    else:
        print(f"❌ {failures} test(s) failed.")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Ablation attribution smoke test")
    parser.add_argument("--base-url", default=_DEFAULT_URL)
    args = parser.parse_args()
    sys.exit(run(args.base_url))


if __name__ == "__main__":
    main()
