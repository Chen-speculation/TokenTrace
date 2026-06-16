#!/usr/bin/env python
"""
Logit Lens 冒烟脚本。

用法:
    python scripts/smoke_logit_lens.py
    python scripts/smoke_logit_lens.py --base-url http://localhost:7860
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
    return requests.post(f"{base_url}/api/logit-lens", json=payload, timeout=120)


def run(base_url):
    failures = 0

    print("\n[1] 正常 top-1 — 中国的首都是")
    r = post(base_url, {"context": "中国的首都是", "model": "base", "source_page": "attribution"})
    if r.status_code != 200:
        print(f"  ✗ {r.status_code}: {r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        n = body["n_layers"]
        layers = body["layers"]
        print(f"  ✓ target={body['target_token']!r}, n_layers={n}, final_prob={body['final_target_prob']:.4f}")
        # 核对目标 token 概率轨迹大致单调上升趋势（对比浅层 vs 深层均值）
        probs = [l["target_prob"] for l in layers]
        shallow_avg = sum(probs[:n//4]) / max(1, n//4)
        deep_avg = sum(probs[-(n//4):]) / max(1, n//4)
        print(f"  浅层 avg={shallow_avg:.4f}  深层 avg={deep_avg:.4f}")
        if deep_avg < shallow_avg:
            print("  ⚠ 深层概率低于浅层，可能不符合预期（不一定是错误）")
        # 首次进入 top-1 的层
        first_top1 = next((l["layer"] for l in layers if l["topk_tokens"][0] == body["target_token"]), None)
        print(f"  首次进入 top-1 的层: {first_top1}")
        # 最终层自检
        assert abs(layers[-1]["target_prob"] - body["final_target_prob"]) < 0.01, "final_target_prob mismatch"
        print("  ✓ final_target_prob 与最终层一致")

    print("\n[2] 互斥目标 — 预期 400")
    r = post(base_url, {"context": "hello", "model": "base", "source_page": "attribution",
                         "target_prediction": "world", "target_token_id": 5})
    if r.status_code != 400:
        print(f"  ✗ 预期 400，得到 {r.status_code}")
        failures += 1
    else:
        print(f"  ✓ 400: {r.json()['message']}")

    print("\n[3] 非法 model — 预期 400")
    r = post(base_url, {"context": "hello", "model": "gpt4", "source_page": "attribution"})
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
