#!/usr/bin/env python3
"""
第二波测试：聚焦可能导致 score 全为 0 的边界场景。
"""

import json
import urllib.request
import urllib.error
import math

API_BASE = "http://127.0.0.1:5001"

# ---------------------------------------------------------------------------
# 边界场景测试
# ---------------------------------------------------------------------------
edge_cases = [
    # --- 模型不"认识"的 target（低概率 target）---
    ("法国的首都是", "华盛顿"),       # 错误的预测
    ("苹果是一种", "水果"),            # 正确
    ("苹果是一种", "橘子"),            # 错误的预测
    ("1+1=", "3"),                     # 错误答案

    # --- target 是概率极低的 token（模型根本不会预测这个）---
    ("上海是中国的", "首都"),               # 上海不是首都
    ("中国的首都是", "上海"),               # 错误城市
    ("太阳从西边", "升起"),                 # 错误方向
    ("企鹅是一种", "哺乳动物"),             # 企鹅不是哺乳动物

    # --- 非常短的输入（可能没有"有效"token）---
    ("A", " B"),
    ("I", " am"),
    ("是", "的"),
    ("好", "人"),

    # --- 非常长的输入 ---
    ("A" * 100, " B"),
    ("测试" * 50, "通"),

    # --- 中英混合 ---
    ("ML is short for 机器", "学习"),
    ("AI代表人工", "智能"),

    # --- 多字 target（取首 token，可能语义变化）---
    ("中国的首都是", "北京欢迎你"),
    ("1+1=2是", "数学常识"),

    # --- 特殊标点 ---
    ("Hello!!!", " World"),
    ("你好？？？", "？"),

    # --- target_prob 本身就近乎 0 的场景（越少见的 token 越危险）---
    ("床前明月", "的"),                # 常见但可能不是 top-1
    ("The quick brown fox jumps over the lazy", " cat"),  # 经典句子
    ("天地玄黄 宇宙洪荒 日月盈昃 辰宿列张 寒来暑往 秋收冬藏", "闰"),  # 千字文

    # --- 纯英文连续高频词 ---
    ("I like to eat", " pizza"),
    ("She went to the", " store"),
    ("They are going to", " school"),

    # --- token 拼接可能导致的问题 ---
    ("Big", " Apple"),                 # "Big Apple" = New York 但分开 tokenize 可能不同
    ("New", " York"),
    ("San", " Francisco"),

    # --- 和一些肯定能正常工作的对比 ---
    ("台湾是中国不可分割的一部分", "中国"),    # 政治敏感词
    ("天空是", "蓝"),
    ("太阳是", "恒星"),
]


def test_via_api(context, target_prediction):
    payload = {
        "context": context,
        "model": "base",
        "source_page": "attribution",
    }
    if target_prediction is not None:
        payload["target_prediction"] = target_prediction

    req = urllib.request.Request(
        f"{API_BASE}/api/ablation-attribute",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return {"error": f"HTTP {e.code}: {body}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def main():
    print(f"🚀 消融归因 — 边界场景测试 ({len(edge_cases)} 个用例)")
    print("=" * 80)

    zero_score_cases = []
    non_zero_cases = []
    borderline_cases = []
    error_cases = []

    for i, (context, target) in enumerate(edge_cases):
        short_ctx = (context[:50] + "...") if len(context) > 50 else context
        print(f"\n[{i+1}/{len(edge_cases)}] context={repr(short_ctx)}, target={target}")

        r = test_via_api(context, target)

        if "error" in r:
            print(f"    ⚠️  error: {r['error'][:80]}")
            error_cases.append((i, context, target, r["error"]))
            continue

        if "token_attribution" not in r:
            print(f"    ⚠️  no token_attribution: {json.dumps(r)[:100]}")
            error_cases.append((i, context, target, "No token_attribution"))
            continue

        scores = [e["score"] for e in r["token_attribution"]]
        target_prob = r.get("target_prob", "?")
        target_token = r.get("target_token", "?")
        debug_info = r.get("debug_info", {})
        topk_tokens = debug_info.get("topk_tokens", [])
        topk_probs = debug_info.get("topk_probs", [])

        non_zero = [s for s in scores if abs(s) > 1e-10]
        total = len(scores)

        # 判断是否在 top-10 中
        in_top10 = target_token in topk_tokens if topk_tokens else "unknown"

        info = {
            "idx": i, "context": context, "target": target,
            "target_token": target_token, "target_prob": target_prob,
            "n_tokens": total, "n_non_zero": len(non_zero),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "all_scores": scores,
            "in_top10": in_top10,
            "topk_tokens": topk_tokens,
            "topk_probs": topk_probs,
        }

        if len(non_zero) == 0:
            print(f"    ❌ 全为 0! target_prob={target_prob}, target_token={target_token!r}")
            print(f"       in_top10={in_top10}, top10={topk_tokens[:5]}")
            zero_score_cases.append(info)
        elif max(scores) < 1e-5 and target_prob is not None and target_prob < 1e-4:
            # 虽然非零但非常小，且 target_prob 也很小（边界情况）
            print(f"    ⚠️  边界: scores 极小 (max={max(scores):.3e}), target_prob={target_prob}, target_token={target_token!r}")
            print(f"       in_top10={in_top10}, top10={topk_tokens[:5]}")
            borderline_cases.append(info)
        else:
            print(f"    ✅ {len(non_zero)}/{total} 非零, max={max(scores):.6f}, target_prob={target_prob}")
            non_zero_cases.append(info)

    # --- 报告 ---
    print("\n" + "=" * 80)
    print("📊 边界场景测试报告")
    print("=" * 80)
    print(f"总用例: {len(edge_cases)}")
    print(f"  ✅ 正常非零: {len(non_zero_cases)}")
    print(f"  ⚠️  边界（值极小）: {len(borderline_cases)}")
    print(f"  ❌ 全为 0: {len(zero_score_cases)}")
    print(f"  ❌ 错误: {len(error_cases)}")

    if zero_score_cases:
        print("\n" + "-" * 60)
        print("🔴 全为 0 的用例:")
        for info in zero_score_cases:
            print(f"  [{info['idx']}] ctx={repr(info['context'][:40])}, target={info['target']}")
            print(f"      target_token={info['target_token']!r}, target_prob={info['target_prob']}")
            print(f"      in_top10={info['in_top10']}, topk={info['topk_tokens'][:3]}")
            print(f"      scores={info['all_scores']}")

    if borderline_cases:
        print("\n" + "-" * 60)
        print("🟡 边界用例（scores 极小）:")
        for info in borderline_cases:
            print(f"  [{info['idx']}] ctx={repr(info['context'][:40])}, target={info['target']}")
            print(f"      target_token={info['target_token']!r}, target_prob={info['target_prob']}")
            print(f"      in_top10={info['in_top10']}, topk={info['topk_tokens'][:5]}")
            print(f"      scores={info['all_scores'][:6]}")

    # --- 总结分析 ---
    print("\n" + "=" * 60)
    print("📈 相关性分析")
    print("=" * 60)
    all_bad = zero_score_cases + borderline_cases
    if all_bad:
        print(f"共 {len(all_bad)} 个有问题的用例:")
        not_in_top10 = [c for c in all_bad if c.get("in_top10") == False or c.get("in_top10") == "unknown"]
        prob_low = [c for c in all_bad if c.get("target_prob") is not None and c["target_prob"] < 0.01]
        print(f"  不在 top-10 中: {len(not_in_top10)}")
        print(f"  target_prob < 0.01: {len(prob_low)}")

        for c in all_bad:
            print(f"    [{c['idx']}] target='{c['target']}' → token='{c['target_token']!r}' prob={c['target_prob']:.6e} in_top10={c['in_top10']}")


if __name__ == "__main__":
    main()