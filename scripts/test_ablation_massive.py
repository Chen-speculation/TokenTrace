#!/usr/bin/env python3
"""
大规模测试消融归因：测试几十个不同场景的 case，观察是否出现"全部为 0"的情况。
"""

import json
import sys
import os
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:5001"

# ---------------------------------------------------------------------------
# Test cases: diverse scenarios
# ---------------------------------------------------------------------------
test_cases = [
    # --- 基础事实 ---
    ("中国的首都是", "北京"),
    ("法国的首都是", "巴黎"),
    ("地球绕着什么转", "太阳"),
    ("1+1=", "2"),
    ("水的化学式是", "H2O"),
    ("太阳从什么方向升起", "东"),

    # --- 单 token 目标 ---
    ("The capital of France is", " Paris"),
    ("The color of the sky is", " blue"),
    ("Apple's founder was", " Steve"),
    ("Python is a programming", " language"),

    # --- 否定 / 对比 ---
    ("猫不属于", "鱼类"),
    ("企鹅虽然不会飞，但属于", "鸟类"),
    ("鲸鱼生活在水中，但它不是鱼，而是", "哺乳动物"),

    # --- 逻辑推理 ---
    ("如果今天是周一，那么明天是", "周二"),
    ("一年有十二个月，一个月大约有30", "天"),
    ("人喝水是为了", "解渴"),
    ("汽车没有油就", "无法"),

    # --- 中文长句子 ---
    ("人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括", "机器人"),
    ("机器学习是人工智能的子领域，它专注于让计算机通过经验自动改善性能。监督学习是最常见的机器学习范式之一，它使用", "标注"),
    ("深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。卷积神经网络特别适合处理", "图像"),

    # --- 英文长句子 ---
    ("Artificial intelligence is a branch of computer science that aims to create machines that can perform tasks that typically require human intelligence. One of the most popular applications is", " machine"),
    ("Natural language processing is a subfield of AI that focuses on the interaction between computers and humans through natural language. A key challenge is", " understanding"),
    ("The transformer architecture introduced in 'Attention is All You Need' has become the foundation of most modern language models. It relies primarily on", " attention"),

    # --- 重复结构 / 模板 ---
    ("I think therefore I", " am"),
    ("To be or not to", " be"),
    ("Hello world is the first program every", " programmer"),

    # --- 数字 / 数学 ---
    ("The square root of 9 is", " 3"),
    ("2 multiplied by 6 equals", " 12"),
    ("The year after 2023 is", " 2024"),

    # --- 知识问答 ---
    ("爱因斯坦提出了", "相对论"),
    ("牛顿发现了", "万有引力"),
    ("莎士比亚是著名的", "戏剧"),
    ("莫扎特是一位", "作曲"),

    # --- 短文本 / 单字目标 ---
    ("天对地，雨对风，大陆对长", "空"),
    ("床前明月", "光"),

    # --- ambiguous / 开放 ---
    ("明天会不会下雨取决于", "天气"),
    ("要成功就必须付出", "努力"),

    # --- top-1（不指定 target）的自动模式 ---
    ("我今天中午吃了", None),  # top-1
    ("The best thing about summer is", None),  # top-1
    ("在我看来，最重要的品质是", None),  # top-1
]


def test_via_api(context, target_prediction):
    """通过 HTTP API 测试"""
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
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return {"error": f"HTTP {e.code}: {body}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def analyze_results(results):
    """分析测试结果，找出全为 0 的情况"""
    zero_score_cases = []
    non_zero_cases = []  # has at least one non-zero score
    error_cases = []

    for i, (case, r) in enumerate(results):
        context, target = case

        if "error" in r:
            error_cases.append((i, context, target, r["error"]))
            continue

        if "token_attribution" not in r:
            error_cases.append((i, context, target, "No token_attribution in response: " + json.dumps(r)[:200]))
            continue

        scores = [e["score"] for e in r["token_attribution"]]
        non_zero_scores = [s for s in scores if abs(s) > 1e-10]

        info = {
            "case_idx": i,
            "context": repr((context[:50] + "...") if len(context) > 50 else context),
            "target": target,
            "target_token": r.get("target_token", "?"),
            "target_prob": r.get("target_prob"),
            "n_tokens": len(r["token_attribution"]),
            "n_non_zero": len(non_zero_scores),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "all_scores": scores,
            "all_deltas": [e.get("delta_logit", "N/A") for e in r["token_attribution"]],
        }

        if len(non_zero_scores) == 0:
            zero_score_cases.append(info)
        else:
            non_zero_cases.append(info)

    return {
        "zero_score_cases": zero_score_cases,
        "non_zero_cases": non_zero_cases,
        "error_cases": error_cases,
        "total": len(results),
        "zero_count": len(zero_score_cases),
        "non_zero_count": len(non_zero_cases),
        "error_count": len(error_cases),
    }


def print_analysis(stats):
    print("\n" + "=" * 80)
    print("📊 消融归因大规模测试报告")
    print("=" * 80)
    print(f"总测试用例: {stats['total']}")
    print(f"  ✅ 有非零分: {stats['non_zero_count']}")
    print(f"  ❌ 全为 0: {stats['zero_count']}")
    print(f"  ⚠️  错误: {stats['error_count']}")
    print()

    if stats['error_cases']:
        print("--- 错误用例 ---")
        for idx, ctx, tgt, err in stats['error_cases']:
            print(f"  [{idx}] context={repr(str(ctx)[:40])}, target={tgt}")
            print(f"       error: {err}")
        print()

    if stats['zero_score_cases']:
        print("--- 🚨 全为 0 的用例 ---")
        for info in stats['zero_score_cases']:
            print(f"  [{info['case_idx']}] context={info['context']}")
            print(f"       target={info['target']}, target_token={info['target_token']!r}")
            print(f"       target_prob={info['target_prob']}, n_tokens={info['n_tokens']}")
            print(f"       scores: {info['all_scores']}")
            # Check delta_logits
            deltas = [d for d in info['all_deltas'] if d != "N/A"]
            if deltas:
                non_zero_deltas = [d for d in deltas if abs(d) > 1e-10]
                print(f"       delta_logits: {info['all_deltas'][:10]}...")
                if non_zero_deltas:
                    print(f"       ⚠️  scores 全 0，但 delta_logits 有非零值! count={len(non_zero_deltas)}")
                else:
                    print(f"       delta_logits 也全 0")
            print()

    if stats['non_zero_cases']:
        print("--- ✅ 有非零分的用例（前 8 个）---")
        for info in stats['non_zero_cases'][:8]:
            non_zero_ratio = f"{info['n_non_zero']}/{info['n_tokens']}"
            print(f"  [{info['case_idx']}] context={info['context']}")
            print(f"       target={info['target']}, non_zero_ratio={non_zero_ratio}")
            print(f"       max_score={info['max_score']:.6f}, min_score={info['min_score']:.4f}, target_prob={info['target_prob']}")
            print(f"       scores={info['all_scores']}")
            print()


def main():
    print(f"🚀 启动消融归因大规模测试 ({len(test_cases)} 个用例)")
    print(f"目标服务器: {API_BASE}")
    print()

    # 先检查服务器是否可达
    try:
        with urllib.request.urlopen(f"{API_BASE}/", timeout=5) as _:
            print("✅ 服务器可达")
    except Exception as e:
        print(f"⚠️  服务器不可达 ({e})，继续尝试...")
    print()

    results = []
    for i, (context, target) in enumerate(test_cases):
        short_ctx = (context[:40] + "...") if len(context) > 40 else context
        print(f"[{i+1}/{len(test_cases)}] testing: context={repr(short_ctx)}, target={target}")
        try:
            r = test_via_api(context, target)
            results.append(((context, target), r))

            # 检查是否有 score
            if "token_attribution" in r:
                scores = [e["score"] for e in r["token_attribution"]]
                non_zero = sum(1 for s in scores if abs(s) > 1e-10)
                total = len(scores)
                if non_zero == 0:
                    print(f"    ❌ 全为 0! target_prob={r.get('target_prob')}, target_token={r.get('target_token')!r}")
                else:
                    max_s = max(scores)
                    print(f"    ✅ {non_zero}/{total} 非零, max_score={max_s:.6f}")
            elif "error" in r:
                print(f"    ⚠️  error: {r['error'][:80]}")
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            results.append(((context, target), {"error": str(e)}))

    stats = analyze_results(results)
    print_analysis(stats)

    # 深入分析全零根因
    if stats['zero_count'] > 0:
        first_zero = stats['zero_score_cases'][0]
        ctx = test_cases[first_zero['case_idx']][0]
        tgt = test_cases[first_zero['case_idx']][1]
        print("=" * 80)
        print("🔍 对第一个全零 case 做深度诊断")
        print("=" * 80)
        diagnose_ablation(ctx, tgt)


def diagnose_ablation(context, target_prediction=None):
    """对单个 case 做详细诊断"""
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
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  API 调用失败: {e}")
        return

    if "token_attribution" in data:
        print(f"  token_attribution 条目数: {len(data['token_attribution'])}")
        print(f"  target_prob: {data['target_prob']}")
        print(f"  target_token: {data['target_token']!r}")
        print(f"  debug_info: {data.get('debug_info')}")

        scores = [e["score"] for e in data["token_attribution"]]
        deltas = [e.get("delta_logit", "N/A") for e in data["token_attribution"]]
        raws = [e["raw"] for e in data["token_attribution"]]

        print(f"  scores: {scores}")
        print(f"  delta_logits: {deltas}")
        print(f"  raws: {raws}")

        # 如果 scores 全 0 但 delta_logit 非全 0，说明 prob 变化在 round_to_sig_figs 之下
        if all(abs(s) < 1e-10 for s in scores):
            numeric_deltas = [d for d in deltas if isinstance(d, (int, float))]
            non_zero_deltas = [d for d in numeric_deltas if abs(d) > 1e-10]
            if non_zero_deltas:
                print(f"\n  🔑 关键发现: scores 全部 round 到 0，但 delta_logits 有非零值!")
                print(f"     说明模型对遮挡确实有反应，但概率变化太小 (< 5e-8)")
                print(f"     非零 deltas 示例: {non_zero_deltas[:10]}")
            else:
                print(f"\n  🔑 delta_logits 也是全 0 (或 {len(numeric_deltas)}/{len(deltas)} 为数值型)，说明遮挡完全不影响预测")

        print(f"\n  📝 debug_info topk_tokens: {data.get('debug_info', {}).get('topk_tokens', [])}")
        print(f"  📝 debug_info topk_probs: {data.get('debug_info', {}).get('topk_probs', [])}")


if __name__ == "__main__":
    main()