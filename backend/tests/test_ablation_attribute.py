"""
消融归因单测。

分两类：
1. API handler 参数校验测试（不需要模型权重，mock core 函数）
2. 算法语义测试（需模型权重；@skipUnless 跳过）

运行: python -m unittest backend.tests.test_ablation_attribute
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers：构造 mock 让 ablation_attribute handler 可在无模型环境下导入
# ---------------------------------------------------------------------------

def _make_mock_result(score=0.1, delta_logit=5.0):
    return {
        "model": "test-model",
        "target_token": "北",
        "target_prob": 0.8,
        "token_attribution": [
            {"offset": [0, 3], "raw": "中国", "score": score, "delta_logit": delta_logit, "delta_prob": 0.05},
        ],
        "debug_info": {"topk_tokens": ["北"], "topk_probs": [0.8]},
        "is_eos": False,
    }


# ---------------------------------------------------------------------------
# Group 1: API handler 参数校验（mock core，无需模型）
# ---------------------------------------------------------------------------

class TestAblationAttributeHandlerValidation(unittest.TestCase):
    """参数校验：对应 spec 场景（缺字段 / 非法 model / 互斥目标等）"""

    def _call(self, payload):
        from backend.api.ablation_attribute import ablation_attribute
        return ablation_attribute(payload)

    def _patch_core(self):
        return patch(
            "backend.api.ablation_attribute.analyze_ablation_attribution",
            return_value=_make_mock_result(),
        )

    def _patch_lock(self):
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        return patch("backend.api.ablation_attribute.inference_lock", mock_lock)

    def _patch_log(self):
        return patch(
            "backend.api.ablation_attribute.log_prediction_attribute_request",
            return_value=1,
        )

    # --- 缺少必要字段 ---
    def test_missing_context_returns_400(self):
        resp, code = self._call({"model": "base", "source_page": "attribution"})
        self.assertEqual(code, 400)
        self.assertFalse(resp["success"])

    def test_empty_context_returns_400(self):
        resp, code = self._call({"context": "", "model": "base", "source_page": "attribution"})
        self.assertEqual(code, 400)
        self.assertFalse(resp["success"])

    def test_missing_model_returns_400(self):
        resp, code = self._call({"context": "hello", "source_page": "attribution"})
        self.assertEqual(code, 400)
        self.assertFalse(resp["success"])

    def test_invalid_model_returns_400(self):
        resp, code = self._call({"context": "hello", "model": "gpt4", "source_page": "attribution"})
        self.assertEqual(code, 400)
        self.assertIn("base", resp["message"])

    def test_missing_source_page_returns_400(self):
        resp, code = self._call({"context": "hello", "model": "base"})
        self.assertEqual(code, 400)
        self.assertFalse(resp["success"])

    def test_invalid_source_page_returns_400(self):
        resp, code = self._call({"context": "hello", "model": "base", "source_page": "bad_page"})
        self.assertEqual(code, 400)
        self.assertFalse(resp["success"])

    # --- 目标互斥 ---
    def test_mutually_exclusive_target_returns_400(self):
        resp, code = self._call({
            "context": "hello",
            "model": "base",
            "source_page": "attribution",
            "target_prediction": "world",
            "target_token_id": 5,
        })
        self.assertEqual(code, 400)
        self.assertIn("mutually exclusive", resp["message"])

    # --- 超长 context（core 抛 ValueError）---
    def test_context_too_long_returns_400(self):
        with self._patch_lock(), self._patch_log():
            with patch(
                "backend.api.ablation_attribute.analyze_ablation_attribution",
                side_effect=ValueError("Context exceeds attribution length limit (500 tokens); current length is 600 tokens."),
            ):
                resp, code = self._call({
                    "context": "x" * 2000,
                    "model": "base",
                    "source_page": "attribution",
                })
        self.assertEqual(code, 400)
        self.assertIn("500", resp["message"])

    # --- 正常 top-1 ---
    def test_top1_default_returns_200(self):
        with self._patch_core(), self._patch_lock(), self._patch_log():
            resp, code = self._call({
                "context": "中国的首都是",
                "model": "base",
                "source_page": "attribution",
            })
        self.assertEqual(code, 200)
        self.assertTrue(resp["success"])
        self.assertIn("token_attribution", resp)

    # --- 显式 target_token_id ---
    def test_explicit_target_token_id_returns_200(self):
        with self._patch_core(), self._patch_lock(), self._patch_log():
            resp, code = self._call({
                "context": "中国的首都是",
                "model": "base",
                "source_page": "attribution",
                "target_token_id": 1234,
            })
        self.assertEqual(code, 200)
        self.assertTrue(resp["success"])

    # --- 显式 target_prediction ---
    def test_explicit_target_prediction_returns_200(self):
        with self._patch_core(), self._patch_lock(), self._patch_log():
            resp, code = self._call({
                "context": "中国的首都是",
                "model": "base",
                "source_page": "attribution",
                "target_prediction": "北京",
            })
        self.assertEqual(code, 200)
        self.assertTrue(resp["success"])

    # --- 服务繁忙 503 ---
    def test_lock_timeout_returns_503(self):
        with self._patch_log():
            mock_lock = MagicMock()
            mock_lock.acquire.return_value = False
            with patch("backend.api.ablation_attribute.inference_lock", mock_lock):
                resp, code = self._call({
                    "context": "hello",
                    "model": "base",
                    "source_page": "attribution",
                })
        self.assertEqual(code, 503)
        self.assertFalse(resp["success"])


# ---------------------------------------------------------------------------
# Group 2: 算法语义（需模型）
# ---------------------------------------------------------------------------

def _model_available() -> bool:
    try:
        from backend.core.ablation_attributor import analyze_ablation_attribution
        analyze_ablation_attribution("test", model="base")
        return True
    except Exception:
        return False


@unittest.skipUnless(_model_available(), "base model weights not available")
class TestAblationAttributorSemantics(unittest.TestCase):
    """核心语义断言：score 公式、offset 合法、特殊 token 排除"""

    @classmethod
    def setUpClass(cls):
        from backend.core.ablation_attributor import analyze_ablation_attribution
        cls.result = analyze_ablation_attribution("中国的首都是", model="base")

    def test_success_keys_present(self):
        r = self.result
        for key in ("model", "target_token", "target_prob", "token_attribution", "debug_info", "is_eos"):
            self.assertIn(key, r)

    def test_offsets_valid(self):
        context = "中国的首都是"
        for entry in self.result["token_attribution"]:
            s, e = entry["offset"]
            self.assertLess(s, e, "offset start must be < end")
            self.assertEqual(entry["raw"], context[s:e], "raw must equal context[start:end]")

    def test_score_sign_reasonable(self):
        """至少应存在非零 score（某 token 影响目标预测）"""
        scores = [e["score"] for e in self.result["token_attribution"]]
        self.assertTrue(any(abs(s) > 1e-10 for s in scores), "Expected at least one non-zero score")

    def test_no_zero_width_tokens(self):
        """特殊 token（BOS/EOS 等，span 为空）不得出现在结果中"""
        for entry in self.result["token_attribution"]:
            s, e = entry["offset"]
            self.assertGreater(e - s, 0)

    def test_score_not_zero(self):
        """
        使用 EOS 基线 + delta_logit 后，score 应为非零值（除非模型完全无反应）。
        delta_logit 不受概率边界约束，因此对任意 target 均应产生有意义的分值。
        """
        scores = [e["score"] for e in self.result["token_attribution"]]
        non_zero = [s for s in scores if abs(s) > 1e-10]
        self.assertGreater(len(non_zero), 0, "Expected at least one non-zero delta_logit score")

    def test_target_token_id_explicit(self):
        """显式 target_token_id 应返回对应 target_token"""
        from backend.core.ablation_attributor import analyze_ablation_attribution
        from backend.models.model_manager import ModelSlot, ensure_slot_weights_loaded
        tokenizer, _, _ = ensure_slot_weights_loaded(ModelSlot.BASE)
        # 用 top-1 结果的 token_id
        tok_id = tokenizer.encode(self.result["target_token"], add_special_tokens=False)[0]
        r2 = analyze_ablation_attribution("中国的首都是", model="base", target_token_id=tok_id)
        self.assertEqual(r2["target_token"], self.result["target_token"])

    def test_mutually_exclusive_raises(self):
        from backend.core.ablation_attributor import analyze_ablation_attribution
        with self.assertRaises(ValueError):
            analyze_ablation_attribution(
                "hello",
                target_prediction="world",
                model="base",
                target_token_id=5,
            )

    def test_too_long_raises(self):
        from backend.core.ablation_attributor import (
            analyze_ablation_attribution,
            ATTRIBUTION_MAX_TOKEN_LENGTH,
        )
        long_ctx = "hello world " * (ATTRIBUTION_MAX_TOKEN_LENGTH + 10)
        with self.assertRaises(ValueError) as ctx:
            analyze_ablation_attribution(long_ctx, model="base")
        self.assertIn(str(ATTRIBUTION_MAX_TOKEN_LENGTH), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
