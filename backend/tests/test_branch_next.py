"""
分叉树 branch-next 单测。

运行: python -m unittest backend.tests.test_branch_next
"""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch

from backend.core.branch_next import BRANCH_NEXT_TOP_K_MAX


def _make_mock_result(n=10):
    return {
        "model": "test-model",
        "prefix_tokens": 6,
        "candidates": [{"token": f"tok{i}", "token_id": i, "prob": round(0.9 / (i + 1), 3)} for i in range(n)],
        "is_context_full": False,
    }


class TestBranchNextHandlerValidation(unittest.TestCase):
    def _call(self, payload):
        from backend.api.branch_next import branch_next
        return branch_next(payload)

    def _patches(self, result=None):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("backend.api.branch_next.expand_branch_next", return_value=result or _make_mock_result()))
        mock_lock = MagicMock(); mock_lock.acquire.return_value = True
        stack.enter_context(patch("backend.api.branch_next.inference_lock", mock_lock))
        stack.enter_context(patch("backend.api.branch_next.log_request"))
        return stack

    def test_missing_prefix_400(self):
        resp, code = self._call({"model": "base", "source_page": "causal_flow"})
        self.assertEqual(code, 400)

    def test_empty_prefix_400(self):
        resp, code = self._call({"prefix": "", "model": "base", "source_page": "causal_flow"})
        self.assertEqual(code, 400)

    def test_invalid_model_400(self):
        resp, code = self._call({"prefix": "hello", "model": "gpt4", "source_page": "causal_flow"})
        self.assertEqual(code, 400)

    def test_invalid_source_page_400(self):
        resp, code = self._call({"prefix": "hello", "model": "base", "source_page": "nowhere"})
        self.assertEqual(code, 400)

    def test_top_k_less_than_1_400(self):
        resp, code = self._call({"prefix": "hello", "model": "base", "source_page": "causal_flow", "top_k": 0})
        self.assertEqual(code, 400)

    def test_top_k_over_limit_clamped(self):
        """top_k > 上限时 clamp 而非报错（设计 D3）"""
        captured = {}
        def mock_expand(prefix, *, model, top_k=10):
            captured["top_k"] = top_k
            return _make_mock_result(top_k)
        mock_lock = MagicMock(); mock_lock.acquire.return_value = True
        with patch("backend.api.branch_next.expand_branch_next", side_effect=mock_expand), \
             patch("backend.api.branch_next.inference_lock", mock_lock), \
             patch("backend.api.branch_next.log_request"):
            resp, code = self._call({"prefix": "hello", "model": "base", "source_page": "causal_flow", "top_k": 999})
        self.assertEqual(code, 200)
        self.assertEqual(captured["top_k"], BRANCH_NEXT_TOP_K_MAX)

    def test_normal_200(self):
        with self._patches():
            resp, code = self._call({"prefix": "中国的首都", "model": "base", "source_page": "causal_flow"})
        self.assertEqual(code, 200)
        self.assertTrue(resp["success"])
        self.assertIn("candidates", resp)

    def test_custom_top_k_200(self):
        with self._patches(_make_mock_result(5)):
            resp, code = self._call({"prefix": "hello", "model": "base", "source_page": "causal_flow", "top_k": 5})
        self.assertEqual(code, 200)

    def test_lock_timeout_503(self):
        mock_lock = MagicMock(); mock_lock.acquire.return_value = False
        with patch("backend.api.branch_next.inference_lock", mock_lock), \
             patch("backend.api.branch_next.log_request"):
            resp, code = self._call({"prefix": "hello", "model": "base", "source_page": "causal_flow"})
        self.assertEqual(code, 503)

    def test_prompt_too_long_400(self):
        from backend.core.completion_generator import PromptTooLongError
        mock_lock = MagicMock(); mock_lock.acquire.return_value = True
        with patch("backend.api.branch_next.expand_branch_next", side_effect=PromptTooLongError("too long")), \
             patch("backend.api.branch_next.inference_lock", mock_lock), \
             patch("backend.api.branch_next.log_request"):
            resp, code = self._call({"prefix": "x" * 5000, "model": "base", "source_page": "causal_flow"})
        self.assertEqual(code, 400)


def _model_available():
    try:
        from backend.core.branch_next import expand_branch_next
        expand_branch_next("test", model="base")
        return True
    except Exception:
        return False


@unittest.skipUnless(_model_available(), "base model not available")
class TestBranchNextSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.core.branch_next import expand_branch_next
        cls.result = expand_branch_next("中国的首都", model="base")

    def test_keys_present(self):
        for k in ("model", "prefix_tokens", "candidates", "is_context_full"):
            self.assertIn(k, self.result)

    def test_candidates_count(self):
        from backend.core.next_token_topk import DEFAULT_NEXT_TOKEN_TOPK
        self.assertEqual(len(self.result["candidates"]), DEFAULT_NEXT_TOKEN_TOPK)

    def test_candidates_sorted_by_prob(self):
        probs = [c["prob"] for c in self.result["candidates"]]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top1_equals_argmax(self):
        """top-1 token_id 等于末位 logits argmax（D1）"""
        import torch
        from backend.models.model_manager import ModelSlot, ensure_slot_weights_loaded
        tokenizer, hf_model, device = ensure_slot_weights_loaded(ModelSlot.BASE)
        enc = tokenizer("中国的首都", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = hf_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        expected_top1 = int(out.logits[0, -1, :].argmax().item())
        self.assertEqual(self.result["candidates"][0]["token_id"], expected_top1)

    def test_no_generate_called(self):
        """不调用 model.generate"""
        from unittest.mock import patch
        from backend.core.branch_next import expand_branch_next
        from backend.models.model_manager import ModelSlot, ensure_slot_weights_loaded
        tokenizer, hf_model, device = ensure_slot_weights_loaded(ModelSlot.BASE)
        with patch.object(hf_model, "generate", side_effect=AssertionError("generate must not be called")):
            r = expand_branch_next("hello", model="base")
        self.assertIn("candidates", r)

    def test_top_k_less_than_1_raises(self):
        from backend.core.branch_next import expand_branch_next
        with self.assertRaises(ValueError):
            expand_branch_next("hello", model="base", top_k=0)


if __name__ == "__main__":
    unittest.main()
