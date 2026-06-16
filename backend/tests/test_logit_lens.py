"""
Logit Lens 单测。

运行: python -m unittest backend.tests.test_logit_lens
"""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch


def _make_mock_layer(layer_idx: int):
    return {
        "layer": layer_idx,
        "is_embedding": layer_idx == 0,
        "topk_tokens": ["北", "京"],
        "topk_probs": [0.8, 0.1],
        "target_prob": 0.8 if layer_idx == 28 else 0.01 * layer_idx,
    }


def _make_mock_result(n_layers=28):
    return {
        "model": "test-model",
        "target_token": "北",
        "n_layers": n_layers,
        "final_target_prob": 0.8,
        "layers": [_make_mock_layer(i) for i in range(n_layers + 1)],
        "debug_info": {"topk_tokens": ["北"], "topk_probs": [0.8]},
        "is_eos": False,
    }


class TestLogitLensHandlerValidation(unittest.TestCase):
    def _call(self, payload):
        from backend.api.logit_lens import logit_lens
        return logit_lens(payload)

    def _patches(self, result=None):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("backend.api.logit_lens.analyze_logit_lens", return_value=result or _make_mock_result()))
        mock_lock = MagicMock(); mock_lock.acquire.return_value = True
        stack.enter_context(patch("backend.api.logit_lens.inference_lock", mock_lock))
        stack.enter_context(patch("backend.api.logit_lens.log_prediction_attribute_request", return_value=1))
        return stack

    def test_missing_context_400(self):
        resp, code = self._call({"model": "base", "source_page": "attribution"})
        self.assertEqual(code, 400)

    def test_empty_context_400(self):
        resp, code = self._call({"context": "", "model": "base", "source_page": "attribution"})
        self.assertEqual(code, 400)

    def test_invalid_model_400(self):
        resp, code = self._call({"context": "hello", "model": "gpt4", "source_page": "attribution"})
        self.assertEqual(code, 400)

    def test_missing_source_page_400(self):
        resp, code = self._call({"context": "hello", "model": "base"})
        self.assertEqual(code, 400)

    def test_invalid_source_page_400(self):
        resp, code = self._call({"context": "hello", "model": "base", "source_page": "bad"})
        self.assertEqual(code, 400)

    def test_mutually_exclusive_400(self):
        resp, code = self._call({
            "context": "hello", "model": "base", "source_page": "attribution",
            "target_prediction": "world", "target_token_id": 5,
        })
        self.assertEqual(code, 400)
        self.assertIn("mutually exclusive", resp["message"])

    def test_context_too_long_400(self):
        mock_lock = MagicMock(); mock_lock.acquire.return_value = True
        with patch("backend.api.logit_lens.inference_lock", mock_lock), \
             patch("backend.api.logit_lens.log_prediction_attribute_request", return_value=1), \
             patch("backend.api.logit_lens.analyze_logit_lens", side_effect=ValueError("Context exceeds attribution length limit (500 tokens); current length is 600 tokens.")):
            resp, code = self._call({"context": "x" * 2000, "model": "base", "source_page": "attribution"})
        self.assertEqual(code, 400)
        self.assertIn("500", resp["message"])

    def test_top1_200(self):
        with self._patches():
            resp, code = self._call({"context": "中国的首都是", "model": "base", "source_page": "attribution"})
        self.assertEqual(code, 200)
        self.assertTrue(resp["success"])
        self.assertIn("layers", resp)

    def test_explicit_target_token_id_200(self):
        with self._patches():
            resp, code = self._call({"context": "hello", "model": "base", "source_page": "attribution", "target_token_id": 100})
        self.assertEqual(code, 200)

    def test_explicit_target_prediction_200(self):
        with self._patches():
            resp, code = self._call({"context": "hello", "model": "base", "source_page": "attribution", "target_prediction": "world"})
        self.assertEqual(code, 200)

    def test_lock_timeout_503(self):
        mock_lock = MagicMock(); mock_lock.acquire.return_value = False
        with patch("backend.api.logit_lens.inference_lock", mock_lock), \
             patch("backend.api.logit_lens.log_prediction_attribute_request", return_value=1):
            resp, code = self._call({"context": "hello", "model": "base", "source_page": "attribution"})
        self.assertEqual(code, 503)


def _model_available():
    try:
        from backend.core.logit_lens import analyze_logit_lens
        analyze_logit_lens("test", model="base")
        return True
    except Exception:
        return False


@unittest.skipUnless(_model_available(), "base model not available")
class TestLogitLensSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.core.logit_lens import analyze_logit_lens
        cls.result = analyze_logit_lens("中国的首都是", model="base")

    def test_keys_present(self):
        for k in ("model", "target_token", "n_layers", "final_target_prob", "layers", "debug_info", "is_eos"):
            self.assertIn(k, self.result)

    def test_layers_length(self):
        # 长度应为 n_layers + 1（含 embedding 层）
        self.assertEqual(len(self.result["layers"]), self.result["n_layers"] + 1)

    def test_embedding_layer_flag(self):
        self.assertTrue(self.result["layers"][0]["is_embedding"])
        self.assertFalse(self.result["layers"][-1]["is_embedding"])

    def test_final_layer_target_prob_matches(self):
        final_layer = self.result["layers"][-1]
        self.assertAlmostEqual(final_layer["target_prob"], self.result["final_target_prob"], places=4)

    def test_topk_length(self):
        from backend.core.next_token_topk import DEFAULT_NEXT_TOKEN_TOPK
        for layer in self.result["layers"]:
            self.assertEqual(len(layer["topk_tokens"]), DEFAULT_NEXT_TOKEN_TOPK)
            self.assertEqual(len(layer["topk_probs"]), DEFAULT_NEXT_TOKEN_TOPK)

    def test_mutually_exclusive_raises(self):
        from backend.core.logit_lens import analyze_logit_lens
        with self.assertRaises(ValueError):
            analyze_logit_lens("hello", target_prediction="world", model="base", target_token_id=5)

    def test_too_long_raises(self):
        from backend.core.logit_lens import analyze_logit_lens
        from backend.core.prediction_attributor import ATTRIBUTION_MAX_TOKEN_LENGTH
        with self.assertRaises(ValueError) as ctx:
            analyze_logit_lens("hello world " * (ATTRIBUTION_MAX_TOKEN_LENGTH + 10), model="base")
        self.assertIn(str(ATTRIBUTION_MAX_TOKEN_LENGTH), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
