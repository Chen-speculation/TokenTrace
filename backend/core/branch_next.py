"""
分叉树：单步 next-token top-k 原子操作。
输入 prefix，一次前向取末位 logits top-k，返回候选列表。
无生成循环、无反传。
"""
from typing import Dict, List

import torch

from backend.platform.format import round_to_sig_figs
from backend.models.device import DeviceManager
from backend.models.model_manager import (
    ModelSlot,
    ensure_slot_weights_loaded,
    get_base_model_display_name,
    get_instruct_model_display_name,
)
from .next_token_topk import DEFAULT_NEXT_TOKEN_TOPK
from .prediction_attributor import slot_for_prediction_attr_model
from backend.core.completion_generator import (
    PromptTooLongError,
    _model_context_token_limit,
)

BRANCH_NEXT_TOP_K_MAX = 50


def expand_branch_next(
    prefix: str,
    *,
    model: str,
    top_k: int = DEFAULT_NEXT_TOKEN_TOPK,
) -> Dict:
    """
    对 prefix 做单步前向，返回下一 token 的 top-k 候选。

    Returns:
        {
            "model": str,
            "prefix_tokens": int,
            "candidates": [{"token": str, "token_id": int, "prob": float}, ...],
            "is_context_full": bool,
        }
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    top_k = min(top_k, BRANCH_NEXT_TOP_K_MAX)

    slot = slot_for_prediction_attr_model(model)
    tokenizer, hf_model, device = ensure_slot_weights_loaded(slot)
    model_display = (
        get_base_model_display_name() if slot == ModelSlot.BASE else get_instruct_model_display_name()
    )

    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prefix_tokens = input_ids.shape[1]

    # 上下文上限校验（D2）
    ctx_limit = _model_context_token_limit(tokenizer, hf_model)
    if prefix_tokens >= ctx_limit:
        raise PromptTooLongError(
            f"Prefix length ({prefix_tokens} tokens) has reached the model context limit "
            f"({ctx_limit}); cannot expand further."
        )

    is_context_full = (prefix_tokens + 1) >= ctx_limit

    try:
        hf_model.eval()
        with torch.no_grad():
            outputs = hf_model(input_ids=input_ids, output_attentions=False, use_cache=False)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(logits, top_k)

        candidates: List[Dict] = []
        for tid, prob_val in zip(top_ids.tolist(), probs[top_ids].tolist()):
            p = prob_val if torch.isfinite(torch.tensor(prob_val)).item() else 0.0
            candidates.append({
                "token": tokenizer.decode([int(tid)], skip_special_tokens=False),
                "token_id": int(tid),
                "prob": round_to_sig_figs(p),
            })

        return {
            "model": model_display,
            "prefix_tokens": prefix_tokens,
            "candidates": candidates,
            "is_context_full": is_context_full,
        }
    finally:
        DeviceManager.clear_cache(device)
