"""
Logit Lens：对每层残差流末位 hidden state 过 final norm + lm_head，
输出该层"如果此刻截断直接输出"会预测什么，以及目标 token 在各层的概率轨迹。
"""
import math
from typing import Dict, Optional

import torch

from backend.platform.format import round_to_sig_figs
from backend.models.device import DeviceManager
from backend.models.model_manager import (
    ModelSlot,
    ensure_slot_weights_loaded,
    get_base_model_display_name,
    get_instruct_model_display_name,
)
from .next_token_topk import decode_topk_ids_to_strings_and_rounded_probs, DEFAULT_NEXT_TOKEN_TOPK
from .prediction_attributor import (
    ATTRIBUTION_MAX_TOKEN_LENGTH,
    slot_for_prediction_attr_model,
)


def analyze_logit_lens(
    context: str,
    target_prediction: Optional[str] = None,
    *,
    model: str,
    target_token_id: Optional[int] = None,
) -> Dict:
    """
    对 context 做 Logit Lens 分析。

    Returns:
        {
            "model": str,
            "target_token": str,
            "n_layers": int,               # L（不含 embedding 层）
            "final_target_prob": float,    # 最终层目标概率
            "layers": [
                {
                    "layer": int,          # 0=embedding 层, L=最终层
                    "is_embedding": bool,
                    "topk_tokens": [str],
                    "topk_probs": [float],
                    "target_prob": float,
                }
            ],
            "debug_info": {"topk_tokens": [...], "topk_probs": [...]},
            "is_eos": bool,
        }
    """
    slot = slot_for_prediction_attr_model(model)
    tokenizer, hf_model, device = ensure_slot_weights_loaded(slot)
    model_display = (
        get_base_model_display_name() if slot == ModelSlot.BASE else get_instruct_model_display_name()
    )

    if target_prediction is not None and target_token_id is not None:
        raise ValueError("target_prediction and target_token_id are mutually exclusive")

    enc = tokenizer(context, return_tensors="pt", return_offsets_mapping=False)
    input_ids = enc["input_ids"].to(device)
    n_tokens = input_ids.shape[1]

    if n_tokens > ATTRIBUTION_MAX_TOKEN_LENGTH:
        raise ValueError(
            "Context exceeds attribution length limit "
            f"({ATTRIBUTION_MAX_TOKEN_LENGTH} tokens); current length is {n_tokens} tokens."
        )

    # norm + lm_head（D1）
    inner_model = getattr(hf_model, "model", None)
    if inner_model is None or not hasattr(inner_model, "norm"):
        raise RuntimeError(
            "Cannot locate hf_model.model.norm; this model may not be a standard Qwen3ForCausalLM."
        )
    final_norm = inner_model.norm
    lm_head = hf_model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("hf_model.get_output_embeddings() returned None.")

    try:
        hf_model.eval()
        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        hidden_states = outputs.hidden_states  # tuple, len = L+1
        n_layers = len(hidden_states) - 1      # L

        # 解析目标 token（用最终层 logits，与 /prediction-attribute 口径一致）
        final_logits = outputs.logits[0, -1, :]
        final_probs = torch.softmax(final_logits, dim=-1)
        _, topk_ids = torch.topk(final_logits, DEFAULT_NEXT_TOKEN_TOPK)
        topk_tokens_final, topk_probs_final = decode_topk_ids_to_strings_and_rounded_probs(
            final_probs, tokenizer, topk_ids
        )

        use_top1 = target_prediction is None and target_token_id is None
        if use_top1:
            target_token_id = int(topk_ids[0].item())
            target_token = tokenizer.decode([target_token_id])
        elif target_token_id is not None:
            if target_token_id < 0 or target_token_id >= final_logits.shape[-1]:
                raise ValueError(
                    f"target_token_id out of range: {target_token_id} "
                    f"(vocab_size={int(final_logits.shape[-1])})"
                )
            target_token = tokenizer.decode([int(target_token_id)])
        else:
            assert target_prediction is not None
            tids = tokenizer.encode(target_prediction, add_special_tokens=False)
            if not tids:
                raise ValueError(f"Cannot tokenize target_prediction: {target_prediction!r}")
            target_token_id = tids[0]
            target_token = tokenizer.decode([target_token_id])

        tid = int(target_token_id)
        final_target_prob = round_to_sig_figs(final_probs[tid].item())

        # D1 自检：最终层投影 top-1 应等于 outputs.logits argmax（半精度用 isclose）
        standard_top1 = int(final_logits.argmax().item())
        with torch.no_grad():
            proj_hidden = final_norm(hidden_states[-1][:, -1:, :])
            proj_logits = lm_head(proj_hidden)[0, 0, :]
        proj_top1 = int(proj_logits.argmax().item())
        if proj_top1 != standard_top1:
            # 半精度允许边界扰动（两者均不是 target 时不影响结果）
            import warnings
            warnings.warn(
                f"Logit Lens self-check: final layer proj top-1 ({proj_top1}) != "
                f"outputs.logits top-1 ({standard_top1}); half-precision drift expected on MPS/CUDA."
            )

        # 逐层投影（D3：即时取 top-k，丢弃 [1,vocab] 张量）
        # 最终层直接用 outputs.logits（而非重新投影）确保与 final_target_prob 数值一致
        layers = []
        for layer_idx, h in enumerate(hidden_states):
            is_final = layer_idx == n_layers
            if is_final:
                # 直接复用已算好的 final_logits/final_probs，与 final_target_prob 口径完全一致
                layer_probs = final_probs
                lt, lp = topk_tokens_final, topk_probs_final
                tp = final_target_prob
            else:
                last_h = h[:, -1:, :]  # [1, 1, hidden]
                with torch.no_grad():
                    normed = final_norm(last_h)
                    layer_logits = lm_head(normed)[0, 0, :]  # [vocab]
                    layer_probs = torch.softmax(layer_logits, dim=-1)
                    _, layer_topk_ids = torch.topk(layer_logits, DEFAULT_NEXT_TOKEN_TOPK)

                lt, lp = decode_topk_ids_to_strings_and_rounded_probs(
                    layer_probs, tokenizer, layer_topk_ids
                )
                tp = layer_probs[tid].item()
                if not math.isfinite(tp):
                    tp = 0.0
                tp = round_to_sig_figs(tp)
                del layer_logits, layer_probs, normed

            layers.append({
                "layer": layer_idx,
                "is_embedding": layer_idx == 0,
                "topk_tokens": lt,
                "topk_probs": lp,
                "target_prob": tp,
            })

        eos_id = tokenizer.eos_token_id
        is_eos = eos_id is not None and tid == int(eos_id)

        return {
            "model": model_display,
            "target_token": target_token,
            "n_layers": n_layers,
            "final_target_prob": final_target_prob,
            "layers": layers,
            "debug_info": {"topk_tokens": topk_tokens_final, "topk_probs": topk_probs_final},
            "is_eos": is_eos,
        }
    finally:
        DeviceManager.clear_cache(device)
