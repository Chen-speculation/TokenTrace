"""
消融归因：对下一 token 预测做基于遮挡（occlusion）的因果归因。

对每个输入 token，把其 embedding 替换为基线向量（全序列均值），重算目标 token 概率，
score = baseline_target_prob − occluded_target_prob（正=支撑目标，负=抑制目标）。
无需反向传播；N+1 个变体拼成一个 batch 一次前向。
"""

import math
import os
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
    PREDICTION_ATTR_MODEL_BASE,
    PREDICTION_ATTR_MODEL_INSTRUCT,
    slot_for_prediction_attr_model,
)

# 内存分批阈值：(N+1) × seq 超过此值时按行分批；可通过环境变量覆盖
_DEFAULT_BATCH_THRESHOLD = 8192


def _batch_threshold() -> int:
    try:
        return int(os.environ.get("ABLATION_BATCH_THRESHOLD", _DEFAULT_BATCH_THRESHOLD))
    except (TypeError, ValueError):
        return _DEFAULT_BATCH_THRESHOLD


def analyze_ablation_attribution(
    context: str,
    target_prediction: Optional[str] = None,
    *,
    model: str,
    target_token_id: Optional[int] = None,
) -> Dict:
    """
    对 context 中各 token 做消融归因。

    Args:
        context: 输入上下文文本（token 数不得超过 ATTRIBUTION_MAX_TOKEN_LENGTH）
        target_prediction: 归因目标文本；tokenize 后取第一个 token。
        target_token_id: 归因目标 token id（与 target_prediction 互斥）。
        两者均省略时自动使用 top-1（贪心解码）。
        model: ``base`` 或 ``instruct``

    Returns:
        {
            "model": str,
            "target_token": str,
            "target_prob": float,          # baseline 目标概率
            "token_attribution": [{"offset": [s, e], "raw": str, "score": float}, ...],
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

    use_top1 = target_prediction is None and target_token_id is None

    enc = tokenizer(context, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    offset_mapping = enc["offset_mapping"][0].tolist()
    n_tokens = input_ids.shape[1]

    if n_tokens > ATTRIBUTION_MAX_TOKEN_LENGTH:
        raise ValueError(
            "Context exceeds attribution length limit "
            f"({ATTRIBUTION_MAX_TOKEN_LENGTH} tokens); current length is {n_tokens} tokens."
        )

    embed_layer = hf_model.get_input_embeddings()
    with torch.no_grad():
        baseline_embeds = embed_layer(input_ids)  # [1, seq, hidden]

    # D2: 基线向量 = EOS token embedding（固定、语义中性）
    # 使用 EOS 而非序列均值，原因：
    #   1. 当 n_occlude=1 时序列均值 = 该 token 自身 → 遮挡无效 → score=0
    #   2. EOS 是模型具备合理表示的特殊 token，不会产生意外语义偏移
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # fallback：无 EOS 时用序列均值
        BASELINE_VECTOR = baseline_embeds[0].mean(dim=0, keepdim=True)  # [1, hidden]
    else:
        eos_tensor = torch.tensor([[eos_token_id]], device=device)
        with torch.no_grad():
            eos_embedding = embed_layer(eos_tensor)  # [1, 1, hidden]
        BASELINE_VECTOR = eos_embedding[0]  # [1, hidden]

    try:
        hf_model.eval()

        # --- baseline 前向：解析目标 token ---
        with torch.no_grad():
            out = hf_model(inputs_embeds=baseline_embeds, output_attentions=False, use_cache=False)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        _, topk_ids = torch.topk(logits, DEFAULT_NEXT_TOKEN_TOPK)
        topk_tokens, topk_probs = decode_topk_ids_to_strings_and_rounded_probs(
            probs, tokenizer, topk_ids
        )

        if use_top1:
            target_token_id = int(topk_ids[0].item())
            target_token = tokenizer.decode([target_token_id])
        elif target_token_id is not None:
            if target_token_id < 0 or target_token_id >= logits.shape[-1]:
                raise ValueError(
                    f"target_token_id out of range: {target_token_id} (vocab_size={int(logits.shape[-1])})"
                )
            target_token = tokenizer.decode([int(target_token_id)])
        else:
            assert target_prediction is not None
            target_ids = tokenizer.encode(target_prediction, add_special_tokens=False)
            if not target_ids:
                raise ValueError(f"Cannot tokenize target_prediction: {target_prediction!r}")
            target_token_id = target_ids[0]
            target_token = tokenizer.decode([target_token_id])

        assert target_token_id is not None
        baseline_target_prob = probs[int(target_token_id)].item()
        baseline_target_logit = logits[int(target_token_id)].item()

        print(f"🔍 ablation: target_token={target_token!r}, target_id={int(target_token_id)}, "
              f"baseline_prob={baseline_target_prob:.6e}, baseline_logit={baseline_target_logit:.4f}, "
              f"occlude_tokens={len([i for i, (s, e) in enumerate(offset_mapping) if s < e])}")

        # --- 构造遮挡 batch ---
        # 对每个非特殊 token（span 非空）构造一行：把第 i 行替换为基线向量
        occlude_indices = [i for i, (s, e) in enumerate(offset_mapping) if s < e]
        n_occlude = len(occlude_indices)

        if n_occlude == 0:
            # 无可归因 token（不应发生，但防御）
            return {
                "model": model_display,
                "target_token": target_token,
                "target_prob": round_to_sig_figs(baseline_target_prob),
                "token_attribution": [],
                "debug_info": {"topk_tokens": topk_tokens, "topk_probs": topk_probs},
                "is_eos": _is_eos(tokenizer, target_token_id),
            }

        # 分批前向：按 (n_occlude + 1) × seq 判断
        threshold = _batch_threshold()
        batch_size_limit = max(1, threshold // n_tokens)

        # 收集各遮挡变体在 target_token_id 处的 logit 和 prob
        occluded_probs = []
        occluded_logits = []

        for batch_start in range(0, n_occlude, batch_size_limit):
            batch_indices = occlude_indices[batch_start: batch_start + batch_size_limit]
            rows = []
            for i in batch_indices:
                row = baseline_embeds[0].clone()  # [seq, hidden]
                row[i] = BASELINE_VECTOR[0]
                rows.append(row)
            batch_embeds = torch.stack(rows, dim=0)  # [B, seq, hidden]

            with torch.no_grad():
                batch_out = hf_model(
                    inputs_embeds=batch_embeds,
                    output_attentions=False,
                    use_cache=False,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

            batch_logits_last = batch_out.logits[:, -1, :]  # [B, vocab]
            batch_probs = torch.softmax(batch_logits_last, dim=-1)

            tid = int(target_token_id)
            occluded_probs.extend(batch_probs[:, tid].float().cpu().tolist())
            occluded_logits.extend(batch_logits_last[:, tid].float().cpu().tolist())

            DeviceManager.clear_cache(device)

        # --- 组装 token_attribution ---
        # score = delta_logit（而非 delta_prob），
        # 因 delta_prob = baseline_prob - occluded_prob 天然受限于 ±baseline_prob，
        # 低概率 target 时信号被压扁。logit 空间无此边界约束。
        token_attribution = []
        nan_count = 0
        occ_iter = iter(zip(occluded_probs, occluded_logits))

        for (s, e), _ in zip(offset_mapping, range(n_tokens)):
            if s >= e:
                continue
            occ_p, occ_l = next(occ_iter)
            # score = delta_logit（不受概率边界约束）
            raw_score = baseline_target_logit - occ_l
            # delta_prob 仍保留供参考
            raw_delta_prob = baseline_target_prob - occ_p
            if not math.isfinite(raw_score):
                raw_score = 0.0
                nan_count += 1
            token_attribution.append({
                "offset": [s, e],
                "raw": context[s:e],
                "score": round_to_sig_figs(raw_score),
                "delta_prob": round_to_sig_figs(raw_delta_prob) if math.isfinite(raw_delta_prob) else 0.0,
                "delta_logit": round_to_sig_figs(raw_score) if math.isfinite(raw_score) else 0.0,
            })

        if nan_count > 0:
            print(f"⚠️ ablation token_attribution 中有 {nan_count} 个 score 为 NaN/Inf，已替换为 0。")

        return {
            "model": model_display,
            "target_token": target_token,
            "target_prob": round_to_sig_figs(baseline_target_prob),
            "token_attribution": token_attribution,
            "debug_info": {"topk_tokens": topk_tokens, "topk_probs": topk_probs},
            "is_eos": _is_eos(tokenizer, target_token_id),
        }
    finally:
        DeviceManager.clear_cache(device)


def _is_eos(tokenizer, token_id: int) -> bool:
    eos_id = tokenizer.eos_token_id
    return eos_id is not None and int(token_id) == int(eos_id)
