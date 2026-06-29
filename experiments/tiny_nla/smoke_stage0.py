#!/usr/bin/env python3
"""
Stage 0: Environment & Injection Smoke

Loads Qwen3-0.6B-Base, checks device, inspects model config,
extracts activations from ~2/3 depth layer, finds stats,
identifies a single-token injection character, and runs
an input_embeds injection smoke test.

Output: prints report + writes nla_meta.yaml sidecar
"""

import os, sys, json, math, yaml, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = Path(__file__).resolve().parent / "nla_meta.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
INSTRUCT_MODEL = "Qwen/Qwen3-0.6B"

# ── device detection ───────────────────────────────────
def detect_device():
    """MPS preferred, CPU fallback. Prints diagnostics."""
    print("=" * 60)
    print("🔧 Device Detection")
    print("=" * 60)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  ✅ MPS available (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"  ⚠️  No accelerator, using CPU")
    print(f"  → Using device: {device}")
    print()
    return device


def mps_backward_smoke(device: torch.device):
    """Quick backward pass test on MPS to detect flaky ops."""
    if device.type != "mps":
        return True
    print("  🔬 MPS backward smoke test...")
    try:
        x = torch.randn(2, 64, 1536, device=device, requires_grad=True)
        loss = (x ** 2).mean()
        loss.backward()
        print(f"  ✅ MPS backward: OK (grad norm={x.grad.norm():.4f})")
        return True
    except Exception as e:
        print(f"  ❌ MPS backward FAILED: {e}")
        print(f"  ⚠️  Falling back to CPU")
        return False


def get_dtype_for_device(device: torch.device):
    """Pick dtype per device. MPS float16 can be flaky → use float32."""
    if device.type == "cuda":
        return torch.float16
    elif device.type == "mps":
        # MPS float16 backward can fail on some ops; float32 is safe
        print("  ℹ️  Using float32 on MPS (safer for backward)")
        return torch.float32
    return torch.float32


# ── model loading ──────────────────────────────────────
def load_model_and_tokenizer(model_name: str, device: torch.device, dtype: torch.dtype):
    """Load model in eval mode on target device with proper settings."""
    print(f"  Loading {model_name}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # safest for MPS/CPU
    ).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    load_time = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Loaded: {n_params/1e6:.1f}M params in {load_time:.1f}s")
    return model, tok


# ── model inspection ───────────────────────────────────
def inspect_model(model) -> dict:
    """Read config and print architecture info."""
    cfg = model.config
    info = {
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    }
    info["layer_index"] = round(info["num_hidden_layers"] * 2 / 3)

    print("=" * 60)
    print("📐 Model Architecture")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  layer_index (2/3 depth): {info['layer_index']}")
    print()
    return info


# ── activation extraction ──────────────────────────────
def extract_activation(model, tokenizer, text: str, layer_idx: int, device: torch.device):
    """
    Run forward with output_hidden_states=True, extract residual stream
    at given layer for the last token position.
    Returns: (activation_vector: torch.Tensor [d_model], token_ids, top_logits)
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )
    # hidden_states is tuple of (layer+1) x (batch, seq, d_model)
    # index layer_idx gives the residual stream AFTER that layer
    hidden = outputs.hidden_states[layer_idx]  # [1, seq, d_model]
    # take last token activation
    act = hidden[0, -1, :]  # [d_model]
    # also get logits for top-k
    logits = outputs.logits[0, -1, :]  # [vocab]
    top_vals, top_idxs = torch.topk(logits, k=10)
    return act.cpu(), inputs["input_ids"][0], top_idxs.cpu(), top_vals.cpu()


def compute_activation_stats(activations: list):
    """Compute L2 norm stats over a list of activation vectors."""
    norms_list = [a.norm().item() for a in activations]
    norms_t = torch.tensor(norms_list)
    stats = {
        "mean": norms_t.mean().item(),
        "p50": norms_t.median().item(),
        "p90": norms_t.kthvalue(max(1, int(len(norms_t) * 0.9))).values.item(),
        "max": norms_t.max().item(),
        "min": norms_t.min().item(),
        "std": norms_t.std().item(),
    }
    return stats


# ── injection token search ─────────────────────────────
def find_injection_token(tokenizer) -> tuple:
    """
    Find a rare character that tokenizes as a single token.
    Returns (char, token_id).
    """
    # Candidates: CJK rare chars that are likely single tokens
    candidates = [
        "㈎", "㈏", "㈐", "㈑", "㈒", "㈓", "㈔", "㈕", "㈖", "㈗",
        "㈘", "㈙", "㈚", "㈛", "㈜", "㈝", "㈞",
        "ⓐ", "ⓑ", "ⓒ", "ⓓ", "ⓔ", "ⓕ",
        "🀀", "🀁", "🀂", "🀃", "🀄", "🀅",
        "✪", "✫", "✬", "✭", "✮", "✯",
        "〓", "■", "□", "◆", "◇",
    ]

    print("=" * 60)
    print("🔤 Injection Token Search")
    print("=" * 60)

    for char in candidates:
        encoded = tokenizer.encode(char, add_special_tokens=False)
        if len(encoded) == 1:
            tid = encoded[0]
            # Verify roundtrip
            decoded = tokenizer.decode([tid])
            print(f"  ✅ Found single token: {char!r} -> id={tid} (decode: {decoded!r})")
            return char, tid

    # Fallback: search for ANY single-token char more systematically
    print("  ⚠️  Searching more broadly for single-token chars...")
    for codepoint in range(0x4E00, 0x9FFF):  # CJK Unified
        char = chr(codepoint)
        encoded = tokenizer.encode(char, add_special_tokens=False)
        if len(encoded) == 1:
            tid = encoded[0]
            decoded = tokenizer.decode([tid])
            print(f"  ✅ Found: {char!r} (U+{codepoint:04X}) -> id={tid}")
            return char, tid

    raise RuntimeError("Could not find a single-token injection character!")


# ── injection smoke test ───────────────────────────────
def injection_smoke(
    model, tokenizer, device, dtype,
    layer_idx: int, injection_scale: float, injection_char: str, injection_token_id: int,
):
    """
    Build a prompt, replace injection token embedding with scaled activation,
    run forward + generation. Verify no crash and shape mismatch.
    """
    print("=" * 60)
    print("🧪 Injection Smoke Test")
    print("=" * 60)

    prompt = f"Context: 这是一个测试句子。\n<concept>{injection_char}</concept>\n<explanation>"
    print(f"  Prompt: {prompt!r}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"  Input shape: {input_ids.shape}")

    # Find the injection token position
    inj_ids = (input_ids[0] == injection_token_id).nonzero(as_tuple=True)[0]
    if len(inj_ids) == 0:
        raise RuntimeError(f"Injection token {injection_char!r} (id={injection_token_id}) not found in prompt!")
    inj_pos = inj_ids[0].item()
    print(f"  Injection token at position: {inj_pos}")

    # Get a dummy activation vector: random vector scaled to injection_scale
    model_config = model.config
    d_model = model_config.hidden_size
    dummy_act = torch.randn(d_model, device=device, dtype=dtype)
    dummy_act = dummy_act / dummy_act.norm() * injection_scale

    # Forward with input_embeds injection
    with torch.no_grad():
        # Get base embeddings
        embeds = model.get_input_embeddings()(input_ids)  # [1, seq, d_model]
        # Replace the injection position's embedding with our activation
        embeds[0, inj_pos, :] = dummy_act

        # Forward with input_embeds
        outputs = model(inputs_embeds=embeds, output_hidden_states=True)
        logits = outputs.logits
        print(f"  Forward output shape: {logits.shape} ✅")

        # Extract the hidden state at injection position from layer layer_idx
        hidden = outputs.hidden_states[layer_idx]
        injected_act = hidden[0, inj_pos, :]
        print(f"  Hidden at injection pos, layer {layer_idx}: shape={injected_act.shape} ✅")

        # Compare input activation vs post-layer activation
        cos_sim = F.cosine_similarity(dummy_act.unsqueeze(0), injected_act.unsqueeze(0))
        print(f"  Cosine(input_act, post_layer_{layer_idx}_act): {cos_sim.item():.4f}")

    # Generation test — greedy
    print("\n  🔄 Generation test (greedy, max 20 tokens)...")
    gen_outputs = model.generate(
        inputs_embeds=embeds,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    print(f"  Generated: {gen_text!r}")
    print(f"  ✅ Injection generation completed without error")

    return {
        "forward_ok": True,
        "gen_ok": True,
        "cosine_post_injection": cos_sim.item(),
    }


# ── main ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🏗️  TINY-NLA STAGE 0 — Environment & Injection Smoke")
    print("=" * 60)
    print()

    # 1. Device
    device = detect_device()

    # 2. MPS backward smoke
    if not mps_backward_smoke(device):
        device = torch.device("cpu")

    # 3. dtype
    dtype = get_dtype_for_device(device)
    print(f"  Using dtype: {dtype}")
    print()

    # 4. Load model
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL, device, dtype)

    # 5. Inspect
    info = inspect_model(model)
    layer_idx = info["layer_index"]
    d_model = info["hidden_size"]

    # 6. Extract sample activations
    print("=" * 60)
    print("📊 Sample Activation Extraction")
    print("=" * 60)
    sample_texts = [
        "人工智能正在改变世界。",
        "深度学习模型可以学习复杂模式。",
        "今天天气真好，适合出去散步。",
        "Qwen3是一个强大的语言模型。",
    ]
    activations = []
    for txt in sample_texts:
        act, ids, topk_ids, topk_vals = extract_activation(model, tokenizer, txt, layer_idx, device)
        activations.append(act)
        top_tokens = [tokenizer.decode([t]) for t in topk_ids[:3]]
        print(f"  Text: {txt}")
        print(f"    Last token activation norm: {act.norm():.4f}")
        print(f"    Top-3 next tokens: {top_tokens}")

    # 7. Activation stats
    act_stats = compute_activation_stats(activations)
    print()
    print("📊 Activation L2 Norm Statistics")
    print("-" * 40)
    for k, v in act_stats.items():
        print(f"  {k}: {v:.4f}")

    injection_scale = round(act_stats["p50"], 4)
    print(f"\n  → injection_scale = p50 = {injection_scale}")

    # 8. Injection token search
    inj_char, inj_token_id = find_injection_token(tokenizer)
    print()

    # 9. Injection smoke test
    smoke_result = injection_smoke(
        model, tokenizer, device, dtype,
        layer_idx, injection_scale, inj_char, inj_token_id,
    )
    print()

    # 10. Write nla_meta.yaml
    meta = {
        "kind": "tiny_nla_model",
        "base_model": BASE_MODEL,
        "av_init_model": INSTRUCT_MODEL,
        "layer_index": layer_idx,
        "num_hidden_layers": info["num_hidden_layers"],
        "d_model": d_model,
        "activation_source": "residual_stream",
        "token_position_policy": "selected_token",
        "extraction": {
            "injection_scale": injection_scale,
            "mse_normalization": "l2_direction",
        },
        "tokens": {
            "injection_char": inj_char,
            "injection_token_id": inj_token_id,
        },
        "prompt_templates": {
            "av": f"<concept>{{injection_char}}</concept>\n<explanation>",
            "ar": f"<explanation>{{explanation_text}}</explanation>",
        },
        "training": {
            "device": device.type,
            "dtype": str(dtype),
            "dataset_size": 0,
            "teacher": "local_qwen_instruct",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "activation_stats": act_stats,
        "smoke_test": smoke_result,
    }

    with open(SIDECAR_PATH, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"  ✅ Sidecar written: {SIDECAR_PATH}")

    # 11. Summary
    print()
    print("=" * 60)
    print("✅ STAGE 0 COMPLETE")
    print("=" * 60)
    print(f"  Device:           {device.type}")
    print(f"  dtype:            {dtype}")
    print(f"  Layers:           {info['num_hidden_layers']}")
    print(f"  Selected layer:   {layer_idx} ({layer_idx/info['num_hidden_layers']*100:.0f}% depth)")
    print(f"  d_model:          {d_model}")
    print(f"  Injection char:   {inj_char!r} (id={inj_token_id})")
    print(f"  Injection scale:  {injection_scale}")
    print(f"  Forward smoke:    {'✅ PASS' if smoke_result['forward_ok'] else '❌ FAIL'}")
    print(f"  Generation smoke: {'✅ PASS' if smoke_result['gen_ok'] else '❌ FAIL'}")
    print(f"  Sidecar:          {SIDECAR_PATH}")
    print()


if __name__ == "__main__":
    main()
