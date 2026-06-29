#!/usr/bin/env python3
"""Final model evaluation: SFT vs RL roundtrip cosine on 100 random samples."""
import os, yaml, random, argparse
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
REPO_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS  = REPO_ROOT / "artifacts" / "tiny_nla"
META       = yaml.safe_load(open(Path(__file__).resolve().parent / "nla_meta.yaml"))
D_MODEL    = META["d_model"]
INJ_CHAR   = META["tokens"]["injection_char"]
INJ_TOK_ID = META["tokens"]["injection_token_id"]
INJ_SCALE  = META["extraction"]["injection_scale"]
BASE_MODEL = META["base_model"]
INST_MODEL = META["av_init_model"]


class ARHead(nn.Module):
    def __init__(self, d): super().__init__(); self.proj = nn.Linear(d, d, bias=False)
    def forward(self, h): return F.normalize(self.proj(h), dim=-1)


def tprint(s): print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)


def load_activations():
    t = pq.read_table(ARTIFACTS / "activations_v2.parquet")
    acts = torch.tensor([t["activation"][i].as_py() for i in range(len(t))], dtype=torch.float32)
    tprint(f"Loaded {len(acts)} activations")
    return acts


def load_av_model(path, device):
    tprint(f"Loading AV from {path}...")
    tok = AutoTokenizer.from_pretrained(INST_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        INST_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(base, path)
    model.eval()
    return model, tok


def load_ar(device):
    ckpt = torch.load(ARTIFACTS / "checkpoints" / "ar_v2" / "ar_head_v2.pt",
                      map_location=device, weights_only=True)
    head = ARHead(D_MODEL).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()
    tprint(f"AR head loaded (val_cos={ckpt.get('val_cosine',0):.4f})")
    return head


def load_ar_backbone(device):
    tprint(f"Loading AR backbone ({BASE_MODEL})...")
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(device)
    m.eval()
    return m


def generate(model, tok, act_scaled, max_new=64):
    prompt = f"<concept>{INJ_CHAR}</concept>\n<explanation>"
    p_ids = tok(prompt, return_tensors="pt")["input_ids"].to(act_scaled.device)
    p_mask = torch.ones(1, p_ids.shape[1], device=act_scaled.device, dtype=torch.long)
    inj_pos = (p_ids[0] == INJ_TOK_ID).nonzero(as_tuple=True)[0][0].item()
    embeds = model.get_input_embeddings()(p_ids).clone()
    embeds[0, inj_pos] = act_scaled[0].to(embeds.dtype)
    with torch.no_grad():
        out = model.generate(inputs_embeds=embeds, attention_mask=p_mask,
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def reconstruct(ar_backbone, ar_head, tok, explanation):
    tok.pad_token_id = tok.eos_token_id
    dev = ar_head.proj.weight.device
    enc = tok([explanation], return_tensors="pt", padding=True,
              truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        h = ar_backbone(**enc, output_hidden_states=True).hidden_states[-1]
        lens = enc["attention_mask"].sum(1) - 1
        last = h[0, lens[0]]
        recon = ar_head(last.unsqueeze(0))
    return recon.float()


def eval_one(model, tok, act_raw, ar_backbone, ar_head, tok_ar):
    act_s = act_raw.unsqueeze(0) / act_raw.norm() * INJ_SCALE
    act_s = act_s.to(ar_head.proj.weight.device)
    act_n = (act_raw.unsqueeze(0) / act_raw.norm()).to(ar_head.proj.weight.device)
    expl  = generate(model, tok, act_s)
    recon = reconstruct(ar_backbone, ar_head, tok_ar, expl)
    return (recon * act_n).sum(-1).item(), expl


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    random.seed(args.seed)

    print(f"\n{'='*55}")
    print(f"  Tiny-NLA Final Eval | samples={args.num_samples} | seed={args.seed}")
    print(f"{'='*55}\n")

    acts = load_activations()
    ar_head = load_ar(device)
    ar_bb   = load_ar_backbone(device)
    tok_ar  = AutoTokenizer.from_pretrained(BASE_MODEL)

    sft_model, sft_tok = load_av_model(ARTIFACTS / "checkpoints" / "av_v2", device)
    rl_model,  rl_tok  = load_av_model(ARTIFACTS / "checkpoints" / "av_rl_best", device)

    eval_idx = random.sample(range(len(acts)), args.num_samples)

    sft_cos, rl_cos = [], []
    sft_expls, rl_expls = [], []

    tprint(f"Evaluating {args.num_samples} samples...")
    for k, idx in enumerate(eval_idx):
        act = acts[idx]
        c1, e1 = eval_one(sft_model, sft_tok, act, ar_bb, ar_head, tok_ar)
        c2, e2 = eval_one(rl_model,  rl_tok,  act, ar_bb, ar_head, tok_ar)
        sft_cos.append(c1); rl_cos.append(c2)
        sft_expls.append(e1); rl_expls.append(e2)
        if (k + 1) % 20 == 0:
            tprint(f"  {k+1}/{args.num_samples} | sft={sum(sft_cos)/(k+1):.4f} | rl={sum(rl_cos)/(k+1):.4f}")

    sft_mean = sum(sft_cos) / len(sft_cos)
    rl_mean  = sum(rl_cos)  / len(rl_cos)
    delta    = rl_mean - sft_mean
    gains    = [rl_cos[i] - sft_cos[i] for i in range(args.num_samples)]
    positive = sum(1 for g in gains if g > 0)

    print(f"\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  {'':16} {'SFT':>10} {'RL':>10} {'Δ':>10}")
    print(f"  {'Mean':16} {sft_mean:10.4f} {rl_mean:10.4f} {delta:+10.4f}")
    print(f"  {'Best':16} {max(sft_cos):10.4f} {max(rl_cos):10.4f} {max(rl_cos)-max(sft_cos):+10.4f}")
    print(f"  {'Worst':16} {min(sft_cos):10.4f} {min(rl_cos):10.4f} {min(rl_cos)-min(sft_cos):+10.4f}")
    print(f"  {'RL wins':16} {positive}/{args.num_samples} ({100*positive/args.num_samples:.0f}%)")
    print(f"{'='*55}")

    # Top 5 improvements
    print(f"\n── TOP 5 GAINS (RL − SFT) ──")
    sorted_idx = sorted(range(args.num_samples), key=lambda i: gains[i], reverse=True)
    for rank, i in enumerate(sorted_idx[:5]):
        print(f"\n  #{rank+1} Δ={gains[i]:+.4f} | SFT cos={sft_cos[i]:.4f}")
        print(f"  SFT: {sft_expls[i][:130]}")
        print(f"  RL:  {rl_expls[i][:130]}")

    # Bottom 5
    print(f"\n── BOTTOM 5 (RL regressions) ──")
    for rank, i in enumerate(sorted_idx[-5:]):
        print(f"\n  #{args.num_samples-4+rank} Δ={gains[i]:+.4f} | RL cos={rl_cos[i]:.4f}")
        print(f"  SFT: {sft_expls[i][:130]}")
        print(f"  RL:  {rl_expls[i][:130]}")

    print(f"\n{tprint('Done.')}")
