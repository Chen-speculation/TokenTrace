#!/usr/bin/env python3
"""
Phase 3: GRPO RL — AV+AR joint training.
Reward = -MSE(L2_norm(original), L2_norm(reconstructed))
Monitor: tensorboard --logdir artifacts/tiny_nla/runs

Usage:
  python train_rl_grpo.py [--steps 1000] [--group-size 4] [--batch 8]
"""
import json, yaml, random, time, argparse, os
from pathlib import Path
from datetime import datetime

# Disable HuggingFace network access — use local cache only, no HEAD checks to huggingface.co
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

REPO_ROOT  = Path(__file__).resolve().parents[2]
ARTIFACTS  = REPO_ROOT / "artifacts" / "tiny_nla"
META_PATH  = Path(__file__).resolve().parent / "nla_meta.yaml"
RUNS_DIR   = ARTIFACTS / "runs"

meta       = yaml.safe_load(open(META_PATH))
D_MODEL    = meta["d_model"]
LAYER      = meta["layer_index"]
INJ_CHAR   = meta["tokens"]["injection_char"]
INJ_TOK_ID = meta["tokens"]["injection_token_id"]
INJ_SCALE  = meta["extraction"]["injection_scale"]
BASE_MODEL = meta["base_model"]
INST_MODEL = meta["av_init_model"]


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def scale_acts(acts, dev):
    a = acts.to(dev).float()
    return a / a.norm(dim=-1, keepdim=True).clamp(1e-6) * INJ_SCALE


def norm_acts(acts, dev):
    a = acts.to(dev).float()
    return a / a.norm(dim=-1, keepdim=True).clamp(1e-6)


# ── Load activations only (no teacher needed for RL) ─
def load_activations():
    table = pq.read_table(ARTIFACTS / "activations_v2.parquet")
    acts  = torch.tensor([table["activation"][i].as_py()
                          for i in range(len(table))], dtype=torch.float32)
    log(f"Loaded {len(acts)} activations | norm {acts.norm(dim=-1).min():.1f}–{acts.norm(dim=-1).max():.1f}")
    return acts


# ── AR head (frozen during RL, used as reward scorer) ─
class ARHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, h):
        return F.normalize(self.proj(h), dim=-1)


def load_ar(dev):
    ckpt = torch.load(ARTIFACTS / "checkpoints" / "ar_v2" / "ar_head_v2.pt",
                      map_location=dev, weights_only=True)
    head = ARHead(D_MODEL).to(dev)
    head.load_state_dict(ckpt["head"])
    head.eval()
    for p in head.parameters(): p.requires_grad_(False)
    log(f"AR head loaded (val_cos={ckpt.get('val_cosine',0):.4f})")
    return head


# ── AV: generate explanations for a batch of activations ─
def av_generate(av_model, tok, acts_scaled, dev, max_new=64, temperature=1.0, num_return=1):
    """
    acts_scaled: [B, D] already scaled to INJ_SCALE
    Returns list of B lists, each with num_return explanation strings.
    """
    prompt  = f"<concept>{INJ_CHAR}</concept>\n<explanation>"
    p_ids   = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
    p_mask  = torch.ones(1, p_ids.shape[1], device=dev, dtype=torch.long)
    inj_pos = (p_ids[0] == INJ_TOK_ID).nonzero(as_tuple=True)[0][0].item()

    all_expls = []
    for b in range(acts_scaled.shape[0]):
        embeds = av_model.get_input_embeddings()(p_ids).clone()  # [1, P, D]
        embeds[0, inj_pos] = acts_scaled[b].to(embeds.dtype)
        expls = []
        for _ in range(num_return):
            with torch.no_grad():
                out = av_model.generate(
                    inputs_embeds=embeds, attention_mask=p_mask,
                    max_new_tokens=max_new,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tok.eos_token_id,
                )
            expls.append(tok.decode(out[0], skip_special_tokens=True).strip())
        all_expls.append(expls)
    return all_expls


# ── AR reconstruct explanations → activation ─────────
def ar_reconstruct(ar_backbone, ar_head, tok, explanations, dev):
    """
    explanations: list of strings
    Returns: [N, D] normalized reconstructions (float32 for stable cosine)
    """
    tok.pad_token_id = tok.eos_token_id
    enc = tok(explanations, return_tensors="pt", padding=True,
              truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        h = ar_backbone(**enc, output_hidden_states=True).hidden_states[-1]
        lens = enc["attention_mask"].sum(1) - 1
        last = h[torch.arange(len(h)), lens]
        recon = ar_head(last)
    return recon.float()  # cast to float32 for downstream cosine


# ── GRPO loss ─────────────────────────────────────────
def grpo_loss(av_model, tok, acts_scaled, acts_norm, ar_backbone, ar_head,
              dev, group_size, max_new, kl_coef, ref_av, temperature=1.2):
    """
    Returns: (policy_loss, mean_reward, reward_std)
    """
    B = acts_scaled.shape[0]
    prompt  = f"<concept>{INJ_CHAR}</concept>\n<explanation>"
    p_ids   = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
    inj_pos = (p_ids[0] == INJ_TOK_ID).nonzero(as_tuple=True)[0][0].item()

    all_log_probs = []  # [B, G]
    all_rewards   = []  # [B, G]
    ref_log_probs = []  # [B, G] for KL

    for b in range(B):
        embeds_base = av_model.get_input_embeddings()(p_ids).clone()
        embeds_base[0, inj_pos] = acts_scaled[b].to(embeds_base.dtype)

        b_lp, b_rew, b_rlp = [], [], []
        for g in range(group_size):
            # Sample with temperature
            with torch.no_grad():
                out = av_model.generate(
                    inputs_embeds=embeds_base.clone(),
                    attention_mask=torch.ones(1, p_ids.shape[1], device=dev),
                    max_new_tokens=max_new, do_sample=True, temperature=temperature,
                    pad_token_id=tok.eos_token_id, return_dict_in_generate=True,
                    output_scores=True,
                )
            gen_ids = out.sequences[0]  # [T]
            expl    = tok.decode(gen_ids, skip_special_tokens=True).strip()

            # Reward: AR roundtrip cosine
            recon   = ar_reconstruct(ar_backbone, ar_head, tok, [expl], dev)  # [1,D]
            orig_n  = acts_norm[b].to(dev).unsqueeze(0)
            reward  = (recon * orig_n).sum(-1).item()  # cosine similarity
            b_rew.append(reward)

            # Policy log-prob of generated tokens (recompute with grad)
            full_embeds = av_model.get_input_embeddings()(p_ids).clone()
            full_embeds[0, inj_pos] = acts_scaled[b].to(full_embeds.dtype)
            # Append gen_ids to prompt via input_ids for teacher forcing
            # (gen_ids are new tokens only from generate with inputs_embeds)
            if len(gen_ids) > 0:
                gen_embeds = av_model.get_input_embeddings()(gen_ids.unsqueeze(0))
                all_embeds = torch.cat([full_embeds, gen_embeds], dim=1)
                attn = torch.ones(1, all_embeds.shape[1], device=dev)
                logits = av_model(inputs_embeds=all_embeds, attention_mask=attn).logits
                # log-prob of generated part
                P = p_ids.shape[1]
                lp = 0.0
                for t, tok_id in enumerate(gen_ids):
                    lp = lp + F.log_softmax(logits[0, P+t-1], dim=-1)[tok_id]
                lp = lp / max(len(gen_ids), 1)
            else:
                lp = torch.tensor(0.0, device=dev)
            b_lp.append(lp)

            # Reference log-prob (KL penalty)
            with torch.no_grad():
                ref_logits = ref_av(inputs_embeds=all_embeds.detach(), attention_mask=attn).logits
                rlp = sum(F.log_softmax(ref_logits[0, P+t-1], dim=-1)[tok_id]
                          for t, tok_id in enumerate(gen_ids))
                rlp = rlp / max(len(gen_ids), 1)
            b_rlp.append(rlp if isinstance(rlp, float) else rlp.item())

        all_rewards.append(b_rew)
        all_log_probs.append(b_lp)
        ref_log_probs.append(b_rlp)

    # GRPO: normalize rewards within group
    loss = torch.tensor(0.0, device=dev, requires_grad=True)
    flat_rewards = [r for group in all_rewards for r in group]
    mean_rew = sum(flat_rewards) / len(flat_rewards)

    for b in range(B):
        rewards = all_rewards[b]
        mu = sum(rewards) / group_size
        std = (sum((r-mu)**2 for r in rewards)/group_size)**0.5 + 1e-6
        for g in range(group_size):
            adv = (rewards[g] - mu) / std
            kl  = all_log_probs[b][g] - ref_log_probs[b][g]
            loss = loss - (adv * all_log_probs[b][g] - kl_coef * kl)

    return loss / (B * group_size), mean_rew, (sum((r-mean_rew)**2 for r in flat_rewards)/len(flat_rewards))**0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int,   default=1000)
    parser.add_argument("--group-size", type=int,   default=8)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--max-new",    type=int,   default=64)
    parser.add_argument("--kl-coef",    type=float, default=0.05)
    parser.add_argument("--lr-av",      type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Sampling temperature for exploration (higher = more diverse)")
    parser.add_argument("--eval-every", type=int,   default=50)
    parser.add_argument("--save-every", type=int,   default=100)
    parser.add_argument("--resume", action="store_true",
                        help="Resume RL from latest checkpoint")
    args = parser.parse_args()

    dev = get_device()
    run_name = f"rl_{datetime.now().strftime('%m%d_%H%M')}"
    writer   = SummaryWriter(RUNS_DIR / run_name)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"GRPO RL | device={dev} | steps={args.steps} | G={args.group_size} | B={args.batch} | T={args.temperature} | kl={args.kl_coef} | lr={args.lr_av}")
    log(f"Monitor: tensorboard --logdir {RUNS_DIR}")

    acts = load_activations()
    acts_n = norm_acts(acts, "cpu")

    # Load AR backbone (frozen scorer)
    log(f"Loading AR backbone ({BASE_MODEL}, float16, sdpa)...")
    ar_backbone = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(dev)
    ar_backbone.eval()
    for p in ar_backbone.parameters(): p.requires_grad_(False)
    ar_head = load_ar(dev)
    ar_head = ar_head.half()  # match backbone dtype
    tok_ar  = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok_ar.pad_token_id = tok_ar.eos_token_id

    # Load AV from SFT or resume checkpoint
    rl_resume_dir = ARTIFACTS / "checkpoints" / "av_rl_resume"
    if args.resume and (rl_resume_dir / "training_state.pt").exists():
        av_ckpt = rl_resume_dir
        log(f"Resuming AV from {av_ckpt}...")
    else:
        if args.resume:
            log(f"⚠️ No RL resume checkpoint found, starting from SFT")
        av_ckpt = ARTIFACTS / "checkpoints" / "av_v2"
        log(f"Loading AV from {av_ckpt}...")
    av_base = AutoModelForCausalLM.from_pretrained(
        INST_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(dev)
    av_model = PeftModel.from_pretrained(av_base, av_ckpt, is_trainable=True).to(dev)
    tok_av   = AutoTokenizer.from_pretrained(av_ckpt, trust_remote_code=True)
    tok_av.pad_token_id = tok_av.eos_token_id

    # Reference model (frozen copy for KL)
    ref_base = AutoModelForCausalLM.from_pretrained(
        INST_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(dev)
    ref_av = PeftModel.from_pretrained(ref_base, av_ckpt).to(dev)
    ref_av.eval()
    for p in ref_av.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(av_model.parameters(), lr=args.lr_av)

    # SFT baseline or resume state
    start_step = 1
    if args.resume and (rl_resume_dir / "training_state.pt").exists():
        ts = torch.load(rl_resume_dir / "training_state.pt",
                        map_location=dev, weights_only=False)
        opt.load_state_dict(ts["optimizer"])
        start_step = ts["step"] + 1
        best_cos = ts["best_cos"]
        best_step = ts["best_step"]
        sft_cos = ts["sft_cos"]
        log(f"Resumed from step {start_step} | best_cos={best_cos:.4f} at step {best_step} | sft_cos={sft_cos:.4f}")
        writer.add_scalar("rl/sft_baseline", sft_cos, 0)
    else:
        log("Computing SFT baseline roundtrip...")
        sft_cosines = []
        with torch.no_grad():
            sample_idx = random.sample(range(len(acts)), min(50, len(acts)))
            for i in sample_idx:
                act_s = scale_acts(acts[i:i+1], dev)
                act_n = acts_n[i:i+1].to(dev)
                expls = av_generate(av_model, tok_av, act_s, dev, args.max_new, temperature=0, num_return=1)
                recon = ar_reconstruct(ar_backbone, ar_head, tok_ar, expls[0], dev)
                sft_cosines.append((recon * act_n).sum(-1).item())
        sft_cos = sum(sft_cosines)/len(sft_cosines)
        log(f"SFT baseline roundtrip cosine: {sft_cos:.4f}")
        writer.add_scalar("rl/sft_baseline", sft_cos, 0)
        best_cos, best_step = sft_cos, 0

    random.seed(42)

    for step in range(start_step, args.steps+1):
        av_model.train()
        batch_idx = random.sample(range(len(acts)), args.batch)
        batch_acts   = acts[batch_idx]
        batch_acts_n = acts_n[batch_idx]
        acts_scaled  = scale_acts(batch_acts, dev)

        loss, mean_rew, rew_std = grpo_loss(
            av_model, tok_av, acts_scaled, batch_acts_n,
            ar_backbone, ar_head, dev,
            args.group_size, args.max_new, args.kl_coef, ref_av,
            temperature=args.temperature,
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(av_model.parameters(), 1.0)
        opt.step()

        writer.add_scalar("rl/loss",       loss.item(),  step)
        writer.add_scalar("rl/mean_reward", mean_rew,    step)
        writer.add_scalar("rl/reward_std",  rew_std,     step)

        if step % 10 == 0:
            log(f"Step {step:4d}/{args.steps} | loss={loss.item():.4f} | "
                f"reward={mean_rew:.4f}±{rew_std:.4f}")

        # Eval roundtrip
        if step % args.eval_every == 0:
            av_model.eval()
            cos_vals = []
            with torch.no_grad():
                eval_idx = random.sample(range(len(acts)), min(50, len(acts)))
                for i in eval_idx:
                    act_s = scale_acts(acts[i:i+1], dev)
                    act_n = acts_n[i:i+1].to(dev)
                    expls = av_generate(av_model, tok_av, act_s, dev, args.max_new, temperature=0)
                    recon = ar_reconstruct(ar_backbone, ar_head, tok_ar, expls[0], dev)
                    cos_vals.append((recon * act_n).sum(-1).item())
            val_cos = sum(cos_vals)/len(cos_vals)
            delta   = val_cos - sft_cos
            flag    = "✓ BEST" if val_cos > best_cos else ("⚠️ regress" if val_cos < sft_cos - 0.02 else "")
            log(f"[EVAL step {step}] roundtrip_cos={val_cos:.4f} | ΔSFT={delta:+.4f} | best={best_cos:.4f} {flag}")
            writer.add_scalar("rl/val_roundtrip_cosine", val_cos, step)
            writer.add_scalar("rl/delta_vs_sft",         delta,   step)

            if val_cos > best_cos:
                best_cos, best_step = val_cos, step
                av_model.save_pretrained(ARTIFACTS / "checkpoints" / "av_rl_best")
                log(f"  Saved best RL checkpoint (cos={best_cos:.4f})")

            # Sample explanation
            sample_act = scale_acts(acts[eval_idx[0]:eval_idx[0]+1], dev)
            expls = av_generate(av_model, tok_av, sample_act, dev, args.max_new, temperature=0)
            writer.add_text("rl/sample_expl", expls[0][0], step)
            log(f"  Sample: {expls[0][0][:80]}")

        if step % args.save_every == 0:
            av_model.save_pretrained(ARTIFACTS / "checkpoints" / f"av_rl_step{step}")
            # Save resume checkpoint
            rl_resume_dir = ARTIFACTS / "checkpoints" / "av_rl_resume"
            rl_resume_dir.mkdir(parents=True, exist_ok=True)
            av_model.save_pretrained(rl_resume_dir)
            torch.save({
                "step": step,
                "optimizer": opt.state_dict(),
                "best_cos": best_cos,
                "best_step": best_step,
                "sft_cos": sft_cos,
            }, rl_resume_dir / "training_state.pt")
            log(f"  Saved resume checkpoint at step {step}")

    log(f"\nRL done | best_cos={best_cos:.4f} at step {best_step} | SFT_cos={sft_cos:.4f}")
    log(f"Best checkpoint: {ARTIFACTS/'checkpoints'/'av_rl_best'}")
    writer.add_hparams(
        {"lr":args.lr_av,"G":args.group_size,"B":args.batch,"kl":args.kl_coef,"temp":args.temperature},
        {"hparam/best_rl_cosine": best_cos, "hparam/sft_cosine": sft_cos}
    )
    writer.close()


if __name__ == "__main__":
    main()
