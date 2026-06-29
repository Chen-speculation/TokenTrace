#!/usr/bin/env python3
"""
Phase 2: AR + AV SFT on v2 data (9810 records).
Real-time monitoring via TensorBoard + console.

Usage:
  python train_sft_v2.py --stage ar   # AR only
  python train_sft_v2.py --stage av   # AV only
  python train_sft_v2.py              # both (AR first)

Monitor:
  tensorboard --logdir artifacts/tiny_nla/runs
"""
import json, yaml, random, time, argparse, sys, os
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS  = REPO_ROOT / "artifacts" / "tiny_nla"
META_PATH  = Path(__file__).resolve().parent / "nla_meta.yaml"
RUNS_DIR   = ARTIFACTS / "runs"

meta       = yaml.safe_load(open(META_PATH))
D_MODEL    = meta["d_model"]
INJ_CHAR   = meta["tokens"]["injection_char"]
INJ_TOK_ID = meta["tokens"]["injection_token_id"]
INJ_SCALE  = meta["extraction"]["injection_scale"]
BASE_MODEL = meta["base_model"]
INST_MODEL = meta["av_init_model"]


def get_device():
    if torch.cuda.is_available():  return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Data loading ─────────────────────────────────────
def load_data():
    table  = pq.read_table(ARTIFACTS / "activations_v2.parquet")
    labels = {(r["text_idx"], r["pos"]): r["teacher_explanation"]
              for r in json.load(open(ARTIFACTS / "teacher_labels_v2.json"))}

    records, acts = [], []
    for i in range(len(table)):
        ti, pos = table["text_idx"][i].as_py(), table["pos"][i].as_py()
        expl = labels.get((ti, pos), "")
        if not expl or len(expl) < 5:
            continue
        act = torch.tensor(table["activation"][i].as_py(), dtype=torch.float32)
        records.append({"text_idx": ti, "pos": pos,
                        "text": table["text"][i].as_py(),
                        "token_text": table["token_text"][i].as_py(),
                        "teacher_explanation": expl})
        acts.append(act)

    acts_t = torch.stack(acts)
    acts_n = F.normalize(acts_t, dim=-1)
    log(f"Loaded {len(records)} records | norm {acts_t.norm(dim=-1).min():.1f}–{acts_t.norm(dim=-1).max():.1f}")
    return records, acts_t, acts_n


# ══════════════════════════════════════════════════════
# AR SFT
# ══════════════════════════════════════════════════════
class ARHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, h):
        return F.normalize(self.proj(h), dim=-1)


class ARDataset(Dataset):
    def __init__(self, records, acts_norm, tok, max_len=96):
        self.items = []
        for rec, act in zip(records, acts_norm):
            ids = tok(rec["teacher_explanation"], truncation=True,
                      max_length=max_len, return_tensors="pt")["input_ids"][0]
            self.items.append((ids, act))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def ar_collate(batch, pad_id):
    ids_list, acts = zip(*batch)
    L = max(x.shape[0] for x in ids_list)
    ids  = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = x; mask[i, :len(x)] = 1
    return ids, mask, torch.stack(acts)


def train_ar(records, acts_t, acts_n):
    dev = get_device()
    run_name = f"ar_{datetime.now().strftime('%m%d_%H%M')}"
    writer = SummaryWriter(RUNS_DIR / run_name)
    log(f"AR SFT | device={dev} | tb: tensorboard --logdir {RUNS_DIR}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    random.seed(42); idx = list(range(len(records))); random.shuffle(idx)
    val_n = max(200, int(len(idx)*0.1))
    tr_idx, va_idx = idx[val_n:], idx[:val_n]

    pad = tok.eos_token_id
    tr_dl = DataLoader(ARDataset([records[i] for i in tr_idx], acts_n[tr_idx], tok),
                       batch_size=32, shuffle=True, collate_fn=lambda b: ar_collate(b, pad))
    va_dl = DataLoader(ARDataset([records[i] for i in va_idx], acts_n[va_idx], tok),
                       batch_size=32, shuffle=False, collate_fn=lambda b: ar_collate(b, pad))

    log(f"Train {len(tr_idx)}, Val {len(va_idx)}")

    log(f"Loading {BASE_MODEL}...")
    backbone = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager").to(dev)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    head = ARHead(D_MODEL).to(dev)
    opt  = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=1e-5)

    # --- checkpoint helper: save best immediately to disk ---
    ckpt_dir = ARTIFACTS / "checkpoints" / "ar_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    def _save_ar_ckpt(state, cos_val, epoch_num):
        p = ckpt_dir / "ar_head_v2.pt"
        torch.save({"head": state, "d_model": D_MODEL,
                     "val_cosine": cos_val, "mean_baseline": mean_cos,
                     "epoch": epoch_num}, p)

    # Baselines
    mean_dir = acts_n.mean(0)
    mean_cos = (acts_n @ mean_dir).mean().item()
    log(f"Mean-direction baseline cosine: {mean_cos:.4f}")
    writer.add_scalar("baseline/mean_cosine", mean_cos, 0)

    best_cos, best_state, global_step = -1.0, None, 0
    patience, pat_cnt = 6, 0

    for epoch in range(25):
        head.train()
        ep_loss = []
        for ids, mask, gold in tr_dl:
            ids, mask, gold = ids.to(dev), mask.to(dev), gold.to(dev)
            with torch.no_grad():
                h = backbone(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                last = h[torch.arange(len(h)), mask.sum(1)-1]
            pred = head(last)
            loss = (2*(1-(pred*gold).sum(-1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss.append(loss.item())
            writer.add_scalar("ar/train_loss_step", loss.item(), global_step)
            global_step += 1

        head.eval(); cos_vals = []
        with torch.no_grad():
            for ids, mask, gold in va_dl:
                ids, mask, gold = ids.to(dev), mask.to(dev), gold.to(dev)
                h = backbone(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
                last = h[torch.arange(len(h)), mask.sum(1)-1]
                pred = head(last)
                cos_vals.extend((pred*gold).sum(-1).tolist())

        train_loss = sum(ep_loss)/len(ep_loss)
        val_cos    = sum(cos_vals)/len(cos_vals)
        sched.step()

        writer.add_scalar("ar/train_loss", train_loss, epoch)
        writer.add_scalar("ar/val_cosine", val_cos, epoch)
        writer.add_scalar("ar/lr", opt.param_groups[0]["lr"], epoch)

        delta = val_cos - mean_cos
        flag = "✓" if val_cos > best_cos else "↓"
        log(f"AR Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | val_cos={val_cos:.4f} "
            f"| Δmean={delta:+.4f} | best={best_cos:.4f} {flag}")

        if val_cos > best_cos:
            best_cos, pat_cnt = val_cos, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            _save_ar_ckpt(best_state, best_cos, epoch + 1)   # flush to disk NOW
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                log(f"Early stop (patience={patience})")
                break

    # --- safety net: ensure best state is on disk even on interrupt ---
    head.load_state_dict(best_state)
    _save_ar_ckpt(best_state, best_cos, epoch + 1)

    writer.add_hparams({"lr": 3e-4, "batch": 32, "d_model": D_MODEL},
                       {"hparam/best_val_cosine": best_cos})
    writer.close()
    log(f"AR done | best_cos={best_cos:.4f} vs mean={mean_cos:.4f} Δ={best_cos-mean_cos:+.4f}")
    log(f"Checkpoint: {ckpt_dir / 'ar_head_v2.pt'}")
    return best_cos


# ══════════════════════════════════════════════════════
# AV SFT
# ══════════════════════════════════════════════════════
class AVDataset(Dataset):
    def __init__(self, records, acts, tok, max_expl=80):
        prompt_ids = tok(f"<concept>{INJ_CHAR}</concept>\n<explanation>",
                         return_tensors="pt")["input_ids"][0]
        self.plen = len(prompt_ids)
        self.items = []
        for rec, act in zip(records, acts):
            expl_ids = tok(rec["teacher_explanation"], add_special_tokens=False,
                           return_tensors="pt")["input_ids"][0][:max_expl]
            eos = torch.tensor([tok.eos_token_id])
            ids = torch.cat([prompt_ids, expl_ids, eos])
            lbl = ids.clone(); lbl[:self.plen] = -100
            self.items.append({"input_ids": ids, "labels": lbl, "act": act,
                               "teacher": rec["teacher_explanation"]})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def av_collate(batch, pad_id):
    L = max(b["input_ids"].shape[0] for b in batch)
    ids  = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), L, dtype=torch.long)
    lbl  = torch.full((len(batch), L), -100, dtype=torch.long)
    acts = torch.stack([b["act"] for b in batch])
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        ids[i,:n] = b["input_ids"]; mask[i,:n] = 1; lbl[i,:n] = b["labels"]
    return {"ids": ids, "mask": mask, "lbl": lbl, "acts": acts}


def scale_acts(acts, dev):
    a = acts.to(dev).float()
    return a / a.norm(dim=-1, keepdim=True).clamp(1e-6) * INJ_SCALE


class AVModel(nn.Module):
    def __init__(self, lora_model):
        super().__init__(); self.model = lora_model
    def forward(self, ids, mask, lbl, acts):
        e = self.model.get_input_embeddings()(ids)
        # Vectorized activation injection (no Python loop)
        positions = (ids == INJ_TOK_ID).float().argmax(dim=1)  # [B]
        b_idx = torch.arange(ids.shape[0], device=ids.device)
        e[b_idx, positions] = acts.to(e.dtype)
        return self.model(inputs_embeds=e, attention_mask=mask, labels=lbl)


def train_av(records, acts_t, resume=False, batch_size=8, lr=5e-5, epochs=30):
    dev = get_device()
    run_name = f"av_{datetime.now().strftime('%m%d_%H%M')}"
    writer = SummaryWriter(RUNS_DIR / run_name)
    log(f"AV SFT | device={dev} | tb: tensorboard --logdir {RUNS_DIR}")

    tok = AutoTokenizer.from_pretrained(INST_MODEL, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id

    random.seed(42); idx = list(range(len(records))); random.shuffle(idx)
    val_n = max(200, int(len(idx)*0.1))
    tr_idx, va_idx = idx[val_n:], idx[:val_n]

    pad = tok.eos_token_id
    tr_ds = AVDataset([records[i] for i in tr_idx], acts_t[tr_idx], tok)
    va_ds = AVDataset([records[i] for i in va_idx], acts_t[va_idx], tok)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=lambda b: av_collate(b, pad))
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       collate_fn=lambda b: av_collate(b, pad))
    log(f"Train {len(tr_idx)}, Val {len(va_idx)}")

    log(f"Loading {INST_MODEL} (float16, sdpa)...")
    base = AutoModelForCausalLM.from_pretrained(
        INST_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(dev)
    resume_dir = ARTIFACTS / "checkpoints" / "av_v2_resume"
    av_v2_dir  = ARTIFACTS / "checkpoints" / "av_v2"
    if resume and (resume_dir / "training_state.pt").exists():
        lora_model = PeftModel.from_pretrained(base, resume_dir, is_trainable=True)
        log(f"Resumed LoRA + optimizer from {resume_dir}")
    elif resume and av_v2_dir.exists():
        lora_model = PeftModel.from_pretrained(base, av_v2_dir, is_trainable=True)
        log(f"⚠️ No resume checkpoint, loading best SFT from {av_v2_dir} (optimizer reset, epoch=0)")
    else:
        if resume:
            log(f"⚠️ No checkpoint found, starting fresh LoRA")
        lora_model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
            lora_dropout=0.1, target_modules=["q_proj","v_proj"], bias="none"))
    lora_model.print_trainable_parameters()

    av  = AVModel(lora_model).to(dev)
    opt = torch.optim.AdamW(av.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/10)

    best_loss, pat_cnt, global_step = float("inf"), 0, 0
    start_epoch = 0
    PATIENCE = 8

    if resume and (resume_dir / "training_state.pt").exists():
        ts = torch.load(resume_dir / "training_state.pt",
                        map_location=dev, weights_only=False)
        opt.load_state_dict(ts["optimizer"])
        sched.load_state_dict(ts["scheduler"])
        start_epoch = ts["epoch"] + 1
        best_loss = ts["best_loss"]
        pat_cnt = ts["pat_cnt"]
        global_step = ts["global_step"]
        log(f"Resumed from epoch {start_epoch} | best_loss={best_loss:.4f} | pat_cnt={pat_cnt}")

    for epoch in range(start_epoch, epochs):
        av.train(); ep_loss = []
        for batch in tr_dl:
            ids  = batch["ids"].to(dev); mask = batch["mask"].to(dev)
            lbl  = batch["lbl"].to(dev); acts = scale_acts(batch["acts"], dev)
            out  = av(ids, mask, lbl, acts)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(av.parameters(), 1.0)
            opt.step()
            ep_loss.append(out.loss.item())
            writer.add_scalar("av/train_loss_step", out.loss.item(), global_step)
            global_step += 1

        av.eval(); vl = []
        with torch.no_grad():
            for batch in va_dl:
                out = av(batch["ids"].to(dev), batch["mask"].to(dev),
                         batch["lbl"].to(dev), scale_acts(batch["acts"], dev))
                vl.append(out.loss.item())

        train_loss = sum(ep_loss)/len(ep_loss)
        val_loss   = sum(vl)/len(vl)
        sched.step()

        writer.add_scalar("av/train_loss", train_loss, epoch)
        writer.add_scalar("av/val_loss",   val_loss,   epoch)
        writer.add_scalar("av/lr", opt.param_groups[0]["lr"], epoch)

        # overfit alert
        gap = train_loss / max(val_loss, 1e-6)
        flag = "⚠️ overfit" if gap < 0.3 else ("✓" if val_loss < best_loss else "↓")
        log(f"AV Epoch {epoch+1:2d} | train={train_loss:.4f} | val={val_loss:.4f} "
            f"| gap={gap:.2f} | best={best_loss:.4f} {flag}")

        if val_loss < best_loss:
            best_loss, pat_cnt = val_loss, 0
            av.model.save_pretrained(ARTIFACTS / "checkpoints" / "av_v2")
            tok.save_pretrained(ARTIFACTS / "checkpoints" / "av_v2")
        else:
            pat_cnt += 1

        # Save resume checkpoint every epoch
        resume_dir = ARTIFACTS / "checkpoints" / "av_v2_resume"
        resume_dir.mkdir(parents=True, exist_ok=True)
        av.model.save_pretrained(resume_dir)
        tok.save_pretrained(resume_dir)
        torch.save({
            "epoch": epoch,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_loss": best_loss,
            "pat_cnt": pat_cnt,
            "global_step": global_step,
        }, resume_dir / "training_state.pt")

        if pat_cnt >= PATIENCE:
            log(f"Early stop (patience={PATIENCE})"); break

    # Sample generations from best checkpoint
    log("Generating samples from best checkpoint...")
    base2 = AutoModelForCausalLM.from_pretrained(
        INST_MODEL, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to(dev)
    best_av = PeftModel.from_pretrained(base2, ARTIFACTS/"checkpoints"/"av_v2").to(dev)
    best_av.eval()

    prompt  = f"<concept>{INJ_CHAR}</concept>\n<explanation>"
    p_ids   = tok(prompt, return_tensors="pt")["input_ids"].to(dev)
    p_mask  = torch.ones(1, p_ids.shape[1], device=dev)
    nonempty = 0
    with torch.no_grad():
        for i in range(min(10, len(va_ds))):
            act = va_ds[i]["act"].unsqueeze(0)
            sc  = scale_acts(act, dev)
            emb = best_av.get_input_embeddings()(p_ids)
            pos = (p_ids[0] == INJ_TOK_ID).nonzero(as_tuple=True)[0]
            if len(pos): emb[0, pos[0]] = sc[0].to(emb.dtype)
            out = best_av.generate(inputs_embeds=emb, attention_mask=p_mask,
                                   max_new_tokens=80, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0], skip_special_tokens=True).strip()
            if gen: nonempty += 1
            writer.add_text("av/samples", f"T: {va_ds[i]['teacher'][:60]}\nG: {gen[:80]}", i)
            log(f"  [{i}] T: {va_ds[i]['teacher'][:55]}")
            log(f"       G: {gen[:80]}")

    log(f"Non-empty: {nonempty}/10")
    writer.add_hparams({"lr":lr,"lora_r":8,"batch":batch_size},
                       {"hparam/best_val_loss": best_loss})
    writer.close()
    log(f"AV done | best_val_loss={best_loss:.4f}")
    log(f"Checkpoint: {ARTIFACTS/'checkpoints'/'av_v2'}")
    return best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ar","av","both"], default="both")
    parser.add_argument("--resume", action="store_true",
                        help="Resume AV training from latest checkpoint")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (smaller = more SGD noise = better generalization)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak LR (lower for warm restart from existing checkpoint)")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    records, acts_t, acts_n = load_data()

    if args.stage in ("ar","both"):
        train_ar(records, acts_t, acts_n)
    if args.stage in ("av","both"):
        train_av(records, acts_t, resume=args.resume,
                 batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)


if __name__ == "__main__":
    main()
