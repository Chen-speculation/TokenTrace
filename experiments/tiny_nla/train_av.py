#!/usr/bin/env python3
"""
Stage 2: AV SFT — Activation Verbalizer

Fixes vs prior version:
  1. No padding='max_length' — pad to batch max length with attention_mask
  2. Pass attention_mask in both train forward and generate
  3. Generate with prompt-only embeds (not full padded sequence)
  4. Use attention_mask in AVModel.forward
"""

import json, os, time, yaml, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = Path(__file__).resolve().parent / "nla_meta.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "av"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUCT_MODEL = "Qwen/Qwen3-0.6B"


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch, pad_id):
    """Pad to batch max length (not global max)."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    activations = torch.stack([b["activation"] for b in batch])
    prompt_lens = [b["prompt_len"] for b in batch]

    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = 1
        labels[i, :n] = b["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "activations": activations,
        "prompt_lens": prompt_lens,
    }


class AVDataset(Dataset):
    def __init__(self, av_data, activations, tokenizer, inj_char, inj_token_id, inj_scale, max_length=96):
        self.tokenizer = tokenizer
        self.inj_token_id = inj_token_id
        self.inj_scale = inj_scale
        self.max_length = max_length
        self.inj_char = inj_char
        self.data = []

        for rec, act in zip(av_data, activations):
            expl = rec.get("teacher_explanation", "") or ""
            if not expl.strip() or expl in ("[空输出]",):
                continue
            self.data.append((expl, act))

        print(f"  AVDataset: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        explanation, activation = self.data[idx]
        prompt = f"<concept>{self.inj_char}</concept>\n<explanation>"

        # Tokenize prompt and full sequence separately (no padding here)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        expl_ids = self.tokenizer(
            explanation,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        # Truncate explanation if needed
        max_expl = self.max_length - len(prompt_ids) - 1  # -1 for eos
        expl_ids = expl_ids[:max_expl]

        # Build full sequence: prompt + explanation + eos
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, expl_ids, eos])

        # Labels: -100 for prompt, actual ids for explanation+eos
        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_ids):] = torch.cat([expl_ids, eos])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "activation": activation,
            "prompt_len": len(prompt_ids),
        }


class AVModel(nn.Module):
    def __init__(self, base_model, inj_token_id):
        super().__init__()
        self.model = base_model
        self.inj_token_id = inj_token_id

    def forward(self, input_ids, attention_mask, labels, activations):
        embeds = self.model.get_input_embeddings()(input_ids)

        for b in range(input_ids.shape[0]):
            positions = (input_ids[b] == self.inj_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                embeds[b, positions[0].item(), :] = activations[b].to(embeds.dtype)

        return self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )


def scale_activations(acts, scale, device):
    acts = acts.to(device)
    norms = acts.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return acts / norms * scale


def train():
    print("=" * 60)
    print("AV SFT — Activation Verbalizer Training")
    print("=" * 60)

    device = detect_device()
    dtype = torch.float32
    print(f"  Device: {device}, dtype: {dtype}")

    with open(SIDECAR_PATH) as f:
        meta = yaml.safe_load(f)

    inj_char = meta["tokens"]["injection_char"]
    inj_token_id = meta["tokens"]["injection_token_id"]
    inj_scale = meta["extraction"]["injection_scale"]
    print(f"  inj_char={inj_char!r} id={inj_token_id} scale={inj_scale}")

    with open(ARTIFACTS_DIR / "av_training_data.json", encoding="utf-8") as f:
        av_data = json.load(f)
    activations = torch.load(ARTIFACTS_DIR / "av_activations.pt", weights_only=True)
    print(f"  Loaded {len(av_data)} records, activations {activations.shape}")

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=True)
    # Use a different pad token to avoid pad=eos issue
    tokenizer.pad_token_id = tokenizer.eos_token_id  # needed for generate only

    random.seed(42)
    idx = list(range(len(av_data)))
    random.shuffle(idx)
    val_n = max(20, int(len(idx) * 0.15))
    train_idx, val_idx = idx[val_n:], idx[:val_n]

    def mk_dataset(indices):
        return AVDataset(
            [av_data[i] for i in indices],
            activations[indices],
            tokenizer, inj_char, inj_token_id, inj_scale,
        )

    train_ds = mk_dataset(train_idx)
    val_ds = mk_dataset(val_idx)

    pad_id = tokenizer.eos_token_id
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_id))

    print(f"\n  Loading {INSTRUCT_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(device)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()

    av_model = AVModel(lora_model, inj_token_id).to(device)
    optimizer = torch.optim.AdamW(av_model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    n_epochs = 50
    best_val_loss = float("inf")
    patience, patience_count = 10, 0

    for epoch in range(n_epochs):
        av_model.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            scaled_acts = scale_activations(batch["activations"].float(), inj_scale, device)

            out = av_model(input_ids, attention_mask, labels, scaled_acts)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(av_model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        av_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                out = av_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                    scale_activations(batch["activations"].float(), inj_scale, device),
                )
                val_losses.append(out.loss.item())

        tl = sum(train_losses) / len(train_losses)
        vl = sum(val_losses) / len(val_losses)

        if vl < best_val_loss:
            best_val_loss = vl
            patience_count = 0
            av_model.model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
        else:
            patience_count += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{n_epochs} | train={tl:.4f} | val={vl:.4f} | best={best_val_loss:.4f}")

        if patience_count >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    print(f"\n  Best val loss: {best_val_loss:.4f}")

    # ── Generate examples (prompt-only embeds, no padding) ──
    print("\n  Generating held-out examples...")
    av_model.eval()
    examples = []

    # Reload best checkpoint
    from peft import PeftModel
    best_base = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL, trust_remote_code=True, dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(device)
    best_model = PeftModel.from_pretrained(best_base, CHECKPOINT_DIR).to(device)
    best_model.eval()

    prompt_template = f"<concept>{inj_char}</concept>\n<explanation>"

    with torch.no_grad():
        for i in range(min(25, len(val_ds))):
            item = val_ds[i]
            act = item["activation"].unsqueeze(0).float()
            scaled = scale_activations(act, inj_scale, device)

            # Tokenize prompt only (no padding)
            p_ids = tokenizer(prompt_template, return_tensors="pt")["input_ids"].to(device)
            p_mask = torch.ones_like(p_ids)

            embeds = best_model.get_input_embeddings()(p_ids)
            inj_pos = (p_ids[0] == inj_token_id).nonzero(as_tuple=True)[0]
            if len(inj_pos) > 0:
                embeds[0, inj_pos[0].item(), :] = scaled[0].to(embeds.dtype)

            out_ids = best_model.generate(
                inputs_embeds=embeds,
                attention_mask=p_mask,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            # output_ids when using inputs_embeds: only generated tokens
            gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            teacher = tokenizer.decode(
                item["labels"][item["prompt_len"]:][item["labels"][item["prompt_len"]:] != -100],
                skip_special_tokens=True,
            )
            examples.append({
                "index": i,
                "teacher": teacher,
                "av_generated": gen_text,
            })

    nonempty = sum(1 for e in examples if e["av_generated"].strip())
    print(f"  Non-empty: {nonempty}/{len(examples)}")
    for e in examples[:5]:
        print(f"    T: {e['teacher'][:60]}")
        print(f"    G: {e['av_generated'][:80]}")
        print()

    out_path = ARTIFACTS_DIR / "av_examples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {out_path}")
    print(f"\n  LoRA adapter: {CHECKPOINT_DIR}")
    return best_val_loss


if __name__ == "__main__":
    train()
