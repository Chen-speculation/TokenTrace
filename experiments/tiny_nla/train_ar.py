#!/usr/bin/env python3
"""
Stage 1: AR SFT — Activation Reconstructor

Input:  explanation text (via teacher)
Output: reconstructed activation vector (d_model)

Architecture:
  - Qwen3-0.6B-Base trunk (frozen)
  - Linear(d_model, d_model) head on last token hidden state
  - L2-normalized MSE loss

Baselines:
  - Mean vector baseline (always predict mean activation)
  - Shuffled label baseline (random pairing)
"""

import json, os, time, math, yaml, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = Path(__file__).resolve().parent / "nla_meta.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "ar"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"

# ── device ─────────────────────────────────────────────
def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── AR Head ────────────────────────────────────────────
class ARHead(nn.Module):
    """Simple linear projection from d_model to d_model."""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        # x: [batch, d_model] — last token hidden state
        return self.linear(x)


# ── Dataset ────────────────────────────────────────────
class ARDataset(Dataset):
    """Explanation text → activation vector."""
    def __init__(self, records, activations, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for rec, act in zip(records, activations):
            explanation = rec.get("teacher_explanation", "") or rec.get("teacher_explanation_raw", "")
            if not explanation or explanation in ("[空输出]", ""):
                continue
            self.data.append((explanation, act))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, act = self.data[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "target": act,  # [d_model]
        }


# ── Loss ───────────────────────────────────────────────
def normalized_mse_loss(pred, target):
    """
    L2-normalize both, then MSE = 2*(1-cosine)
    Equivalent to ||normalized_pred - normalized_target||²
    """
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return F.mse_loss(pred_n, target_n)


def cosine_similarity(pred, target):
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (pred_n * target_n).sum(dim=-1)


# ── Training ───────────────────────────────────────────
def train():
    print("=" * 60)
    print("🔧 AR SFT — Activation Reconstructor Training")
    print("=" * 60)
    
    device = detect_device()
    dtype = torch.float32
    print(f"  Device: {device}, dtype: {dtype}")
    
    # Load sidecar
    with open(SIDECAR_PATH, "r") as f:
        nla_meta = yaml.safe_load(f)
    d_model = nla_meta["d_model"]
    print(f"  d_model: {d_model}")
    
    # Load dataset
    print("\n  Loading dataset...")
    jsonl_path = ARTIFACTS_DIR / "dataset.jsonl"
    act_path = ARTIFACTS_DIR / "activations.pt"
    
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    activations = torch.load(act_path, weights_only=True)
    print(f"  Loaded {len(records)} records, activations shape: {activations.shape}")
    
    # Train/val split
    random.seed(42)
    indices = list(range(len(records)))
    random.shuffle(indices)
    val_size = max(1, int(len(indices) * 0.15))
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_records = [records[i] for i in train_idx]
    train_acts = activations[train_idx]
    val_records = [records[i] for i in val_idx]
    val_acts = activations[val_idx]
    
    train_dataset = ARDataset(train_records, train_acts, tokenizer)
    val_dataset = ARDataset(val_records, val_acts, tokenizer)
    print(f"  Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Load base model (frozen)
    print("\n  Loading base model (frozen)...")
    t0 = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print(f"  Base model loaded in {time.perf_counter() - t0:.1f}s")
    
    # AR head
    ar_head = ARHead(d_model).to(device)
    print(f"  AR head params: {sum(p.numel() for p in ar_head.parameters()):,}")
    
    optimizer = torch.optim.AdamW(ar_head.parameters(), lr=1e-3)
    
    # ── Compute baselines ──
    print("\n📊 Computing baselines...")
    
    # Mean vector baseline
    all_train_acts = torch.stack([train_dataset[i]["target"] for i in range(len(train_dataset))])
    mean_vec = all_train_acts.mean(dim=0)  # [d_model]
    
    mean_cosines = []
    for i in range(len(val_dataset)):
        target = val_dataset[i]["target"]
        cos = F.cosine_similarity(mean_vec.unsqueeze(0), target.unsqueeze(0))
        mean_cosines.append(cos.item())
    mean_baseline = torch.tensor(mean_cosines).mean().item()
    print(f"  Mean-vector baseline cosine: {mean_baseline:.4f}")
    
    # Shuffled baseline
    val_targets = torch.stack([val_dataset[i]["target"] for i in range(len(val_dataset))])
    shuffled = val_targets[torch.randperm(len(val_targets))]
    shuffled_cosines = F.cosine_similarity(val_targets, shuffled, dim=-1)
    shuffled_baseline = shuffled_cosines.mean().item()
    print(f"  Shuffled baseline cosine:      {shuffled_baseline:.4f}")
    
    # ── Training loop ──
    print("\n🏋️  Training AR head...")
    n_epochs = 50
    best_val_loss = float("inf")
    best_val_cosine = 0.0
    
    for epoch in range(n_epochs):
        # Train
        ar_head.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Get last token hidden state from last layer
                last_hidden = outputs.hidden_states[-1]  # [batch, seq, d_model]
                # Use the last non-pad token position
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(last_hidden.size(0), device=device)
                last_token_hidden = last_hidden[batch_indices, seq_lens, :]  # [batch, d_model]
            
            pred = ar_head(last_token_hidden)
            loss = normalized_mse_loss(pred, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_loss / n_batches
        
        # Validation
        ar_head.eval()
        val_losses = []
        val_cosines = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)
                
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden = outputs.hidden_states[-1]
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(last_hidden.size(0), device=device)
                last_token_hidden = last_hidden[batch_indices, seq_lens, :]
                
                pred = ar_head(last_token_hidden)
                loss = normalized_mse_loss(pred, targets)
                cos = cosine_similarity(pred, targets)
                
                val_losses.append(loss.item())
                val_cosines.extend(cos.cpu().tolist())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_cosine = sum(val_cosines) / len(val_cosines)
        
        if avg_val_cosine > best_val_cosine:
            best_val_cosine = avg_val_cosine
            best_val_loss = avg_val_loss
            torch.save(ar_head.state_dict(), CHECKPOINT_DIR / "best_ar_head.pt")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{n_epochs} | train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | val_cos={avg_val_cosine:.4f} | best_cos={best_val_cosine:.4f}")
    
    # ── Final evaluation ──
    print(f"\n📈 Final Results")
    print(f"  {'':30} {'Loss':>10} {'Cosine':>10}")
    print(f"  {'Mean-vector baseline':30} {'':>10} {mean_baseline:>10.4f}")
    print(f"  {'Shuffled baseline':30} {'':>10} {shuffled_baseline:>10.4f}")
    print(f"  {'AR trained (best)':30} {best_val_loss:>10.6f} {best_val_cosine:>10.4f}")
    
    improvement = best_val_cosine - max(mean_baseline, shuffled_baseline)
    print(f"\n  Improvement over best baseline: {improvement:+.4f}")
    
    if improvement <= 0:
        print("  ⚠️  AR does NOT beat baselines! May need fixes.")
    
    print(f"\n  Checkpoint: {CHECKPOINT_DIR / 'best_ar_head.pt'}")
    
    return best_val_cosine, best_val_loss


if __name__ == "__main__":
    train()
