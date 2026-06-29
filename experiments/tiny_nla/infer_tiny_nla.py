#!/usr/bin/env python3
"""
Tiny-NLA Inference Script

Reads nla_meta.yaml and provides:
  extract(text, positions) → activation vectors at layer 19
  explain(activation) → AV-generated explanation
  reconstruct(explanation) → AR-reconstructed activation + cosine/MSE

Usage:
  python infer_tiny_nla.py --text "你的文本" --position 5
  python infer_tiny_nla.py --interactive
"""

import argparse, json, os, sys, yaml, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── paths ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = Path(__file__).resolve().parent / "nla_meta.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "tiny_nla" / "checkpoints"
AV_CHECKPOINT = CHECKPOINT_DIR / "av"
AR_CHECKPOINT = CHECKPOINT_DIR / "ar" / "best_ar_head.pt"


class TinyNLA:
    """
    Tiny-NLA inference wrapper.
    Loads models lazily to avoid memory waste.
    """
    
    def __init__(self, sidecar_path=SIDECAR_PATH):
        with open(sidecar_path, "r") as f:
            self.meta = yaml.safe_load(f)
        
        self.base_model_name = self.meta["base_model"]
        self.av_model_name = self.meta["av_init_model"]
        self.layer_idx = self.meta["layer_index"]
        self.d_model = self.meta["d_model"]
        self.inj_char = self.meta["tokens"]["injection_char"]
        self.inj_token_id = self.meta["tokens"]["injection_token_id"]
        self.inj_scale = self.meta["extraction"]["injection_scale"]
        
        self.device = self._detect_device()
        self.dtype = torch.float32
        
        # Lazy-loaded models
        self.base_model = None
        self.base_tokenizer = None
        self.av_model = None
        self.av_tokenizer = None
        self.ar_head = None
        self.ar_loaded = False
    
    def _detect_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _load_base(self):
        if self.base_model is not None:
            return
        print(f"  Loading base model ({self.base_model_name})...")
        t0 = time.perf_counter()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self.base_model.eval()
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        print(f"    Done in {time.perf_counter() - t0:.1f}s")
    
    def _load_av(self):
        if self.av_model is not None:
            return
        print(f"  Loading AV model ({self.av_model_name} + LoRA)...")
        t0 = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(
            self.av_model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self.av_model = PeftModel.from_pretrained(base, AV_CHECKPOINT)
        self.av_model.eval()
        self.av_tokenizer = AutoTokenizer.from_pretrained(
            self.av_model_name, trust_remote_code=True
        )
        print(f"    Done in {time.perf_counter() - t0:.1f}s")
    
    def _load_ar(self):
        if self.ar_loaded:
            return
        if not AR_CHECKPOINT.exists():
            print(f"  ⚠️  AR checkpoint not found at {AR_CHECKPOINT}")
            self.ar_loaded = False
            return
        print(f"  Loading AR head...")
        state = torch.load(AR_CHECKPOINT, map_location=self.device, weights_only=True)
        # ARHead saved as {'linear.weight': ...} but we want {'weight': ...}
        if "linear.weight" in state:
            state = {"weight": state["linear.weight"]}
        self.ar_head = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.ar_head.load_state_dict(state)
        self.ar_head.to(self.device)
        self.ar_head.eval()
        self.ar_loaded = True
        print(f"    Done")
    
    def extract(self, text: str, position: int = -1):
        """
        Extract activation at given layer for the specified token position.
        If position < 0, uses last token.
        Returns: (activation_vector [d_model], input_ids, top_tokens)
        """
        self._load_base()
        
        inputs = self.base_tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        
        if position < 0 or position >= seq_len:
            position = seq_len - 1
        
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
            )
        
        hidden = outputs.hidden_states[self.layer_idx]  # [1, seq, d_model]
        activation = hidden[0, position, :].cpu()  # [d_model]
        
        # Top-k from logits at this position
        logits = outputs.logits[0, position, :]
        topk_vals, topk_idxs = torch.topk(logits, k=10)
        top_tokens = [self.base_tokenizer.decode([t]) for t in topk_idxs]
        
        return {
            "activation": activation,
            "position": position,
            "seq_len": seq_len,
            "token_text": self.base_tokenizer.decode([inputs["input_ids"][0, position].item()]),
            "top_tokens": top_tokens,
            "activation_norm": activation.norm().item(),
        }
    
    def explain(self, activation: torch.Tensor, max_new_tokens=64):
        """Generate AV explanation from activation vector."""
        self._load_av()
        
        prompt = f"<concept>{self.inj_char}</concept>\n<explanation>"
        inputs = self.av_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Inject activation at injection token position
        embeds = self.av_model.get_input_embeddings()(inputs["input_ids"])
        inj_positions = (inputs["input_ids"][0] == self.inj_token_id).nonzero(as_tuple=True)[0]
        
        if len(inj_positions) > 0:
            inj_pos = inj_positions[0].item()
            # Normalize activation to injection_scale norm
            norm = activation.norm()
            if norm > 0:
                activation = activation / norm * self.inj_scale
            scaled_act = activation.to(embeds.dtype).to(self.device)
            embeds[0, inj_pos, :] = scaled_act
        
        with torch.no_grad():
            output_ids = self.av_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.av_tokenizer.pad_token_id or self.av_tokenizer.eos_token_id,
            )
        
        # Extract just the generated part (token-level, not string-level)
        prompt_len_tokens = inputs["input_ids"].shape[1]
        gen_token_ids = output_ids[0][prompt_len_tokens:]
        explanation = self.av_tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        
        return {
            "explanation": explanation,
        }
    
    def reconstruct(self, explanation: str):
        """AR: explanation text → reconstructed activation + metrics."""
        self._load_ar()
        self._load_base()
        
        if not self.ar_loaded:
            return {"error": "AR checkpoint not available"}
        
        # Tokenize explanation with base model's tokenizer
        inputs = self.base_tokenizer(
            explanation,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1]
            seq_len = inputs["attention_mask"].sum(dim=1) - 1
            last_token_hidden = last_hidden[0, seq_len[0], :]
            
            reconstructed = self.ar_head(last_token_hidden)
        
        return {"reconstructed": reconstructed.cpu()}
    
    def roundtrip(self, text: str, position: int = -1):
        """Full round-trip: extract → explain → reconstruct."""
        # Extract
        ext = self.extract(text, position)
        activation = ext["activation"]
        
        # Explain
        expl = self.explain(activation)
        
        # Reconstruct
        rec = self.reconstruct(expl["explanation"])
        
        result = {
            "text": text,
            "position": ext["position"],
            "token_text": ext["token_text"],
            "activation_norm": ext["activation_norm"],
            "top_tokens": ext["top_tokens"],
            "av_explanation": expl["explanation"],
        }
        
        if "reconstructed" in rec:
            orig_n = F.normalize(activation.unsqueeze(0), dim=-1)
            recon = rec["reconstructed"].unsqueeze(0)
            recon_n = F.normalize(recon, dim=-1)
            cosine = (orig_n * recon_n).sum(dim=-1).item()
            mse = F.mse_loss(orig_n, recon_n).item()
            
            result["ar_cosine"] = round(cosine, 4)
            result["ar_normalized_mse"] = round(mse, 6)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Tiny-NLA Inference")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--position", type=int, default=-1, help="Token position (-1 = last)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", type=str, help="JSON file with list of {text, position}")
    args = parser.parse_args()
    
    nla = TinyNLA()
    
    if args.interactive:
        print("Tiny-NLA Interactive Mode (Ctrl+D to exit)\n")
        while True:
            try:
                text = input("Text: ").strip()
                if not text:
                    continue
                pos_input = input("Position (default=last): ").strip()
                pos = int(pos_input) if pos_input else -1
                
                print("\n  Processing...")
                result = nla.roundtrip(text, pos)
                
                print(f"\n  Token:       {result['token_text']!r}")
                print(f"  Position:    {result['position']}")
                print(f"  Top tokens:  {result['top_tokens'][:5]}")
                print(f"  AV:          {result['av_explanation'][:120]}")
                if "ar_cosine" in result:
                    print(f"  AR cosine:   {result['ar_cosine']}")
                print()
            except EOFError:
                break
    
    elif args.text:
        result = nla.roundtrip(args.text, args.position)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            batch = json.load(f)
        results = []
        for item in batch:
            r = nla.roundtrip(item["text"], item.get("position", -1))
            results.append(r)
            print(f"  [{len(results)}/{len(batch)}] {r['token_text']!r} -> cosine={r.get('ar_cosine', 'N/A')}")
        
        out_path = Path(args.batch).parent / "roundtrip_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to {out_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
