#!/usr/bin/env python3
"""
Round-trip evaluation: extract → explain → reconstruct → metrics
Runs on held-out data and produces a comprehensive report.
"""

import json, sys, yaml, random
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_tiny_nla import TinyNLA


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tiny_nla"


def main():
    print("=" * 60)
    print("🔄 Tiny-NLA Round-Trip Evaluation")
    print("=" * 60)
    
    nla = TinyNLA()
    
    # Load dataset for held-out samples
    with open(ARTIFACTS_DIR / "dataset.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    
    activations = torch.load(ARTIFACTS_DIR / "activations.pt", weights_only=True)
    
    # Filter to valid explanations
    valid = []
    for i, r in enumerate(records):
        exp = r.get("teacher_explanation", "") or r.get("teacher_explanation_raw", "")
        if exp and exp not in ("[空输出]", ""):
            valid.append((r, activations[i]))
    
    print(f"  Total records: {len(records)}, Valid: {len(valid)}")
    
    # Use a set of held-out indices for evaluation
    # We'll use the first N samples from different texts
    # Stratify by text to ensure diversity
    texts_grouped = {}
    for i, (r, _) in enumerate(valid):
        texts_grouped.setdefault(r["text_idx"], []).append(i)
    
    held_out = []
    for tidx, indices in texts_grouped.items():
        # Take last 2 from each text group as held-out
        held_out.extend(indices[-2:])
    
    # Make sure we have at least 20 held-out
    if len(held_out) < 20:
        extra = [i for i in range(len(valid)) if i not in held_out]
        random.Random(42).shuffle(extra)
        held_out.extend(extra[:20 - len(held_out)])
    
    print(f"  Held-out samples: {len(held_out)}")
    
    # Round-trip evaluation
    results = []
    for idx in held_out:
        rec, act = valid[idx]
        
        # We need original text + position for context
        text = rec["text"]
        pos = rec["pos"]
        token_text = rec["token_text"]
        teacher_exp = rec.get("teacher_explanation", "") or rec.get("teacher_explanation_raw", "")
        
        # Run round-trip
        try:
            # Extract activation
            ext = nla.extract(text, pos)
            
            # Generate AV explanation (from the activation, not the text)
            av_result = nla.explain(ext["activation"])
            av_explanation = av_result["explanation"]
            
            # Reconstruct from AV output
            rec_result = nla.reconstruct(av_explanation)
            
            row = {
                "text": text,
                "position": pos,
                "token_text": token_text,
                "teacher_explanation": teacher_exp[:120],
                "av_explanation": av_explanation[:120],
            }
            
            if "reconstructed" in rec_result:
                orig_n = F.normalize(act.unsqueeze(0), dim=-1)
                recon = rec_result["reconstructed"].unsqueeze(0)
                recon_n = F.normalize(recon, dim=-1)
                cosine = (orig_n * recon_n).sum(dim=-1).item()
                mse = F.mse_loss(orig_n, recon_n).item()
                row["roundtrip_cosine"] = round(cosine, 4)
                row["roundtrip_mse"] = round(mse, 6)
            
            # Also compute: teacher_explanation → original_activation cosine
            tea_result = nla.reconstruct(teacher_exp)
            if "reconstructed" in tea_result:
                tea_recon = tea_result["reconstructed"].unsqueeze(0)
                tea_recon_n = F.normalize(tea_recon, dim=-1)
                tea_cosine = (orig_n * tea_recon_n).sum(dim=-1).item()
                tea_mse = F.mse_loss(orig_n, tea_recon_n).item()
                row["teacher_to_activation_cosine"] = round(tea_cosine, 4)
                row["teacher_to_activation_mse"] = round(tea_mse, 6)
            
            results.append(row)
            
        except Exception as e:
            print(f"  ⚠️  Error at idx {idx}: {e}")
            continue
    
    # Summary statistics
    rt_cosines = [r.get("roundtrip_cosine", 0) for r in results if "roundtrip_cosine" in r]
    tea_cosines = [r.get("teacher_to_activation_cosine", 0) for r in results if "teacher_to_activation_cosine" in r]
    
    print(f"\n📊 Round-Trip Metrics")
    print(f"  Samples evaluated: {len(results)}")
    if rt_cosines:
        print(f"  Round-trip (AV→AR) cosine:")
        print(f"    Mean: {sum(rt_cosines)/len(rt_cosines):.4f}")
        print(f"    Min:  {min(rt_cosines):.4f}")
        print(f"    Max:  {max(rt_cosines):.4f}")
    if tea_cosines:
        print(f"  Teacher→Activation cosine (AR upper bound):")
        print(f"    Mean: {sum(tea_cosines)/len(tea_cosines):.4f}")
        print(f"    Min:  {min(tea_cosines):.4f}")
        print(f"    Max:  {max(tea_cosines):.4f}")
    
    # Save results
    out_path = ARTIFACTS_DIR / "roundtrip_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {out_path}")
    
    # Show 20 worked examples
    print(f"\n📝 20 Worked Examples")
    print("=" * 60)
    for i, r in enumerate(results[:20]):
        print(f"\n  [{i+1}] Text: {r['text'][:50]}...")
        print(f"      Token: {r['token_text']!r} (pos={r['position']})")
        print(f"      Teacher: {r['teacher_explanation'][:80]}")
        print(f"      AV:      {r['av_explanation'][:80]}")
        if "roundtrip_cosine" in r:
            print(f"      Round-trip cos: {r['roundtrip_cosine']}")
        if "teacher_to_activation_cosine" in r:
            print(f"      Teacher→Act cos: {r['teacher_to_activation_cosine']}")


if __name__ == "__main__":
    main()
