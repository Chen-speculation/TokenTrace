#!/usr/bin/env python3
"""Generate a standalone HTML demo from real teacher labels."""
import json
from pathlib import Path
from collections import defaultdict

LABELS_FILE = Path("/Users/cccmmd/InfoLens/artifacts/tiny_nla/teacher_labels_hq.json")
OUTPUT_HTML = Path("/Users/cccmmd/InfoLens/experiments/tiny_nla/demo_activation_translator.html")

with open(LABELS_FILE) as f:
    labels = json.load(f)

by_text = defaultdict(list)
for r in labels:
    by_text[r["text_idx"]].append(r)

sentences = []
for tid in sorted(by_text):
    recs = sorted(by_text[tid], key=lambda x: x["pos"])
    sent = recs[0]["text"]
    tokens = []
    for r in recs:
        tokens.append({
            "text": r["token_text"],
            "pos": r["pos"],
            "top5": r["top_tokens"],
            "explanation": r["teacher_explanation"],
            "norm": round(r["activation_norm"], 1),
        })
    sentences.append({"text": sent, "tokens": tokens})
    if len(sentences) >= 5:
        break

sentences_json = json.dumps(sentences, ensure_ascii=False)

# Read template
import os
template_path = Path(__file__).parent / "demo_template.html"
html = template_path.read_text(encoding="utf-8")
html = html.replace("__SENTENCES_DATA__", sentences_json)
OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"Demo written to {OUTPUT_HTML}")
print(f"Sentences: {len(sentences)}, tokens: {sum(len(s['tokens']) for s in sentences)}")