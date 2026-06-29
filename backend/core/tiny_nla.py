"""Tiny-NLA 引擎：LoRA + AR head 模型管理（lazy 单例）。

职责：
  - lazy 加载 Qwen/Qwen3-0.6B-Base（float32 + eager，对齐训练）+ LoRA adapter + AR head
  - extract_activation(text, token_index) → layer 19 残差流
  - explain(activation) → 注入激活 → generate → 自然语言解释
  - reconstruct_cosine(activation, explanation) → AR head 重建 → cosine

所有模型均独立加载（float32 + eager），不复用 base 槽（float16），确保与训练精度对齐。
"""

import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.models.device import DeviceManager

tiny_nla_lock = threading.Lock()
TINY_NLA_LOCK_TIMEOUT = 30.0

REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_PATH = REPO_ROOT / "experiments" / "tiny_nla" / "nla_meta.yaml"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "tiny_nla" / "checkpoints"
AV_CHECKPOINT = CHECKPOINT_DIR / "av"
AR_CHECKPOINT = CHECKPOINT_DIR / "ar" / "best_ar_head.pt"


class TinyNLAEngine:
    """LoRA + AR head 单例引擎。首次调用时 lazy 加载全部模型。"""

    _instance = None
    _init_done = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._init_done:
            return
        with open(SIDECAR_PATH, "r") as f:
            self.meta = yaml.safe_load(f)

        self.base_model_name = self.meta["base_model"]
        self.av_model_name = self.meta["av_init_model"]
        self.layer_idx = self.meta["layer_index"]
        self.d_model = self.meta["d_model"]
        self.inj_char = self.meta["tokens"]["injection_char"]
        self.inj_token_id = self.meta["tokens"]["injection_token_id"]
        self.inj_scale = self.meta["extraction"]["injection_scale"]

        self.device = DeviceManager.get_device()
        self.dtype = torch.float32

        self._base_model = None
        self._base_tokenizer = None
        self._av_model = None
        self._av_tokenizer = None
        self._ar_head = None
        self._init_done = True

    def _ensure_loaded(self):
        if self._base_model is not None:
            return

        t0 = time.perf_counter()
        print(f"  [TinyNLA] Loading base ({self.base_model_name})...")
        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self._base_model.eval()
        self._base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        DeviceManager.print_model_load_stats(self._base_model, time.perf_counter() - t0)

        t1 = time.perf_counter()
        print(f"  [TinyNLA] Loading AV base ({self.av_model_name}) + LoRA...")
        av_base = AutoModelForCausalLM.from_pretrained(
            self.av_model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self._av_model = PeftModel.from_pretrained(av_base, str(AV_CHECKPOINT))
        self._av_model.eval()
        self._av_tokenizer = AutoTokenizer.from_pretrained(
            str(AV_CHECKPOINT), trust_remote_code=True
        )
        DeviceManager.print_model_load_stats(self._av_model, time.perf_counter() - t1)

        if AR_CHECKPOINT.exists():
            print(f"  [TinyNLA] Loading AR head...")
            state = torch.load(
                str(AR_CHECKPOINT), map_location=self.device, weights_only=True
            )
            if "linear.weight" in state:
                state = {"weight": state["linear.weight"]}
            self._ar_head = torch.nn.Linear(self.d_model, self.d_model, bias=False)
            self._ar_head.load_state_dict(state)
            self._ar_head.to(self.device)
            self._ar_head.eval()
        else:
            print(f"  [TinyNLA] ⚠ AR checkpoint not found at {AR_CHECKPOINT}")
            self._ar_head = None

    def extract_activation(self, text: str, token_index: int) -> torch.Tensor:
        """用 float32 base 模型提取 layer 19 残差流。"""
        self._ensure_loaded()
        inputs = self._base_tokenizer(text, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        if token_index >= seq_len:
            raise ValueError(f"token_index {token_index} out of range (seq_len={seq_len})")
        with torch.no_grad():
            outputs = self._base_model(**inputs, output_hidden_states=True, use_cache=False)
        DeviceManager.synchronize(self.device)
        activation = outputs.hidden_states[self.layer_idx][0, token_index, :].cpu()
        return activation

    def explain(self, activation: torch.Tensor, max_new_tokens: int = 64) -> str:
        """注入激活向量 → generate → 返回 explanation 文本。"""
        self._ensure_loaded()

        prompt = f"<concept>{self.inj_char}</concept>\n<explanation>"
        inputs = self._av_tokenizer(prompt, return_tensors="pt").to(self.device)

        embeds = self._av_model.get_input_embeddings()(inputs["input_ids"])
        inj_positions = (inputs["input_ids"][0] == self.inj_token_id).nonzero(as_tuple=True)[0]

        if len(inj_positions) > 0:
            inj_pos = inj_positions[0].item()
            norm = activation.norm()
            if norm > 0:
                activation = activation / norm * self.inj_scale
            scaled_act = activation.to(embeds.dtype).to(self.device)
            embeds[0, inj_pos, :] = scaled_act

        with torch.no_grad():
            output_ids = self._av_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._av_tokenizer.pad_token_id or self._av_tokenizer.eos_token_id,
            )

        DeviceManager.synchronize(self.device)

        prompt_len_tokens = inputs["input_ids"].shape[1]
        gen_token_ids = output_ids[0][prompt_len_tokens:]
        explanation = self._av_tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip()
        return explanation

    def reconstruct_cosine(self, activation: torch.Tensor, explanation: str) -> float:
        """AR head 重建 → 计算 cosine。用 float32 base 模型提取 explanation 的 last hidden。"""
        self._ensure_loaded()
        if self._ar_head is None:
            return 0.0

        inputs = self._base_tokenizer(
            explanation, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self._base_model(**inputs, output_hidden_states=True, use_cache=False)
            last_hidden = outputs.hidden_states[-1]
            seq_len = inputs["attention_mask"].sum(dim=1) - 1
            last_token_hidden = last_hidden[0, seq_len[0], :]

            reconstructed = self._ar_head(last_token_hidden)

        DeviceManager.synchronize(self.device)

        orig_n = F.normalize(activation.unsqueeze(0).to(self.device), dim=-1)
        recon_n = F.normalize(reconstructed.unsqueeze(0), dim=-1)
        cosine = (orig_n * recon_n).sum(dim=-1).item()
        return cosine
