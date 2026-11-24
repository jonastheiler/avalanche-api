"""Qwen3-1.7B with the fine-tuned LoRA adapter, answering on the CPU."""

import threading

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from avalanche_api.prompt import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER = "jth00/qwen3-1.7b-avalanche-lora"
# Pinned Hub commits: a retrained adapter pushed later doesn't silently change the answers.
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ADAPTER_REVISION = "ced385b5b585ceb02cf072ee4e4a95c7d809b057"
MAX_NEW_TOKENS = 512


class Extractor:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        # float32: most CPUs have no fast float16 arithmetic. 1.7B parameters take ~7 GB.
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32
        )
        # Add the low-rank LoRA update into the base weights once: afterwards it is one plain
        # model, with no extra adapter matrix multiplications per token.
        adapted = PeftModel.from_pretrained(base, ADAPTER, revision=ADAPTER_REVISION)
        self.model = adapted.merge_and_unload().eval()
        # One generation at a time: torch already uses every core for a single one.
        self.lock = threading.Lock()

    def generate(self, text: str) -> str:
        """The raw answer for one bulletin, built exactly as in training (finetune.prompt)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.lock, torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        answer = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()
