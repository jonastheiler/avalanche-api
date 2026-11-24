"""Qwen3-1.7B with the fine-tuned LoRA adapter, answering on the CPU."""

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from avalanche_api.prompt import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER = "jth00/qwen3-1.7b-avalanche-lora"
# Pinned Hub commits: a retrained adapter pushed later doesn't silently change the answers.
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ADAPTER_REVISION = "ced385b5b585ceb02cf072ee4e4a95c7d809b057"
MAX_NEW_TOKENS = 512


def download() -> None:
    """Fetch both models into the Hugging Face cache, e.g. while building the Docker image."""
    snapshot_download(BASE_MODEL, revision=BASE_REVISION)
    snapshot_download(ADAPTER, revision=ADAPTER_REVISION)


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
        # Generation continues at the right end of each row, so pad shorter prompts on the left.
        self.tokenizer.padding_side = "left"

    def generate_batch(self, texts: list[str]) -> list[str]:
        """The raw answers for several bulletins in one forward pass per token. The prompt is
        built exactly as in training (finetune.prompt in avalanche-llm)."""
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in texts
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        answers = output[:, inputs["input_ids"].shape[1] :]
        return [a.strip() for a in self.tokenizer.batch_decode(answers, skip_special_tokens=True)]
