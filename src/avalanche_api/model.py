"""Qwen3-1.7B with the fine-tuned LoRA adapter, answering on the CPU."""

import copy

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
        # The system prompt starts every prompt: run it through the model once and keep its
        # attention keys and values, so each request only reads its own bulletin.
        empty = self.prompt("")
        self.prefix = empty[: empty.index("<|im_start|>user")]
        self.prefix_ids = self.tokenizer(self.prefix)["input_ids"]
        with torch.no_grad():
            self.prefix_cache = self.model(torch.tensor([self.prefix_ids])).past_key_values

    def prompt(self, text: str) -> str:
        """Built exactly as in training (finetune.prompt in avalanche-llm)."""
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate_batch(self, texts: list[str]) -> list[str]:
        """The raw answers for several bulletins in one forward pass per token."""
        # The prefix ends in a newline before a special token, so tokenizing the rest
        # separately gives the same tokens as tokenizing the whole prompt.
        rests = [self.prompt(text)[len(self.prefix) :] for text in texts]
        suffixes = self.tokenizer(rests, add_special_tokens=False)["input_ids"]
        # Pad between the shared prefix and each bulletin, so every row starts with the cached
        # prefix. The mask hides the padding, and position ids follow the mask, so the answers
        # are the same as without padding.
        width = max(len(s) for s in suffixes)
        pad, n = self.tokenizer.pad_token_id, len(self.prefix_ids)
        input_ids = torch.tensor([self.prefix_ids + [pad] * (width - len(s)) + s for s in suffixes])
        attention_mask = torch.tensor(
            [[1] * n + [0] * (width - len(s)) + [1] * len(s) for s in suffixes]
        )
        # Generation appends to the cache: give it a copy, one row per bulletin.
        cache = copy.deepcopy(self.prefix_cache)
        cache.batch_repeat_interleave(len(texts))
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache=cache,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        answers = output[:, input_ids.shape[1] :]
        return [a.strip() for a in self.tokenizer.batch_decode(answers, skip_special_tokens=True)]
