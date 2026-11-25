"""The real model on the README's example bulletin: pytest -m slow (downloads 3.4 GB).

This is the check against train/serve skew: if the prompt or the loading drifts from
training, this answer changes.
"""

import pytest
import torch

from avalanche_api.schema import EXAMPLE, Bulletin


@pytest.mark.slow
def test_real_model_reads_the_example_bulletin_alone_and_in_a_batch():
    from avalanche_api.model import MAX_NEW_TOKENS, Extractor

    model = Extractor()
    # The plain way, without the cached system prompt: the whole prompt as in training.
    inputs = model.tokenizer(model.prompt(EXAMPLE), return_tensors="pt")
    with torch.no_grad():
        output = model.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    plain = model.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    alone = model.generate_batch([EXAMPLE])[0]
    in_batch = model.generate_batch([EXAMPLE, "Gliding avalanches are possible."])[0]

    # In avalanche-llm's evaluation the 1,000-example adapter found new snow and gliding snow.
    problems = Bulletin.model_validate_json(alone).problems
    assert {p.type for p in problems} == {"new_snow", "gliding_snow"}
    # The cached prefix and the padding must not change the answer: they are optimisations,
    # not a new model.
    assert alone == plain.strip()
    assert in_batch == alone
