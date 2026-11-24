"""The real model on the README's example bulletin: pytest -m slow (downloads 3.4 GB).

This is the check against train/serve skew: if the prompt or the loading drifts from
training, this answer changes.
"""

import pytest

from avalanche_api.schema import EXAMPLE, Bulletin


@pytest.mark.slow
def test_real_model_reads_the_example_bulletin_alone_and_in_a_batch():
    from avalanche_api.model import Extractor

    model = Extractor()
    alone = model.generate_batch([EXAMPLE])[0]
    in_batch = model.generate_batch([EXAMPLE, "Gliding avalanches are possible."])[0]

    # In avalanche-llm's evaluation the 1,000-example adapter found new snow and gliding snow.
    problems = Bulletin.model_validate_json(alone).problems
    assert {p.type for p in problems} == {"new_snow", "gliding_snow"}
    # Left padding must not change the answer: batching is an optimisation, not a new model.
    assert in_batch == alone
