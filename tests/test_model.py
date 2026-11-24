"""The real model on the README's example bulletin: pytest -m slow (downloads 3.4 GB).

This is the check against train/serve skew: if the prompt or the loading drifts from
training, this answer changes.
"""

import pytest

from avalanche_api.schema import EXAMPLE, Bulletin


@pytest.mark.slow
def test_real_model_reads_the_example_bulletin():
    from avalanche_api.model import Extractor

    bulletin = Bulletin.model_validate_json(Extractor().generate(EXAMPLE))

    # In avalanche-llm's evaluation the 1,000-example adapter found new snow and gliding snow.
    assert {p.type for p in bulletin.problems} == {"new_snow", "gliding_snow"}
