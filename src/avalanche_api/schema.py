"""The API's contract: what a request holds and what the model must answer.

Bulletin and Problem are copied from avalanche-llm (schema.py), the schema the adapter was
trained on. Copied, not imported: the research repo would pull in its whole toolchain.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProblemType = Literal[
    "new_snow",
    "wind_slab",
    "persistent_weak_layers",
    "wet_snow",
    "gliding_snow",
    "no_distinct_avalanche_problem",
]
Aspect = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class Problem(BaseModel):
    type: ProblemType
    danger_level: int = Field(ge=1, le=5, description="1 low, 2 moderate, ... 5 very high")
    danger_modifier: Literal["+", "-"] | None = Field(
        None, description="Swiss subdivision of the level, e.g. 2+ or 3-; null if none"
    )
    aspects: list[Aspect] = Field([], description="Slope aspects where the problem occurs")
    elevation_lower: int | None = Field(None, description="Occurs above this elevation (m)")
    elevation_upper: int | None = Field(None, description="Occurs below this elevation (m)")


class Bulletin(BaseModel):
    problems: list[Problem]


# A bulletin from 27 November 2025 that no model saw in training: Swagger UI's "Try it out".
EXAMPLE = (
    "A lot of snow has fallen since Sunday. The new snow is in some cases still prone to "
    "triggering.\nAs a consequence of bise wind, mostly small wind slabs formed on Thursday. "
    "These are to be evaluated with care and prudence in particular in very steep terrain.\n"
    "Careful route selection is recommended.\n\nGliding avalanches are possible on steep "
    "grassy slopes. These can in isolated cases reach medium size."
)


class ExtractRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": EXAMPLE}]})

    # The upper bound caps the compute one request can take: generation time grows with it.
    text: str = Field(
        min_length=1,
        max_length=10_000,
        description="The avalanche problem descriptions of one SLF bulletin (de, fr, it or en)",
    )


class ModelOutputError(BaseModel):
    detail: str
    output: str = Field(description="The model's raw answer")
