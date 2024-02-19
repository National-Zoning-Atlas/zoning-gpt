import json
from functools import cache
from pathlib import Path


@cache
def get_value_ranges() -> dict[str, list[str]]:
    # Global thesaurus
    with Path(__file__).parent.joinpath("value_ranges.json").open(encoding="utf-8") as f:
        return json.load(f)
