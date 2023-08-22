import json
from functools import cache
from pathlib import Path


@cache
def get_thesaurus() -> dict[str, list[str]]:
    # Global thesaurus
    with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
        return json.load(f)
