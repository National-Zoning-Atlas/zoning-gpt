import warnings
import asyncio

import pickle
import re
import json

from ..value_ranges import get_value_ranges
from ...prompting import prompt
from ...utils import batched, get_jinja_environment, logger
from ..thesaurus import get_thesaurus
from ..types import (
    District,
    LookupOutput,
    PageSearchOutput,
    ExtractionOutput,
    ExtractionOutput2,
)
from .map import MapExtractor
from .utils import include_context_around_phrase
from ...utils import flatten, logger

from ...prompting import prompt
from .utils import lookup_extraction_prompt
from ..search.utils import page_coverage

from ...utils import get_jinja_environment


tmpl = get_jinja_environment().get_template("extract_chat_completion.pmpt.tpl")


def get_json(text):
    match = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
    return json.loads(match.group(1)) if match else None


def format_districts(districts):
    return "\n".join([f"* {d.full_name} ({d.short_name})" for d in districts])


class MapUnionExtractor(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    # async def extract(
    def extract(
        self,
        pages: list[PageSearchOutput],
        district: District,
        districts: list[District],
        term: str,
        town: str,
    ):
        results = []
        for page in pages:
            instruction = tmpl.render(
                passage=page.text,
                term=term,
                synonyms=", ".join(get_thesaurus().get(term, [])),
                zone_name=district.full_name,
                zone_abbreviation=district.short_name,
                districts=format_districts(districts),
            )
            r = get_json(
                asyncio.run(
                    prompt(
                        self.model_name,
                        instruction,
                        max_tokens=384,
                    )
                )
            )
            o = ExtractionOutput2(
                district_explanation=r["district_explanation"],
                district=r["district"],
                term_explanation=r["term_explanation"],
                term=r["term"],
                explanation=r["explanation"],
                answer=r["answer"],
            )
            o = ExtractionOutput(
                extracted_text=[
                    r["district_explanation"],
                    r["district"],
                    r["term_explanation"],
                    r["term"],
                ],
                rationale=r["explanation"],
                answer=r["answer"],
            )
            results.append(
                LookupOutput(
                    output=o,
                    search_pages=[page],
                    search_pages_expanded=flatten(page_coverage([page])),
                )
            )
        import pdb

        pdb.set_trace()
        return results
