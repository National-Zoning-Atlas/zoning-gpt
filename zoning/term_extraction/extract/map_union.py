import warnings
import asyncio
from openai import AsyncOpenAI, OpenAI

import pdb
import diskcache as dc
import pickle
import re
import json

from ..value_ranges import get_value_ranges
from ...utils import batched, get_jinja_environment, logger
from ..thesaurus import get_thesaurus
from ..types import (
    District,
    LookupOutput,
    PageSearchOutput,
    ExtractionOutput,
    ExtractionOutput2,
)
from .base import Extractor
from .utils import include_context_around_phrase
from ...utils import flatten, logger
from ...utils import cached, get_project_root, limit_global_concurrency

from ..search.utils import page_coverage


client = AsyncOpenAI()
#client = OpenAI()
cache = dc.Cache(get_project_root() / ".diskcache")
tmpl = get_jinja_environment().get_template("extract_chat_completion.pmpt.tpl")


def get_json(text):
    if text is None:
        return None
    return json.loads(text)
    #match = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
    match = re.search(r"(\{.*?\})\n", text, re.DOTALL)
    return json.loads(match.group(1)) if match else None


def format_districts(districts):
    return "\n".join([f"* {d.full_name} ({d.short_name})" for d in districts])


@cached(cache, lambda *args, **kwargs: json.dumps(args) + json.dumps(kwargs))
async def prompt(model_name, input_prompt, max_tokens):
#def prompt(model_name, input_prompt, max_tokens):
    base_params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = await client.chat.completions.create(
        **base_params,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful expert architectural lawyer.",
            },
            {
                "role": "user",
                "content": input_prompt,
            },
        ],
        response_format={"type": "json_object"},
    )
    top_choice = resp.choices[0]  # type: ignore
    return top_choice.message.content


class MapUnionExtractor(Extractor):
    def __init__(self, model_name: str, k: int):
        self.model_name = model_name
        self.k = k

    async def extract(
    #def extract(
        self,
        pages: list[PageSearchOutput],
        district: District,
        districts: list[District],
        term: str,
        town: str,
    ):
        results = []
        for page in pages:
        #for page in pages:
            instruction = tmpl.render(
                passage=page.text,
                term=term,
                synonyms=", ".join(get_thesaurus().get(term, [])),
                zone_name=district.full_name,
                zone_abbreviation=district.short_name,
                districts=format_districts(districts),
            )
            output = await prompt(self.model_name, instruction, max_tokens=384)
            #output = prompt(self.model_name, instruction, max_tokens=384)
            r = get_json(output)
            """
            o = ExtractionOutput2(
                district_explanation=r["district_explanation"],
                district=r["district"],
                term_explanation=r["term_explanation"],
                term=r["term"],
                explanation=r["explanation"],
                answer=r["answer"],
            )
            """
            o = ExtractionOutput(
                extracted_text=[
                    "District explanation: " + r["district_explanation"],
                    "District: " + r["district"],
                    "Term explanation: " + r["term_explanation"],
                    "Term: " + r["term"],
                ],
                rationale=r["explanation"],
                answer=r["answer"],
            )
            # async
            yield LookupOutput(
                output=o,
                search_pages=[page],
                search_pages_expanded=flatten(page_coverage([page])),
            )
            # non async
            results.append(
                LookupOutput(
                    output=o,
                    search_pages=[page],
                    search_pages_expanded=flatten(page_coverage([page])),
                )
            )

        # non async
        #pdb.set_trace()
        #return results
