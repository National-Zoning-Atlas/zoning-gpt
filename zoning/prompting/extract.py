import json
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Generator, Optional

import openai
from joblib import Memory
from jinja2 import Template
from minichain import OpenAI, prompt
from pydantic import BaseModel, ValidationError

from ..utils import chunks, get_project_root, load_jsonl
from .eval_results import clean_string_units
from .search import (
    PageSearchOutput,
    get_non_overlapping_chunks,
    nearest_pages,
    page_coverage,
)

with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

with (get_project_root() / "templates" / "extraction.pmpt.tpl").open(
    encoding="utf-8"
) as f:
    extraction_tmpl = f.read()

memory = Memory(get_project_root() / ".joblib_cache", verbose=0)


class District(BaseModel):
    T: str
    Z: str


class PromptOutput(BaseModel):
    answer: str
    extracted_text: str
    pages: list[int]


class LookupOutput(BaseModel):
    output: Optional[PromptOutput]
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """


class AllLookupOutput(BaseModel):
    town: str
    district: District
    sizes: dict[str, list[LookupOutput]]


@memory.cache
def lookup_term_prompt(page_text, district, term) -> PromptOutput | None:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": Template(extraction_tmpl).render(
                    passage=page_text,
                    term=term,
                    synonyms=" ,".join(thesaurus.get(term, [])),
                    zone_name=district["T"],
                    zone_abbreviation=district["Z"],
                ),
            },
            {
                "role": "user",
                "content": f"Input: \n\n {page_text}\n\n Output:",
            },
        ],
    )
    top_choice = resp.choices[0]
    try:
        return PromptOutput(
            **json.loads(top_choice.message.content),
        )
    except (ValidationError, TypeError):
        return None


class ExtractionMethod(str, Enum):
    NONE = "search_only"
    STUFF = "stuff"
    MAP = "map"


def extract_size(
    town, district, term, top_k_pages, method: ExtractionMethod = ExtractionMethod.STUFF
) -> list[LookupOutput]:
    pages = nearest_pages(town, district, term)
    pages = get_non_overlapping_chunks(pages)[:top_k_pages]

    if len(pages) == 0:
        return []

    match method:
        case ExtractionMethod.NONE:
            outputs = []  # if only running search
            for page in pages:
                # print(town, "page", page.page_number)
                # print(town, "pages_covered", page_coverage([page])[0])
                outputs.append(
                    LookupOutput(
                        output=None,
                        search_pages=[page],
                        search_pages_expanded=page_coverage([page])[0],
                    )
                )
            return outputs
        case ExtractionMethod.STUFF:
            # Stuff all pages into prompt, in order of page number
            all_page = reduce(
                lambda a, b: a + b.text, sorted(pages, key=lambda p: p.page_number), ""
            )
            # This is the length of the prompt before any template interpolation
            # TODO: Determine this automatically
            prompt_base_token_length = 256
            for chunk in chunks(all_page, 8192 - prompt_base_token_length):
                result: PromptOutput | None = lookup_term_prompt(chunk, district, term)
                if result is not None:
                    return [
                        LookupOutput(
                            output=result,
                            search_pages=pages,
                            search_pages_expanded=page_coverage(pages)[0],
                        )
                    ]
        case ExtractionMethod.MAP:
            outputs = []
            with ThreadPoolExecutor(max_workers=20) as executor:
                for page, result in executor.map(
                    lambda page: (page, lookup_term_prompt(page.text, district, term)),
                    pages,
                ):
                    if result is not None:
                        outputs.append(
                            LookupOutput(
                                output=result,
                                search_pages=[page],
                                search_pages_expanded=page_coverage([page])[0],
                            )
                        )
            return outputs

    return []


def extract_all_sizes(
    town_districts: list[dict], terms: list[str], top_k_pages: int
) -> Generator[AllLookupOutput, None, None]:
    for d in town_districts:
        town = d["Town"]
        districts = d["Districts"]

        for district in districts:
            yield AllLookupOutput(
                town=town,
                district=district,
                sizes={
                    term: extract_size(
                        town, district, term, top_k_pages, method=ExtractionMethod.MAP
                    )
                    for term in terms
                },
            )


def main():
    districts_file = get_project_root() / "data" / "results" / "districts_gt.jsonl"
    import pandas as pd

    gt = pd.read_csv(
        get_project_root() / "data" / "ground_truth.csv",
        index_col=["town", "district_abb"],
    )

    town_districts = load_jsonl(districts_file)

    for result in extract_all_sizes(
        town_districts, ["min lot size", "min unit size"], 6
    ):
        for term, lookups in result.sizes.items():
            for l in lookups:
                expected = set(
                    float(f)
                    for f in gt.loc[
                        result.town, result.district.Z
                    ].min_lot_size_gt.split(", ")
                )
                actual = (
                    set(clean_string_units(l.output.answer))
                    if l.output is not None
                    else set()
                )
                is_correct = any(expected & actual)
                if not is_correct:
                    print(
                        f"{result.town} - {result.district.T} ({l.output.pages}): {term} | Expected: {expected} | Actual: {actual} | Correct: {is_correct}"
                    )


if __name__ == "__main__":
    main()
