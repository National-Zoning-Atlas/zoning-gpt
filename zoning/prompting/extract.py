from enum import Enum
from functools import reduce
import json
from pathlib import Path
from typing import Generator

from minichain import OpenAI, prompt
from pydantic import BaseModel

from ..utils import get_project_root, load_jsonl, chunks
from .search import nearest_pages, get_non_overlapping_chunks, PageSearchOutput

with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

with (get_project_root() / "templates" / "extraction.pmpt.tpl").open(encoding="utf-8") as f:
    extraction_tmpl = f.read()


class District(BaseModel):
    T: str
    Z: str


class PromptOutput(BaseModel):
    answer: str
    extracted_text: str
    pages: list[int]


class LookupOutput(BaseModel):
    output: PromptOutput
    search_pages: list[PageSearchOutput]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """


class AllLookupOutput(BaseModel):
    town: str
    district: District
    sizes: dict[str, list[LookupOutput]]




# TODO: minichain appears to currently ignore the model argument. We should fix
# this and enable it to use GPT-4.
@prompt(OpenAI(model="text-davcinci-003"), template=extraction_tmpl, parser="json")
def lookup_term_prompt(model, page_text, district, term) -> PromptOutput | None:
    return model(
        dict(
            passage=page_text,
            term=term,
            synonyms=" ,".join(thesaurus.get(term, [])),
            zone_name=district["T"],
            zone_abbreviation=district["Z"],
        )
    )

class ExtractionMethod(str, Enum):
    STUFF = "stuff"
    MAP = "map"

def extract_size(town, district, term, top_k_pages, method: ExtractionMethod = ExtractionMethod.STUFF) -> list[LookupOutput]:
    pages = nearest_pages(town, district, term)[:top_k_pages]
    pages = get_non_overlapping_chunks(pages)

    if len(pages) == 0:
        return []

    match method:
        case ExtractionMethod.STUFF:
            # Stuff all pages into prompt, in order of page number
            all_page = reduce(
                lambda a, b: a + b.text, sorted(pages, key=lambda p: p.page_number), ""
            )

            # This is the length of the prompt before any template interpolation
            # TODO: Determine this automatically
            prompt_base_token_length = 256
            for chunk in chunks(all_page, 2047 - prompt_base_token_length):
                result: PromptOutput | None = lookup_term_prompt(chunk, district, term).run()  # type: ignore
                if result is not None:
                    return [LookupOutput(
                        output=result,
                        search_pages=pages,
                    )]
        case ExtractionMethod.MAP:
            outputs = []
            for page in pages:
                result: PromptOutput | None = lookup_term_prompt(page.text, district, term).run()  # type: ignore
                if result is not None:
                    outputs.append(LookupOutput(
                        output=result,
                        search_pages=[page],
                    ))
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
                    term: extract_size(town, district, term, top_k_pages, method=ExtractionMethod.MAP)
                    for term in terms
                },
            )


def main():
    districts_file = (
        get_project_root() / "data" / "results" / "districts_matched_2.jsonl"
    )

    town_districts = load_jsonl(districts_file)
    for result in extract_all_sizes(
        town_districts, ["min lot size", "min unit size"], 6
    ):
        for term, lookups in result.sizes.items():
            for l in lookups:
                print(
                    f"{result.town} - {result.district.T}: {term} is {l.output.answer if l.output is not None else 'not found'}"
                )


if __name__ == "__main__":
    main()
