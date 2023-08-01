import json
from asyncio import gather
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Optional

import diskcache as dc
import openai
import rich
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, wait_random_exponential

from ..utils import (
    chunks,
    flatten,
    get_jinja_environment,
    get_project_root,
    limit_global_concurrency,
    cached,
)
from .search import (
    PageSearchOutput,
    get_non_overlapping_chunks,
    nearest_pages,
    page_coverage,
)
from .types import District

with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

extraction_chat_completion_tmpl = get_jinja_environment().get_template(
    "extraction_chat_completion.pmpt.tpl"
)
extraction_completion_tmpl = get_jinja_environment().get_template(
    "extraction_completion.pmpt.tpl"
)

cache = dc.Cache(get_project_root() / ".diskcache")


class PromptOutput(BaseModel):
    answer: str
    extracted_text: list[str]
    pages: list[int]
    confidence: float


class LookupOutput(BaseModel):
    output: Optional[PromptOutput]
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """


TEMPLATE_MAPPING = {
    "text-davinci-003": extraction_completion_tmpl,
    "gpt-3.5-turbo": extraction_chat_completion_tmpl,
    "gpt-4": extraction_chat_completion_tmpl,
}


@cached(cache, lambda *args: json.dumps(args))
@retry(
    retry=retry_if_exception_type(
        (
            openai.error.APIError,
            openai.error.RateLimitError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            openai.error.TryAgain,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
)
@limit_global_concurrency(100)
async def prompt(
    model_name: str, input_prompt: str | list[dict[str, str]]
) -> PromptOutput | None:
    base_params = {
        "model": model_name,
        "max_tokens": 256,
        "temperature": 0.0,
    }

    try:
        match model_name:
            case "text-davinci-003":
                resp = await openai.Completion.acreate(
                    **base_params,
                    prompt=input_prompt,
                )
                top_choice = resp.choices[0]  # type: ignore
                text = top_choice.text
            case "gpt-3.5-turbo" | "gpt-4":
                resp = await openai.ChatCompletion.acreate(
                    **base_params,
                    messages=input_prompt,
                )
                top_choice = resp.choices[0]  # type: ignore
                text = top_choice.message.content
            case _:
                raise ValueError(f"Unknown model name: {model_name}")
    except openai.error.InvalidRequestError as exc:
        rich.print("Error running extraction", exc)
        return None

    try:
        json_body = json.loads(text)
        if json_body is None:
            # The model is allowed to return null if it cannot find the answer,
            # so just pass this onwards.
            return None
        return PromptOutput(**json_body)
    except (ValidationError, TypeError, json.JSONDecodeError) as exc:
        rich.print("Error parsing response from model during extraction:", exc)
        rich.print(f"Response: {text}")
        return None


def lookup_term_prompt(
    model_name: str, page_text: str, district: District, term: str
) -> str | list[dict[str, str]]:
    match model_name:
        case "text-davinci-003":
            return TEMPLATE_MAPPING[model_name].render(
                passage=page_text,
                term=term,
                synonyms=", ".join(thesaurus.get(term, [])),
                zone_name=district.full_name,
                zone_abbreviation=district.short_name,
            )
        case "gpt-3.5-turbo" | "gpt-4":
            return [
                {
                    "role": "system",
                    "content": TEMPLATE_MAPPING[model_name].render(
                        term=term,
                        synonyms=", ".join(thesaurus.get(term, [])),
                        zone_name=district.full_name,
                        zone_abbreviation=district.short_name,
                    ),
                },
                {
                    "role": "user",
                    "content": f"Input: \n\n {page_text}\n\n Output:",
                },
            ]
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


class ExtractionMethod(str, Enum):
    NONE = "search_only"
    STUFF = "stuff"
    MAP = "map"


async def extract_answer(
    town: str,
    district: District,
    term: str,
    top_k_pages: int,
    method: ExtractionMethod = ExtractionMethod.MAP,
    model_name: str = "text-davinci-003",
) -> list[LookupOutput]:
    """
    Given a town name, a district in that town, and a term to search for, will
    attempt to extract the value for the term from the zoning document that
    corresponds to the town.
    """
    pages = nearest_pages(town, district, term)
    pages = get_non_overlapping_chunks(pages)[:top_k_pages]

    if len(pages) == 0:
        return []

    outputs = []
    match method:
        case ExtractionMethod.NONE:
            for page in pages:
                outputs.append(
                    LookupOutput(
                        output=None,
                        search_pages=[page],
                        search_pages_expanded=flatten(page_coverage([page])),
                    )
                )
        case ExtractionMethod.STUFF:
            # Stuff all pages into prompt, in order of page number
            all_page = reduce(
                lambda a, b: a + b.text, sorted(pages, key=lambda p: p.page_number), ""
            )
            # This is the length of the prompt before any template interpolation
            # TODO: Determine this automatically
            prompt_base_token_length = 256
            for chunk in chunks(all_page, 8192 - prompt_base_token_length):
                outputs.append(
                    LookupOutput(
                        output=await prompt(
                            model_name,
                            lookup_term_prompt(model_name, chunk, district, term),
                        ),
                        search_pages=pages,
                        search_pages_expanded=flatten(page_coverage(pages)),
                    )
                )
        case ExtractionMethod.MAP:
            async def worker(page):
                return (
                    page,
                    await prompt(
                        model_name,
                        lookup_term_prompt(model_name, page.text, district, term),
                    ),
                )

            for page, result in await gather(*map(worker, pages)):
                outputs.append(
                    LookupOutput(
                        output=result,
                        search_pages=[page],
                        search_pages_expanded=flatten(page_coverage([page])),
                    )
                )

    return sorted(
        outputs, key=lambda x: x.output.confidence if x.output else 0, reverse=True
    )
