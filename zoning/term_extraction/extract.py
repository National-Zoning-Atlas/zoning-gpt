import json
from asyncio import gather
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Optional, AsyncGenerator
import warnings

import diskcache as dc
import openai
import rich
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, wait_random_exponential
import tiktoken

from ..utils import (
    batched,
    flatten,
    get_jinja_environment,
    get_project_root,
    limit_global_concurrency,
    cached,
)
from .search import (
    PageSearchOutput,
    page_coverage,
)
from .types import District

with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE = 4
TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 1000


extraction_chat_completion_tmpl = get_jinja_environment().get_template(
    "extraction_chat_completion.pmpt.tpl"
)
extraction_completion_tmpl = get_jinja_environment().get_template(
    "extraction_completion.pmpt.tpl"
)
tournament_reduce_tmpl = get_jinja_environment().get_template("tournament.pmpt.tpl")

cache = dc.Cache(get_project_root() / ".diskcache")


class ExtractionOutput(BaseModel):
    extracted_text: list[str]
    rationale: str
    answer: str


class LookupOutput(BaseModel):
    output: Optional[ExtractionOutput]
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """


class ExtractionMethod(str, Enum):
    NONE = "search_only"
    STUFF = "stuff"
    MAP = "map"
    TOURNAMENT_REDUCE = "tournament_reduce"


TEMPLATE_MAPPING = {
    "text-davinci-003": extraction_completion_tmpl,
    "gpt-3.5-turbo": extraction_chat_completion_tmpl,
    "gpt-4": extraction_chat_completion_tmpl,
}


@cached(cache, lambda *args, **kwargs: json.dumps(args) + json.dumps(kwargs))
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
    model_name: str,
    input_prompt: str | list[dict[str, str]],
    max_tokens=256,
) -> str | None:
    base_params = {
        "model": model_name,
        "max_tokens": max_tokens,
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
                return top_choice.text
            case "gpt-3.5-turbo" | "gpt-4":
                resp = await openai.ChatCompletion.acreate(
                    **base_params,
                    messages=input_prompt,
                )
                top_choice = resp.choices[0]  # type: ignore
                return top_choice.message.content
            case _:
                raise ValueError(f"Unknown model name: {model_name}")
    except openai.error.InvalidRequestError as exc:
        rich.print("Error running prompt", exc)
        return None


def parse_extraction_output(text: str | None) -> ExtractionOutput | None:
    if text is None:
        return None

    try:
        json_body = json.loads(text)
        if json_body is None:
            # The model is allowed to return null if it cannot find the answer,
            # so just pass this onwards.
            return None
        return ExtractionOutput(**json_body)
    except (ValidationError, TypeError, json.JSONDecodeError) as exc:
        rich.print("Error parsing response from model during extraction:", exc)
        rich.print(f"Response: {text}")
        return None


def lookup_extraction_prompt(
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


def include_context_around_phrase(phrase: str, document: str, n_tokens: int):
    """
    n_tokens: the total number of tokens that should be included in the response
    """
    enc = tiktoken.encoding_for_model("text-davinci-003")

    if phrase not in document:
        warnings.warn(f"Phrase {phrase} was not in the supplied document: {document}")
        # If the phrase wasn't supplied, as a fallback just return the middle
        # 2000 tokens of the document
        surrounding_tokens = n_tokens // 2
        middle = len(document) // 2
        before, after = document[:middle], document[-middle:]
    else:
        phrase_token_length = len(enc.encode(phrase))
        surrounding_tokens = (n_tokens - phrase_token_length) // 2

        # Find the index of the phrase in the document
        # Split the document at that index
        split = document.split(phrase)
        if len(split) != 2:
            warnings.warn(f"Phrase {phrase} was found more than once in the document.")
            before, after = split[0], "".join(split[1:])
        else:
            before, after = split

    before = enc.decode(enc.encode(before)[-surrounding_tokens:])
    after = enc.decode(enc.encode(after)[:surrounding_tokens])

    return "".join((before, phrase, after))


def template_answer(i, r):
    context = include_context_around_phrase(
        r.output.extracted_text[0],
        "\n\n".join(p.text for p in r.search_pages),
        TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER,
    )

    return f"""INDEX {i}
Answer: {r.output.answer}
Rationale: {r.output.rationale}
Extracted Text: {r.output.extracted_text}
Supporting Text:
{context}"""


async def tournament_reduce(
    results: list[LookupOutput], term: str, district: District
) -> LookupOutput | None:
    if len(results) == 0:
        return None
    if len(results) == 1:
        return results[0]

    winners = []
    for competitor_batch in batched(
        (r for r in results if r.output is not None),
        TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE,
    ):
        input_prompt = tournament_reduce_tmpl.render(
            term=term,
            synonyms=", ".join(thesaurus.get(term, [])),
            zone_name=district.full_name,
            zone_abbreviation=district.short_name,
            answers="\n\n===\n\n".join(
                template_answer(i, r) for i, r in enumerate(competitor_batch)
            ),
        )

        text = await prompt(
            "gpt-4", [{"role": "user", "content": input_prompt}], max_tokens=1
        )
        if text is None:
            warnings.warn("No winner was present for a round in a tournament reduce.")
            continue

        try:
            index = int(text)
        except ValueError:
            warnings.warn(
                "Failed to parse index from tournament reduce response. Response was: {text}."
            )
            continue

        winner = competitor_batch[index]
        winners.append(winner)

    return await tournament_reduce(winners, term, district)


async def extract_answer(
    pages: list[PageSearchOutput],
    term: str,
    district: District,
    method: ExtractionMethod,
    model_name: str,
) -> AsyncGenerator[LookupOutput, None]:
    """
    Given a term to search for, will attempt to extract the value for the term
    from the provided pages.
    """

    if len(pages) == 0:
        return

    match method:
        case ExtractionMethod.NONE:
            for page in pages:
                yield LookupOutput(
                    output=None,
                    search_pages=[page],
                    search_pages_expanded=flatten(page_coverage([page])),
                )
        case ExtractionMethod.STUFF:
            # Stuff all pages into prompt, in order of page number
            all_page = reduce(
                lambda a, b: a + b.text, sorted(pages, key=lambda p: p.page_number), ""
            )
            # This is the length of the prompt before any template interpolation
            # TODO: Determine this automatically
            prompt_base_token_length = 256
            for chunk in batched(all_page, 8192 - prompt_base_token_length):
                yield LookupOutput(
                    output=parse_extraction_output(
                        await prompt(
                            model_name,
                            lookup_extraction_prompt(model_name, chunk, district, term),
                        )
                    ),
                    search_pages=pages,
                    search_pages_expanded=flatten(page_coverage(pages)),
                )
        case ExtractionMethod.TOURNAMENT_REDUCE:
            # We first map extraction across all pages.
            results = []
            async for r in extract_answer(
                pages, term, district, ExtractionMethod.MAP, model_name
            ):
                if r.output is not None:
                    results.append(r)

            # Then we reduce the answers to one in a tournament.
            final_result = await tournament_reduce(results, term, district)
            if final_result is not None:
                yield final_result
        case ExtractionMethod.MAP:

            async def worker(page):
                return (
                    page,
                    parse_extraction_output(
                        await prompt(
                            model_name,
                            lookup_extraction_prompt(
                                model_name, page.text, district, term
                            ),
                            max_tokens=384,
                        )
                    ),
                )

            for page, result in await gather(*map(worker, pages)):
                yield LookupOutput(
                    output=result,
                    search_pages=[page],
                    search_pages_expanded=flatten(page_coverage([page])),
                )
