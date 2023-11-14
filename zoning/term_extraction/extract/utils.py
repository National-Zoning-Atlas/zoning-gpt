import json
import warnings

import rich
import tiktoken
from pydantic import ValidationError

from ...prompting import prompt
from ...utils import (
    get_jinja_environment,
)
from ..thesaurus import get_thesaurus
from ..types import District, ExtractionOutput


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


extraction_chat_completion_tmpl = get_jinja_environment().get_template(
    "extraction_chat_completion.pmpt.tpl"
)
extraction_completion_tmpl = get_jinja_environment().get_template(
    "extraction_completion.pmpt.tpl"
)


TEMPLATE_MAPPING = {
    "text-davinci-003": extraction_completion_tmpl,
    "gpt-3.5-turbo": extraction_chat_completion_tmpl,
    "gpt-4": extraction_chat_completion_tmpl,
    "gpt-4-1106-preview": extraction_chat_completion_tmpl,
}


def parse_extraction_output(text: str | None) -> ExtractionOutput | None:
    if text is None or text == "null":
        return None

    try:
        # text = text[7:-4]
        if text[:7] == "```json":
            text = text[7:-4]
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
                synonyms=", ".join(get_thesaurus().get(term, [])),
                zone_name=district.full_name,
                zone_abbreviation=district.short_name,
            )
        case "gpt-3.5-turbo" | "gpt-4":
            return [
                {
                    "role": "system",
                    "content": TEMPLATE_MAPPING[model_name].render(
                        term=term,
                        synonyms=", ".join(get_thesaurus().get(term, [])),
                        zone_name=district.full_name,
                        zone_abbreviation=district.short_name,
                    ),
                },
                {
                    "role": "user",
                    "content": f"Input: \n\n {page_text}\n\n Output:",
                },
            ]
        case "gpt-4-1106-preview":
            return [
                {
                    "role": "system",
                    "content": TEMPLATE_MAPPING[model_name].render(
                        term=term,
                        synonyms=", ".join(get_thesaurus().get(term, [])),
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


async def run_extraction_prompt(
    model_name: str, district: District, term: str, contents: str
):
    return parse_extraction_output(
        await prompt(
            model_name,
            lookup_extraction_prompt(model_name, contents, district, term),
            max_tokens=384,
            # response_format={"type": "json_object" }
        )
    )
