import json
import warnings
import csv
import datetime
import itertools
from zoning.utils import get_project_root, logger
import re

import rich
import tiktoken
from pydantic import ValidationError

from ...prompting import prompt
from ...utils import (
    get_jinja_environment,
)
from typing import List, Tuple, Optional
from ..thesaurus import get_thesaurus
from ..types import District, ExtractionOutput, RelevantContext


def sanitize_for_filename(input_str, max_length=255):
    # Remove special characters and limit length
    sanitized = re.sub(r'[\\/*?:"<>|\n]', "", input_str)
    return sanitized[:max_length]


def log_to_csv(
    phrase: str,
    document: str,
    n_tokens: int,
    occurrence: str,
    before_context: str,
    after_context: str,
    town: str,
    district: District,
    term: str,
):
    sanitized_phrase = sanitize_for_filename(phrase.replace(" ", "-"), 50)
    sanitized_district_fn = sanitize_for_filename(
        district.full_name.replace(" ", "-"), 50
    )
    sanitized_district_sn = sanitize_for_filename(
        district.short_name.replace(" ", "-"), 50
    )
    sanitized_town = sanitize_for_filename(town, 50)
    sanitized_term = sanitize_for_filename(term, 50)
    now = datetime.datetime.now()

    filename = f"timestamp={now.strftime('%Y-%m-%d_%H-%M')}_town={sanitized_town}_district={sanitized_district_fn}_term={sanitized_term}_phrase={sanitized_phrase}_tokens={n_tokens}_occurrence={occurrence}.csv"

    filename = sanitize_for_filename(filename, 100)

    root_directory = get_project_root()
    sub_folder_path = (
        root_directory / "data" / "logs" / "included_context_phrases" / occurrence
    )

    # Ensure the sub-folder exists
    sub_folder_path.mkdir(parents=True, exist_ok=True)
    csv_file_path = sub_folder_path / filename

    # Check if the file exists and create it with headers if it does not
    file_exists = csv_file_path.exists()
    # Construct the full path for the CSV file

    mode = "a" if file_exists else "w"

    with open(csv_file_path, mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write header only if the file did not exist
        if not file_exists:
            writer.writerow(
                [
                    "phrase",
                    "town",
                    "district.full_name",
                    "district.short_name",
                    "term",
                    "n_tokens",
                    "occurrence",
                    "before-context",
                    "after-context",
                    "document",
                    "timestamp",
                ]
            )

        # Write the data row
        writer.writerow(
            [
                phrase,
                town,
                sanitized_district_fn,
                sanitized_district_sn,
                term,
                n_tokens,
                occurrence,
                before_context,
                after_context,
                document,
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
            ]
        )


def find_regex_matched_phrase_cell(original_phrase: str, document: str):
    """
    Find a regex matched phrase in the document that corresponds to the original phrase
    with variations in whitespace and characters between segments of the sentence.
    """
    # Escape special characters in the original phrase
    escaped_phrase = re.escape(original_phrase.split("\n")[0])

    # Create a regex pattern to match the variations of the target sentence
    pattern = rf"{escaped_phrase}\s*\(?\s*CELL\s*\d*\(?\d*,?\s*\d*\)?\)?\s*:\s*\n\d+"

    # Search for the pattern in the document
    match = re.search(pattern, document)

    # Return the matched phrase or the original phrase if no match is found
    return match.group(0) if match else original_phrase


# def find_regex_matched_phrase_multiple_returns(original_phrase: str, document: str):
#     """
#     Find a regex matched phrase in the document that corresponds to the original phrase,
#     accounting for multiple \n that might be ignored and need to be returned, and variations in whitespace.
#     """
#     # Escape special characters in the original phrase and create a regex pattern
#     # that allows for any whitespace (including newlines) between each part of the phrase
#     pattern = re.escape(original_phrase)
#     pattern = pattern.replace(r"\ ", r"\s*").replace(r"\\n", r"(\\n|\s)+")

#     # Search for the pattern in the document, allowing '.' to match newlines with re.DOTALL
#     match = re.search(pattern, document, re.DOTALL)

#     # If a match is found, reconstruct the phrase with the original line breaks and whitespace
#     if match:
#         matched_text = match.group(0)
#         # Normalize the matched text by collapsing multiple spaces and newlines into single spaces
#         normalized_text = re.sub(r"\s+", " ", matched_text)
#         # Reconstruct the phrase with the original line breaks
#         reconstructed_phrase = original_phrase.replace(" ", r"\s*").replace(
#             "\n", r"(\\n|\s)+"
#         )
#         return re.sub(reconstructed_phrase, normalized_text, original_phrase)
#     else:
#         return original_phrase


def include_context_around_phrase(
    phrase: str, document: str, n_tokens: int, term: str, district: District, town: str
):
    """
    n_tokens: the total number of tokens that should be included in the response
    """
    enc = tiktoken.encoding_for_model("text-davinci-003")

    occurrences_list = document.split(phrase)
    occurrences_count = len(occurrences_list)
    # Create an occurrences dict to map the count to the case
    occurrences_map = {1: "not-found", 2: "found-once"}
    # occurrences_map using occurrences_count as the key and defaulting to "found-multiple-times" if higher
    occurrence = occurrences_map.get(occurrences_count, "found-multiple-times")

    surrounding_tokens = n_tokens // 2
    before = ""
    after = ""

    if occurrence == "not-found":
        # If the phrase wasn't supplied, as a fallback just return the middle
        # 2000 tokens of the document
        # filtered_phrase = find_regex_matched_phrase_multiple_returns(phrase, document)
        new_phrase = find_regex_matched_phrase_cell(phrase, document)

        if new_phrase not in document:
            # warnings.warn(
            #     f"Phrase {phrase} was not in the supplied. document: {document}"
            # )
            middle = len(document) // 2
            before, after = document[:middle], document[-middle:]
        else:
            return include_context_around_phrase(
                phrase=new_phrase,
                document=document,
                n_tokens=n_tokens,
                term=term,
                district=district,
                town=town,
            )

    elif occurrence == "found-once":
        before, after = occurrences_list[0], occurrences_list[1]
    else:
        logger.warn(f"Phrase {phrase} was found more than once in the document.")
        # can be improved
        before, after = occurrences_list[0], "".join(occurrences_list[1:])
        phrase_token_length = len(enc.encode(phrase))
        surrounding_tokens = (n_tokens - phrase_token_length) // 2

    before_context = enc.decode(enc.encode(before)[-surrounding_tokens:])
    after_context = enc.decode(enc.encode(after)[:surrounding_tokens])

    log_to_csv(
        phrase=phrase,
        document=document,
        n_tokens=n_tokens,
        occurrence=occurrence,  # this should be the string "not-found", "found-once", or "found-multiple-times"
        before_context=before,
        after_context=after,
        town=town,
        district=district,  # make sure this is the District object that has full_name and short_name
        term=term,
    )

    return "".join((before_context, phrase, after_context))


extraction_chat_completion_tmpl = get_jinja_environment().get_template(
    "extraction_chat_completion.pmpt.tpl"
)
extraction_completion_tmpl = get_jinja_environment().get_template(
    "extraction_completion.pmpt.tpl"
)

extract_chat_completion_tmpl = get_jinja_environment().get_template(
    "extract_chat_completion.pmpt.tpl"
)


TEMPLATE_MAPPING = {
    "text-davinci-003": extraction_completion_tmpl,
    "gpt-3.5-turbo": extraction_chat_completion_tmpl,
    "gpt-4": extraction_chat_completion_tmpl,
    "gpt-4-1106-preview": extraction_chat_completion_tmpl,
    "gpt-4-turbo": extraction_chat_completion_tmpl,
}


def parse_extraction_output(text: str | None) -> ExtractionOutput | None:
    if text is None or text == "null":
        return None

    try:
        # TODO: this is something that came with new gpt update. This is a bandaid solution that i'll look into later
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
        case "gpt-4-1106-preview" | "gpt-4-turbo":
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
    #raise NotImplementedError
    return parse_extraction_output(
        await prompt(
            model_name,
            lookup_extraction_prompt(model_name, contents, district, term),
            max_tokens=384,
        )
    )

