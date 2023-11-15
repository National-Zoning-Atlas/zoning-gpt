import json
import warnings
import csv
import datetime
import itertools
from zoning.utils import get_project_root

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
    sanitized_phrase = phrase.replace(" ", "-")
    sanitized_district_fn = district.full_name.replace(" ", "-")
    sanitized_district_sn = district.short_name.replace(" ", "-")
    now = datetime.datetime.now()

    filename = f"timestamp={now.strftime('%Y-%m-%d_%H-%M')}_town={town}_district={sanitized_district_fn}_term={term}_phrase={sanitized_phrase[:25]}_tokens={n_tokens}_occurrence={occurrence}.csv"

    root_directory = get_project_root()
    sub_folder_path = (
        root_directory / "data" / "logs" / "included_context_phrases" / occurrence
    )
    sub_folder_path.mkdir(parents=True, exist_ok=True)
    csv_file_path = sub_folder_path / filename

    if not csv_file_path.exists():
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.writer(file)
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

    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
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


# def adjust_context_indices(
#     document: str, start_idx: int, end_idx: int, char_budget: int = 500
# ) -> Tuple[int, int]:
#     """
#     Adjust the start and end indices for a context based on the character budget.

#     Parameters:
#     - document: The document string.
#     - start_idx: Starting index of the context.
#     - end_idx: Ending index of the context.
#     - char_budget: The total character count allowed for the context (default is 500).

#     Returns:
#     - Tuple of (adjusted_start_idx, adjusted_end_idx).
#     """
#     pre_context_budget = 100
#     pre_context_end = start_idx
#     pre_context_start = max(
#         0,
#         pre_context_end - pre_context_budget,
#         document.rfind("\n", 0, pre_context_end) + 1,
#     )

#     remaining_budget = (
#         char_budget - (pre_context_end - pre_context_start) - (end_idx - start_idx)
#     )
#     post_context_start = end_idx
#     post_context_end = min(len(document), post_context_start + remaining_budget)

#     return pre_context_start, post_context_end


def get_multiple_matches(document: str, phrase: str) -> List[int]:
    """Return all starting indices of a phrase within a document."""
    starts = []
    idx = document.find(phrase)
    while idx != -1:
        starts.append(idx)
        idx = document.find(phrase, idx + 1)
    return starts


# def select_best_candidate(context_details: List[RelevantContext]) -> RelevantContext:
#     # Select the best candidates sorting by minimum distance:
#     context_details.sort(key=lambda x: x[1])
#     return context_details[0]

# def fallback_context(
#     document: str, phrase: str, index: int = None
# ) -> Tuple[int, int, Optional[int], Optional[int], str]:
#     """
#     Retrieve a context around a given phrase or index if the phrase is not found.

#     Parameters:
#     - document: The document string.
#     - phrase: Target phrase.
#     - index: Index of the phrase (default is None).

#     Returns:
#     - Context details.
#     """
#     if index is None:
#         index = document.find(phrase)

#     if index == -1:
#         warnings.warn(f"Phrase {phrase} was not in the supplied document.")
#         start, end = adjust_context_indices(
#             document, len(document) // 2, len(document) // 2
#         )
#     else:
#         start, end = adjust_context_indices(document, index, index + len(phrase))

#     context = document[start:end]
#     chars = len(context)
#     distance = 42424242 if index == -1 else 0
#     line_idx = document.split("\n").index(phrase) + 1 if index != -1 else None
#     return (chars, distance, line_idx, -1, context)


def get_relevant_context(document: str, *phrases: str) -> List[RelevantContext]:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
    valid_phrases = [phrase for phrase in phrases if phrase]
    if not valid_phrases:
        warnings.warn("No valid phrases were supplied.")
        return ""

    if len(valid_phrases) == 1:
        return fallback_context(document, valid_phrases[0], index=-1)

    phrase_pairs = list(itertools.combinations(valid_phrases, 2))
    context_details = []
    lines = document.split("\n")

    for pair in phrase_pairs:
        idx_0_starts, idx_1_starts = get_multiple_matches(
            document, pair[0]
        ), get_multiple_matches(document, pair[1])

        # Handle multiple matches
        if len(idx_0_starts) > 1:
            warnings.warn(
                f"More than one match found for '{pair[0]}'. Considering all matches."
            )
        if len(idx_1_starts) > 1:
            warnings.warn(
                f"More than one match found for '{pair[1]}'. Considering all matches."
            )

        for i_0 in idx_0_starts:
            for i_1 in idx_1_starts:
                if i_0 < i_1:
                    start, end = adjust_context_indices(
                        document, i_0, document.find("\n", i_1) + 1
                    )
                    context = document[start:end]
                    chars = len(context)
                    distance = i_1 - i_0
                    line_idx_0, line_idx_1 = (
                        document[:i_0].count("\n") + 1,
                        document[:i_1].count("\n") + 1,
                    )
                    context_details.append(
                        (chars, distance, line_idx_0, line_idx_1, context)
                    )
                else:
                    context_details.append(fallback_context(document, pair[0]))

    before_context = enc.decode(enc.encode(before))
    after_context = enc.decode(enc.encode(after))

    # return "".join((before_context, phrase, after_context))

    return select_best_candidate(context_details)


def phrase_token_length_counter(phrase: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")
    return


def include_context_around_phrase(
    phrase: str, document: str, n_tokens: int, term: str, district: District, town: str
):
    occurrences = document.split(phrase)
    # create an occurences dict to map the index of document.split(phrase) to the case
    ocurrences_map = {1: "not-found", 2: "found-once", 3: "found-multiple-times"}
    occurrences = ocurrences_map[len(occurrences)]
    if ocurrences_map[occurrences] == "not-found":
        _, _, _, _, context = fallback_context(document, phrase)
        before = context[: context.find(phrase)]
        after = context[context.find(phrase) + len(phrase) :]
        occurrence = "not-found"
    elif ocurrences_map[occurrences]:
        before, after = occurrences
        occurrence = "found-once"
    else:
        warnings.warn(f"Phrase {phrase} was found more than once in the document.")
        selected_context = get_relevant_context(document, *phrase)
        _, _, _, _, context = selected_context
        before = context[: context.find(phrase)]
        after_context = context[context.find(phrase) + len(phrase) :]
        occurrence = "found-multiple-times"

    log_to_csv(
        phrase,
        document,
        n_tokens,
        occurrence,
        original_context,
        town,
        district,
        term,
    )

    return context


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
}


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
            print(f"USING LEGACY MODEL: {TEMPLATE_MAPPING}")
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
        )
    )
