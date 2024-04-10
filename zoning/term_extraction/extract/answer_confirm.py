import warnings
import re
import json

from zoning.term_extraction.extract.tournament_reduce import TournamentReduceExtractor
from ..value_ranges import get_value_ranges

from ...prompting import prompt
from ...utils import get_jinja_environment, logger
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput, LookupOutputConfirmed
from .utils import include_context_around_phrase

TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 3500
# final_answer_tmpl = get_jinja_environment().get_template("answer_confirm.pmpt.tpl")
final_answer_tmpl = get_jinja_environment().get_template("answer_confirm_explain.pmpt.tpl")


async def answer_confirm(
        result: LookupOutput, term: str, district: District, town: str
) -> LookupOutputConfirmed:
    logger.info(f"Starting answer_confirm for term: {term}, district: {district.full_name}, town: {town}")

    thesaurus = get_thesaurus()
    value_ranges = get_value_ranges()

    def template_answer(record):
        context = include_context_around_phrase(
            phrase=record.output.extracted_text[0],
            document="\n\n".join(page.text for page in record.search_pages),
            n_tokens=TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER,
            term=term,
            district=district,
            town=town,
        )

        return (
            f"Answer: {record.output.answer}\n"
            f"Rationale: {record.output.rationale}\n"
            f"Extracted Text: {record.output.extracted_text}\n"
            f"Supporting Text:\n{context}"
        )

    not_synonyms = []
    for key in value_ranges:
        if key != term:
            not_synonyms.extend(thesaurus.get(key, []))

    input_prompt = final_answer_tmpl.render(
        term=term,
        value_range=value_ranges.get(term, None),
        synonyms=", ".join(thesaurus.get(term, [])),
        not_synonyms=", ".join(not_synonyms),
        district=district,
        town=town,
        answer=template_answer(result),
    )

    output = await prompt(
        "gpt-4-1106-preview", [{"role": "user", "content": input_prompt}],
        max_tokens=256,
    )

    logger.info(f"<ConfirmExtractor>: GPT Response: {output}")
    pattern = r"(\{[^}]+\})"
    matches = re.findall(pattern, output)

    try:
        parsed_json = json.loads(matches[0]) if len(matches) > 0 else {}
    except json.JSONDecodeError:
        logger.error(f"Error parsing JSON: {matches[0]}")
        parsed_json = {}

    def get_json_value(key, default_value):
        try:
            return parsed_json.get(key, default_value)
        except KeyError:
            logger.error(f"Key {key} not found in JSON.")
            return default_value

    answer_confirm_flag = get_json_value("Answer", "N")
    rationale = get_json_value("Rationale", "Explanation not provided")
    is_district_presented = get_json_value("is_district_presented", "N")
    is_term_presented = get_json_value("is_term_presented", "N")
    is_correct_value_present = get_json_value("is_correct_value_present", "N")
    extracted_district = get_json_value("extracted_district", "N/A")
    extracted_term = get_json_value("extracted_term", "N/A")

    logger.info(f"<ConfirmExtractor>: answer_confirm_flag: {answer_confirm_flag}")
    logger.info(f"<ConfirmExtractor>: rationale: {rationale}")
    logger.info(f"<ConfirmExtractor>: is_district_presented: {is_district_presented}")
    logger.info(f"<ConfirmExtractor>: is_term_presented: {is_term_presented}")
    logger.info(f"<ConfirmExtractor>: is_correct_value_present: {is_correct_value_present}")
    logger.info(f"<ConfirmExtractor>: extracted_district: {extracted_district}")
    logger.info(f"<ConfirmExtractor>: extracted_term: {extracted_term}")

    logger.info(
        f"<ConfirmExtractor>: town: {town}, district: {district.full_name}, answer: {result.output}, response: {answer_confirm_flag}")

    subquestions = {
        "is_district_presented": is_district_presented,
        "is_term_presented": is_term_presented,
        "is_correct_value_present": is_correct_value_present,
        'answer_confirm_flag': answer_confirm_flag,
        'extracted_district': extracted_district,
        'extracted_term': extracted_term,
    }

    if answer_confirm_flag is None or answer_confirm_flag == "NO_ANSWER":
        logger.warn(
            "Null GPT response"
        )
    elif answer_confirm_flag == "Y":
        return LookupOutputConfirmed(
            output=result.output,
            search_pages=result.search_pages,
            search_pages_expanded=result.search_pages_expanded,
            confirmed=True,
            confirmed_raw=output,
            original_output=result.output,
            subquestions=subquestions
        )
    elif answer_confirm_flag == "N":
        return LookupOutputConfirmed(
            output=None,
            search_pages=result.search_pages,
            search_pages_expanded=result.search_pages_expanded,
            confirmed=False,
            confirmed_raw=output,
            original_output=result.output,
            subquestions=subquestions
        )
    else:
        logger.warn(f"GPT returned something unexpected, val: {answer_confirm_flag}")


class ConfirmExtractor(TournamentReduceExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name, k)
        self.k = k

    async def extract(
            self, pages: list[PageSearchOutput], district: District, term: str, town: str
    ):
        # We first tournament reduce
        results = []
        empty_results = []
        async for r in super().extract(pages, district, term, town):
            if (r.output is not None) and r.output.extracted_text:
                results.append(r)
            else:
                empty_results.append(r)

        for r in results:
            result = await answer_confirm(r, term, district, town)
            yield result

        # Ensure that we yield one empty result to handle case when the expected output is None
        if len(results) == 0:
            yield empty_results[0]
