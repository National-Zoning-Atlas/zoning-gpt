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
            f"- Answer: {record.output.answer}\n"
            f"- Rationale: {record.output.rationale}\n"
            f"- Extracted Text: {record.output.extracted_text}\n"
            f"- Supporting Text:\n{context}"
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
        text = json.loads(matches[0])["Answer"] if len(matches) > 0 else "N"
    except json.JSONDecodeError:
        text = "N"
    logger.info(f"<ConfirmExtractor>: Text: {text}")
    logger.info(
        f"<ConfirmExtractor>: town: {town}, district: {district.full_name}, answer: {result.output}, response: {text}")
    if text is None or text == "NO_ANSWER":
        logger.warn(
            "Null GPT response"
        )
    elif text == "Y":
        return LookupOutputConfirmed(
            output=result.output,
            search_pages=result.search_pages,
            search_pages_expanded=result.search_pages_expanded,
            confirmed=True,
            confirmed_raw=output,
            original_output=result.output
        )
    elif text == "N":
        return LookupOutputConfirmed(
            output=None,
            search_pages=result.search_pages,
            search_pages_expanded=result.search_pages_expanded,
            confirmed=False,
            confirmed_raw=output,
            original_output=result.output
        )
    else:
        logger.warn(f"GPT returned something unexpected, val: {text}")


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
