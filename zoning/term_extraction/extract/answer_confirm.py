import warnings

from zoning.term_extraction.extract.tournament_reduce import TournamentReduceExtractor
from ..value_ranges import get_value_ranges

from ...prompting import prompt
from ...utils import get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput, LookupOutputConfirmed
from .utils import include_context_around_phrase

TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 3500
final_answer_tmpl = get_jinja_environment().get_template("answer_confirm.pmpt.tpl")



async def answer_confirm(
    result: LookupOutput, term: str, district: District, town: str
) -> LookupOutputConfirmed:

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

    text = await prompt(
        "gpt-4-1106-preview", [{"role": "user", "content": input_prompt}], max_tokens=1
    )

    print(town, district.full_name, "answer: ", result.output.answer, "response: ", text)
    if text is None or text == "NO_ANSWER":
        warnings.warn(
            "Null GPT response"
        )
    elif text == "Y":
        return LookupOutputConfirmed(
            output=result.output,
            search_pages=result.search_pages,
            search_pages_expanded=result.search_pages_expanded,
            confirmed=True,
            confirmed_raw=text,
            original_output=result.output
        )
    elif text == "N":
        return LookupOutputConfirmed(
                output=None,
                search_pages=[],
                search_pages_expanded=[],
                confirmed=False,
                confirmed_raw=text,
                original_output=result.output
            )
    else:
        warnings.warn("GPT returned something unexpected")


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
