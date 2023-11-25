import warnings

from ...prompting import prompt
from ...utils import batched, get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput
from .map import MapExtractor
from .utils import include_context_around_phrase

MULTIPLE_CHOICE_MAX_ANSWERS_PER_STAGE = 10
MULTIPLE_CHOICE_CONTEXT_TOKENS_PER_ANSWER = 500

multiple_choice_tmpl = get_jinja_environment().get_template("multiple_choice.pmpt.tpl")


async def multiple_choice(
    results: list[LookupOutput], term: str, district: District, k: int
) -> list[LookupOutput]:
    if len(results) <= 1:
        return results

    def template_answer(i, r):
        context = include_context_around_phrase(
            r.output.extracted_text[0],
            "\n\n".join(p.text for p in r.search_pages),
            MULTIPLE_CHOICE_CONTEXT_TOKENS_PER_ANSWER,
        )

        return f"INDEX {i}\nAnswer: {r.output.answer}\nRationale: {r.output.rationale}\nExtracted Text: {r.output.extracted_text}\nSupporting Text:\n{context}"

    thesaurus = get_thesaurus()
    winners = []

    # make batch size k so that all k results are passed in at once
    for competitor_batch in batched(
        (r for r in results if r.output is not None),
        k,
    ):
        input_prompt = multiple_choice_tmpl.render(
            term=term,
            synonyms=", ".join(thesaurus.get(term, [])),
            district=district,
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
        if int(index) == -1:
            warnings.warn("GPT chose answer: None of the above.")
            continue

        winner = competitor_batch[index]
        winners.append(winner)

    return await multiple_choice(winners, term, district, k)


class MultipleChoiceExtractor(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, pages: list[PageSearchOutput], district: District, term: str, town: str
    ):
        # We first map extraction across all pages.
        results = []
        empty_results = []
        async for r in super().extract(pages, district, term, town):
            if r.output is not None:
                results.append(r)
            else:
                empty_results.append(r)
        for result in await multiple_choice(results, term, district, self.k):
            yield result

        if len(empty_results) != 0:
            yield empty_results[0]
