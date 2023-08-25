import warnings

from ...prompting import prompt
from ...utils import batched, get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput
from .map import MapExtractor
from .utils import include_context_around_phrase

TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE = 4
TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 500

tournament_reduce_tmpl = get_jinja_environment().get_template("tournament.pmpt.tpl")


async def tournament_reduce(
    results: list[LookupOutput], term: str, district: District, k: int
) -> list[LookupOutput]:
    if len(results) <= k:
        return results

    def template_answer(i, r):
        context = include_context_around_phrase(
            r.output.extracted_text[0],
            "\n\n".join(p.text for p in r.search_pages),
            TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER,
        )

        return f"INDEX {i}\nAnswer: {r.output.answer}\nRationale: {r.output.rationale}\nExtracted Text: {r.output.extracted_text}\nSupporting Text:\n{context}"

    thesaurus = get_thesaurus()

    winners = []
    for competitor_batch in batched(
        (r for r in results if r.output is not None),
        TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE,
    ):
        input_prompt = tournament_reduce_tmpl.render(
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

        winner = competitor_batch[index]
        winners.append(winner)

    return await tournament_reduce(winners, term, district, k)


class TournamentReduceExtractor(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, pages: list[PageSearchOutput], district: District, term: str
    ):
        # We first map extraction across all pages.
        results = []
        async for r in super().extract(pages, district, term):
            if r.output is not None:
                results.append(r)
        for result in await tournament_reduce(results, term, district, self.k):
            yield result
