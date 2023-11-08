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
# tournament_reduce_tmpl = get_jinja_environment().get_template("tournament_allow_none.pmpt.tpl")


async def tournament_reduce(
    results: list[LookupOutput], term: str, district: District, k: int, town: str
) -> list[LookupOutput]:
    if len(results) <= k:
        return results

    def template_answer(idx, record):
        context = include_context_around_phrase(
            phrase=record.output.extracted_text[0],
            document="\n\n".join(page.text for page in record.search_pages),
            n_tokens=TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER,
            term=term,
            district=district,
            town=town,
        )

        return (
            f"INDEX {idx}\n"
            f"Answer: {record.output.answer}\n"
            f"Rationale: {record.output.rationale}\n"
            f"Extracted Text: {record.output.extracted_text}\n"
            f"Supporting Text:\n{context}"
        )

    thesaurus = get_thesaurus()
    # start first first item as initial comparison point
    current_winner_index = 0
    # compare the current best answer to each other answer in results
    for i in range(1, len(results)):
        # create prompt out of current pair for comparison
        for competitor_batch in batched(
            (
                r
                for r in (results[current_winner_index], results[i])
                if r.output is not None
            ),
            2,
        ):
            input_prompt = tournament_reduce_tmpl.render(
                term=term,
                synonyms=", ".join(thesaurus.get(term, [])),
                district=district,
                town=town,
                answers="\n\n===\n\n".join(
                    template_answer(i, r) for i, r in enumerate(competitor_batch)
                ),
            )
            text = await prompt(
                "gpt-4", [{"role": "user", "content": input_prompt}], max_tokens=1
            )

            if text is None or text == "NO_ANSWER":
                warnings.warn(
                    "No winner was present for a round in a tournament reduce."
                )
                continue

            try:
                index = int(text)
            except ValueError:
                warnings.warn(
                    "Failed to parse index from tournament reduce response. Response was: {text}."
                )
                continue
            # extract best answer from gpt result
            if index == 1:
                # this means it chose option 2 which was the ith result
                current_winner_index = i

    winner = results[current_winner_index]
    return [winner]


class TournamentReduceExtractor(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, pages: list[PageSearchOutput], district: District, term: str, town: str
    ):
        # We first map extraction across all pages.
        results = []
        empty_results = []
        async for r in super().extract(pages, district, term):
            if r.output is not None:
                results.append(r)
            else:
                empty_results.append(r)

        for result in await tournament_reduce(results, term, district, self.k, town):
            yield result

        # Ensure that we yield one empty result to handle case when the expected output is None
        if len(empty_results) != 0:
            yield empty_results[0]
