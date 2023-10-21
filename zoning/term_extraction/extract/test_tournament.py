import warnings
import pickle 

from ...prompting import prompt
from ...utils import batched, get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput
from .map import MapExtractor
from .utils import include_context_around_phrase

from typing import Any

TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE = 4
TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 500

tournament_reduce_tmpl = get_jinja_environment().get_template("tournament.pmpt.tpl")
#tournament_reduce_tmpl = get_jinja_environment().get_template("tournament_allow_none.pmpt.tpl")


async def tournament_reduce(
        correct: LookupOutput, incorrect: LookupOutput, term: str, district: District, k: int
) -> list[LookupOutput]:
    
    def template_answer(i, r):
        context = include_context_around_phrase(
            r.output.extracted_text[0],
            "\n\n".join(p.text for p in r.search_pages),
            TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER,
        )

        return f"INDEX {i}\nAnswer: {r.output.answer}\nRationale: {r.output.rationale}\nExtracted Text: {r.output.extracted_text}\nSupporting Text:\n{context}"

    thesaurus = get_thesaurus()

    winners = []
    winner_indices = []
    
    for competitor_batch in batched(
        (r for r in (correct, incorrect) if r.output is not None),
        2,
    ):
        input_prompt = tournament_reduce_tmpl.render(
            term=term,
            synonyms=", ".join(thesaurus.get(term, [])),
            district=district,
            answers="\n\n===\n\n".join(
                template_answer(i, r) for i, r in enumerate(competitor_batch)
            ),
        )

        with open("tournament_prompts.txt", "a") as f:
            f.write(input_prompt)
            f.write("\n")
        text = await prompt(
            "gpt-4", [{"role": "user", "content": input_prompt}], max_tokens=1
        )
        if text is None or text == "NO_ANSWER":
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
        winner_indices.append(index)
        winners.append(winner)

    return winners, winner_indices

    # return await tournament_reduce(winners, term, district, k)


class TournamentTester(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, gpt_answers: list[LookupOutput], district: District, term: str, correct_answer: LookupOutput
    ):
        winners = []
        winner_indices = []
        for answer in gpt_answers: 
            # winner between gpt answer and true answer
            winner, winner_index = await tournament_reduce(correct=correct_answer, incorrect=answer, term=term, district=district, k=self.k)
            winners.append(winner)
            winner_indices.append(winner_index)
        
        print(winner_indices)
        return winner_indices
    # maybe here i can use expected and i can go for every result in outputs 
    # for each result in outputs, call my test_tournament on that result and the ground truth 
    # for each of those results, we can just count the number of times that the ground truth index is given back 
        # # We first map extraction across all pages.
        # results = []
        # empty_results = []
        # async for r in super().extract(pages, district, term):
        #     if r.output is not None:
        #         results.append(r)
        #     else:
        #         empty_results.append(r)

        # # save snapshot of the map results so that you can avoid running map stuff 
        # with open("map_results.dat", "wb") as f:
        #     pickle.dump(results, f)

        # with open("map_results.dat", "rb") as f:
        #     results = pickle.load(f)
        
        # print("NUMBER OF RESULTS:", str(len(results)))

        # counter = 0
        # for result in results:
        #     counter += 1
        #     # print("RESULT NUMBER: ", counter, result)
        #     yield result

        # SNAPSHOT_PATH = str(search_method) + "_" + str(extraction_method) + "_" + str(k) + "_" + str(tournament_k) + ".csv"
        # SNAPSHOT_METRICS_PATH = str(search_method) + "_" + str(extraction_method) + "_" + str(k) + "_" + str(tournament_k) + ".yaml"
        # df = pd.read_parquet(EVAL_OUTPUT_PATH, engine='pyarrow')
        # df.to_csv(SNAPSHOT_PATH, index=False)
        
        # for result in await tournament_reduce(results, term, district, self.k):
        #     yield result

        # # Ensure that we yield one empty result to handle case when the expected output is None
        # if len(empty_results) != 0:
        #     yield empty_results[0]