import warnings
import pickle 

from ...prompting import prompt
from ...utils import batched, get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput, ExtractionOutput
from .map import MapExtractor
from .utils import include_context_around_phrase

from typing import Any
import polars as pl

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

        with open("10-22-tournament_prompts.txt", "a") as f:
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

    return winners, winner_indices, input_prompt


class TournamentTester(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(self, correct_answers: Any, all_answers: Any, district: District, term: str):
        # correct answers is polars datafram, all answers is polars dataframe 
        winners = []
        winner_indices = []
        # not sure why theres multiple correct answers but for now just pick one 
        correct_pickle = correct_answers.select("pickled_result")[0].item()
        correct_object = pickle.loads(correct_pickle)

        for row in range(len(all_answers)):
            curr_pickle = all_answers.select("pickled_result")[row].item() 
            curr_object = pickle.loads(curr_pickle)
            # if the answer has an output at all #TODO: check which ones don't and why
            if curr_object.output and correct_object.output:
                # dont need to compare two of the same answer value
                if curr_object.output.answer == correct_object.output.answer:
                    continue
                # winner between gpt answer and true answer
                winner, winner_index, input_prompt = await tournament_reduce(correct=correct_object, incorrect=curr_object, term=term, district=district, k=self.k)
                winners.append(winner)
                if winner_index[0] == 1:
                    print("wrong!")
                    
                winner_indices.append(winner_index)
        
        print(winner_indices)
        return winners, winner_indices