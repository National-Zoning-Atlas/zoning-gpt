import warnings

from ...prompting import prompt
from ...utils import batched, get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput
from .map import MapExtractor
from .utils import include_context_around_phrase
import pickle 

TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE = 4
TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 500

# tournament_reduce_tmpl = get_jinja_environment().get_template("tournament.pmpt.tpl")
tournament_reduce_tmpl = get_jinja_environment().get_template("tournament_allow_none.pmpt.tpl")


async def tournament_reduce(
    results: list[LookupOutput], term: str, district: District, k: int
) -> int:
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
    # start first first item as initial comparison point 
    current_winner_index = 0
    # compare the current best answer to each other answer in results 
    for i in range(1, len(results)):
        # create prompt out of current pair for comparison 
        for competitor_batch in batched((r for r in (results[current_winner_index], results[i]) if r.output is not None), 2):
            input_prompt = tournament_reduce_tmpl.render(
                term=term,
                synonyms=", ".join(thesaurus.get(term, [])),
                district=district,
                answers="\n\n===\n\n".join(
                    template_answer(i, r) for i, r in enumerate(competitor_batch)
                ),
            )

            text = await prompt(
                "gpt-4-1106-preview", [{"role": "user", "content": input_prompt}], max_tokens=1
            )

            if text is None or text == "NO_ANSWER":
                warnings.warn("No winner was present for a round in a tournament reduce.")
                continue

            try:
                index = int(text)
            except ValueError:
                warnings.warn(
                    f"Failed to parse index from tournament reduce response. Response was: {text}."
                )
                continue
            # extract best answer from gpt result 
            if index == -1:
                current_winner_index = -1
            if index == 1:
                # this means it chose option 2 which was the ith result 
                current_winner_index = i
            
    if current_winner_index == -1:
        warnings.warn("GPT chose neither answer.")
        return current_winner_index
    winner = results[current_winner_index]
    return current_winner_index

class TournamentReduceExtractor(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, pages: list[LookupOutput], district: District, term: str
    ):
        results = pages
        print("PAGES", len(pages))
        for result in await tournament_reduce(results, term, district, self.k):
            yield result

        # # Ensure that we yield one empty result to handle case when the expected output is None
        # if len(empty_results) != 0:
        #     yield empty_results[0]
        

class TestTournamentReduceExtractor(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, results: list[LookupOutput], district: District, term: str
    ):
        # We first map extraction across all pages.
        results = []
        empty_results = []
        map_pickle = []
        async for r in super().extract(pages, district, term):
            map_pickle.append(r)
            if (r.output is not None) and r.output.extracted_text:
                results.append(r)
            else:
                empty_results.append(r)
                
        # Load existing data if the file already exists
        try:
            with open("./ryurtyn/map_results_11_14_0.dat", "rb") as f:
                existing_data = pickle.load(f)
        except FileNotFoundError:
            existing_data = {}  # If the file doesn't exist, initialize an empty dictionary

        try:
            with open("./ryurtyn/districts_11_14_0.dat", "rb") as f:
                existing_data_district = pickle.load(f)
        except FileNotFoundError:
            existing_data_district = {} 
        # Assuming district.short_name and results are defined somewhere in your code
        new_data = {district.short_name: map_pickle}
        new_data_district = {district.short_name: district}

        # Merge the existing data with the new data
        existing_data.update(new_data)
        existing_data_district.update(new_data_district)

        # Write the updated data back to the file
        with open("./ryurtyn/map_results_11_14_0.dat", "wb") as f:
            pickle.dump(existing_data, f)

        with open("./ryurtyn/districts_11_14_0.dat", "wb") as f:
            pickle.dump(existing_data_district, f)

        # for result in await tournament_reduce(results, term, district, self.k):
        #     yield result

        # Ensure that we yield one empty result to handle case when the expected output is None
        if len(empty_results) != 0:
            yield empty_results[0]