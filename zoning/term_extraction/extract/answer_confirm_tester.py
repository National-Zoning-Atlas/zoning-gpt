import warnings

from ...prompting import prompt
from ...utils import batched, get_jinja_environment
from ..thesaurus import get_thesaurus
from ..types import District, LookupOutput, PageSearchOutput
from .map import MapExtractor
from .utils import include_context_around_phrase
import pickle 

TOURNAMENT_REDUCE_MAX_ANSWERS_PER_STAGE = 4
TOURNAMENT_REDUCE_CONTEXT_TOKENS_PER_ANSWER = 2000
final_answer_tmpl = get_jinja_environment().get_template("answer_confirm.pmpt.tpl")

async def answer_confirm_test(
    result: LookupOutput, term: str, district: District, k: int, town: str
) -> str:

    thesaurus = get_thesaurus()
    # start first first item as initial comparison point
    current_winner_index = 0
    # compare the current best answer to each other answer in results

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

    input_prompt = final_answer_tmpl.render(
        term=term,
        synonyms=", ".join(thesaurus.get(term, [])),
        district=district,
        town=town,
        answer="\n\n===\n\n".join(template_answer(result)), 
    )
    print(input_prompt)

    text = await prompt(
        "gpt-4-1106-preview", [{"role": "user", "content": input_prompt}], max_tokens=1
    )
    
    if text is None or text == "NO_ANSWER":
        warnings.warn(
            "Null GPT response"
        )

    return text

class AnswerConfirmTester(MapExtractor):
    def __init__(self, model_name: str, k: int):
        super().__init__(model_name)
        self.k = k

    async def extract(
        self, input: LookupOutput, district: District, term: str, town: str
    ):
        result = input
        for r in await answer_confirm_test(result, term, district, self.k, town):
            yield r

        # # Ensure that we yield one empty result to handle case when the expected output is None
        # if len(empty_results) != 0:
        #     yield empty_results[0]