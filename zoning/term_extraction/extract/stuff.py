from functools import reduce

from ...utils import batched, flatten

# TODO: Move this elsewhere
from ..search.utils import page_coverage
from ..types import District, LookupOutput, PageSearchOutput
from .base import Extractor
from .utils import run_extraction_prompt


class StuffExtractor(Extractor):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def extract(
        self, pages: list[PageSearchOutput], district: District, term: str
    ):
        # Stuff all pages into prompt, in order of page number
        all_page = reduce(
            lambda a, b: a + b.text, sorted(pages, key=lambda p: p.page_number), ""
        )
        # This is the length of the prompt before any template interpolation
        # TODO: Determine this automatically
        prompt_base_token_length = 256
        for chunk in batched(all_page, 8192 - prompt_base_token_length):
            yield LookupOutput(
                output=await run_extraction_prompt(
                    self.model_name, district, term, chunk
                ),
                search_pages=pages,
                search_pages_expanded=flatten(page_coverage(pages)),
            )
