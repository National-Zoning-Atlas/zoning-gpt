from asyncio import gather

from ...utils import flatten, logger
from ..search.utils import page_coverage
from ..types import LookupOutput, PageSearchOutput, District


from .utils import run_extraction_prompt
from .base import Extractor


class MapExtractor(Extractor):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def extract(
        self, pages: list[PageSearchOutput], district: District, districts: list[District], term: str, town: str
    ):
        async def worker(page):
            return (
                page,
                await run_extraction_prompt(self.model_name, district, term, page.text),
            )

        for page, result in await gather(*map(worker, pages)):
            # logger.info(
            #     f"<MapExtractor>: search_pages_expanded: {flatten(page_coverage([page]))}"
            # )
            yield LookupOutput(
                output=result,
                search_pages=[page],
                search_pages_expanded=flatten(page_coverage([page])),
            )
