from typing import AsyncGenerator

from ...utils import flatten
from ..search.utils import page_coverage
from ..types import District, LookupOutput, PageSearchOutput
from .base import Extractor


class DummyExtractor(Extractor):
    async def extract(
        self, pages: list[PageSearchOutput], district: District, term: str
    ) -> AsyncGenerator[LookupOutput, None]:
        for page in pages:
            yield LookupOutput(
                output=None,
                search_pages=[page],
                search_pages_expanded=flatten(page_coverage([page])),
            )
