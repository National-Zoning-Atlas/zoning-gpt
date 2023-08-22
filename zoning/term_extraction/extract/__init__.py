from enum import Enum
from typing import AsyncGenerator

from ..types import District, LookupOutput, PageSearchOutput
from .base import Extractor
from .dummy import DummyExtractor
from .map import MapExtractor
from .stuff import StuffExtractor
from .tournament_reduce import TournamentReduceExtractor


class ExtractionMethod(str, Enum):
    NONE = "none"
    STUFF = "stuff"
    MAP = "map"
    TOURNAMENT_REDUCE = "tournament_reduce"


async def extract_answer(
    pages: list[PageSearchOutput],
    term: str,
    district: District,
    method: ExtractionMethod,
    model_name: str,
    k: int = 1,
) -> AsyncGenerator[LookupOutput, None]:
    """
    Given a term to search for, will attempt to extract the value for the term
    from the provided pages.

    k is only used for extraction method TOURNAMENT_REDUCE, and specifies at
    what number of results to stop the tournament.
    """

    if len(pages) == 0:
        return

    extractor: Extractor
    match method:
        case ExtractionMethod.NONE:
            extractor = DummyExtractor()
        case ExtractionMethod.STUFF:
            extractor = StuffExtractor(model_name)
        case ExtractionMethod.TOURNAMENT_REDUCE:
            extractor = TournamentReduceExtractor(model_name, k)
        case ExtractionMethod.MAP:
            extractor = MapExtractor(model_name)

    async for result in extractor.extract(pages, district, term):
        yield result
