from enum import Enum
from typing import AsyncGenerator, Coroutine

from ..types import District, LookupOutput, PageSearchOutput
from .base import Extractor
from .dummy import DummyExtractor
from .map import MapExtractor
from .stuff import StuffExtractor
from .tournament_reduce import TournamentReduceExtractor
from .multiple_choice import MultipleChoiceExtractor
from .answer_confirm import ConfirmExtractor


class ExtractionMethod(str, Enum):
    NONE = "none"
    STUFF = "stuff"
    MAP = "map"
    TOURNAMENT_REDUCE = "tournament_reduce"
    MULTIPLE_CHOICE = "multiple_choice"
    REDUCE_AND_CONFIRM = "answer_confirm"


async def extract_answer(
    pages: list[PageSearchOutput],
    term: str,
    town: str,
    district: District,
    method: ExtractionMethod,
    model_name: str,
    tournament_k: int = 1,
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
            extractor = TournamentReduceExtractor(model_name, tournament_k)
        case ExtractionMethod.MULTIPLE_CHOICE:
            extractor = MultipleChoiceExtractor(model_name, tournament_k)
        case ExtractionMethod.MAP:
            extractor = MapExtractor(model_name)
        case ExtractionMethod.REDUCE_AND_CONFIRM:
            extractor = ConfirmExtractor(model_name, tournament_k)
        case _:
            extractor = DummyExtractor()
    async for result in extractor.extract(pages, district, term, town):
        yield result
