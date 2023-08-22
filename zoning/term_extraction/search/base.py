from abc import ABC, abstractmethod
from typing import Generator

from ..types import District, PageSearchOutput


class Searcher(ABC):
    @abstractmethod
    def search(
        self, town: str, district: District, term: str
    ) -> Generator[PageSearchOutput, None, None]:
        pass
