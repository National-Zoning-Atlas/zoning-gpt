from abc import ABC, abstractmethod
from typing import AsyncGenerator

from ..types import District, PageSearchOutput, LookupOutput


class Extractor(ABC):
    @abstractmethod
    async def extract(
        self, pages: list[PageSearchOutput], district: District, districts: list[District], term: str, town: str

    ) -> AsyncGenerator[LookupOutput, None]:
        pass
