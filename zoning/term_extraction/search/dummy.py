from typing import Generator

from ..types import District, PageSearchOutput
from .base import Searcher
from .utils import fill_to_token_length, get_lookup_tables


class DummySearcher(Searcher):
    def __init__(self):
        self.ds, self.df = get_lookup_tables()

    def search(
        self, town: str, district: District, term: str
    ) -> Generator[PageSearchOutput, None, None]:
        for x in iter(self.ds.filter(lambda x: x["Town"] == town)):
            yield PageSearchOutput(
                text=fill_to_token_length(x["Page"], self.df.loc[town], 1900),
                page_number=x["Page"],
                score=0,
                highlight=[],
                query="",
            )
