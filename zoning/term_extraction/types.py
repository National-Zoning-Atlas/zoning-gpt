from pydantic import BaseModel
from typing import Optional


class District(BaseModel):
    full_name: str
    short_name: str


class PageSearchOutput(BaseModel):
    text: str
    page_number: int
    highlight: list[str]
    score: float
    query: str


class ExtractionOutput(BaseModel):
    extracted_text: list[str]
    rationale: str
    answer: str


class LookupOutput(BaseModel):
    output: ExtractionOutput | None
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """


class RelevantContext(BaseModel):
    characters: int
    min_distance: int
    idx_first_match: Optional[int]
    idx_second_match: Optional[int]
    context: str
