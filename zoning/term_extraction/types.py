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
    log: dict[str, str]


class ExtractionOutput(BaseModel):
    extracted_text: list[str]
    rationale: str
    answer: str

    def __str__(self):
        return f"ExtractionOutput(extracted_text={self.extracted_text}, rationale={self.rationale}, answer={self.answer})"


class LookupOutput(BaseModel):
    output: ExtractionOutput | None
    search_pages: list[PageSearchOutput]
    search_pages_expanded: list[int]
    """
    The set of pages, in descending order or relevance, used to produce the
    result.
    """
    def __str__(self):
        return f"LookupOutput(output={self.output}, search_pages=[...], search_pages_expanded={self.search_pages_expanded})"


class LookupOutputConfirmed(LookupOutput):
    confirmed: bool
    confirmed_raw: str
    original_output: ExtractionOutput | None
    """
    The confirmed result by GPT.
    """


class RelevantContext(BaseModel):
    characters: int
    min_distance: int
    idx_first_match: Optional[int]
    idx_second_match: Optional[int]
    context: str
