from pydantic import BaseModel

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
