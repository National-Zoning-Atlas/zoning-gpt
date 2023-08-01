import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.query import Query
from pydantic import BaseModel

import rich
from .types import District

es = Elasticsearch("http://localhost:9200")  # default client

# Global thesaurus
with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

class PageSearchOutput(BaseModel):
    text: str
    page_number: int
    highlight: list[str]
    score: float
    query: dict


def expand_term(term: str):
    min_variations = thesaurus.get("min", [])
    max_variations = thesaurus.get("max", [])
    for query in thesaurus.get(term, []):
        if "min" in query or "minimum" in query:
            for r in min_variations:
                yield Q("match_phrase", Text=query.replace("min", r))
        elif "max" in query or "maximum" in query:
            for r in max_variations:
                yield Q("match_phrase", Text=query.replace("max", r))
        else:
            yield Q("match_phrase", Text=query)


def nearest_pages(town: str, district: District, term: str) -> list[PageSearchOutput]:
    # Search in town
    s = Search(using=es, index=town)

    # Search for district
    district_query = (
        Q("match_phrase", Text=district.full_name)
        | Q("match_phrase", Text=district.short_name)
        | Q("match_phrase", Text=district.short_name.replace("-", ""))
        | Q("match_phrase", Text=district.short_name.replace(".", ""))
    )

    term_query = Q(
        "bool",
        should=list(expand_term(term)),
        minimum_should_match=1,
    )

    dim_query = Q(
        "bool",
        should=list(expand_term(f"{term} dimensions")),
        minimum_should_match=1,
    )

    s.query = district_query & term_query & dim_query

    s = s.highlight("Text")
    res = s.execute()
    
    return [
        PageSearchOutput(
            text=r.Text,
            page_number=r.Page,
            highlight=list(r.meta.highlight.Text),
            score=r.meta.score,
            query=s.query.to_dict()
        )
        for r in res
    ]


def page_coverage(search_result: list[PageSearchOutput]) -> list[list[int]]:
    pages_covered = []
    for r in search_result:
        chunks = r.text.split("NEW PAGE ")
        pages = []
        for chunk in chunks[1:]:
            page = chunk.split("\n")[0]
            pages.append(int(page))
        pages_covered.append(pages)
    return pages_covered


def get_non_overlapping_chunks(
    search_result: list[PageSearchOutput],
) -> list[PageSearchOutput]:
    indices = [r.page_number for r in search_result]
    pages_covered = page_coverage(search_result)
    non_overlapping_indices: list[int] = []
    non_overlapping_chunks: list[PageSearchOutput] = []
    for i, index in enumerate(indices):
        has_overlap = False
        current_pages = set(pages_covered[i])
        for prev_index in non_overlapping_indices:
            prev_pages = set(pages_covered[indices.index(prev_index)])
            if current_pages.intersection(prev_pages):
                has_overlap = True
                break
        if not has_overlap:
            non_overlapping_indices.append(index)
            non_overlapping_chunks.append(search_result[i])
    return non_overlapping_chunks
