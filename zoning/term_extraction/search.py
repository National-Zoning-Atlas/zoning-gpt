from functools import cache
import json
from pathlib import Path
from enum import Enum
import pandas as pd

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from pydantic import BaseModel
import datasets
from openai.embeddings_utils import get_embedding
import numpy as np

from .types import District

@cache
def get_elasticsearch_client():
    return Elasticsearch("http://localhost:9200")  # default client


@cache
def get_thesaurus() -> dict[str, list[str]]:
    # Global thesaurus
    with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
        return json.load(f)


class PageSearchOutput(BaseModel):
    text: str
    page_number: int
    highlight: list[str]
    score: float
    query: str


class SearchMethod(str, Enum):
    ELASTICSEARCH = "elasticsearch"
    EMBEDDINGS_KNN = "embeddings_knn"


def expand_term(term: str):
    min_variations = get_thesaurus().get("min", [])
    max_variations = get_thesaurus().get("max", [])
    for query in get_thesaurus().get(term, []):
        if "min" in query or "minimum" in query:
            for r in min_variations:
                yield query.replace("min", r)
        elif "max" in query or "maximum" in query:
            for r in max_variations:
                yield query.replace("max", r)
        else:
            yield query


def nearest_pages(
    town: str,
    district: District,
    term: str,
    method: SearchMethod = SearchMethod.ELASTICSEARCH,
):
    match method:
        case SearchMethod.ELASTICSEARCH:
            # Search in town
            s = Search(using=get_elasticsearch_client(), index=town)

            # Search for district
            district_query = (
                Q("match_phrase", Text=district.full_name)
                | Q("match_phrase", Text=district.short_name)
                | Q("match_phrase", Text=district.short_name.replace("-", ""))
                | Q("match_phrase", Text=district.short_name.replace(".", ""))
            )

            term_query = Q(
                "bool",
                should=list(Q("match_phrase", Text=t) for t in expand_term(term)),
                minimum_should_match=1,
            )

            dim_query = Q(
                "bool",
                should=list(Q("match_phrase", Text=t) for t in expand_term(f"{term} dimensions")),
                minimum_should_match=1,
            )

            s.query = district_query & term_query & dim_query

            s = s.highlight("Text")
            res = s.execute()

            yield from (
                PageSearchOutput(
                    text=r.Text,
                    page_number=r.Page,
                    highlight=list(r.meta.highlight.Text),
                    score=r.meta.score,
                    query=json.dumps(s.query.to_dict()),
                )
                for r in res
            )
        case SearchMethod.EMBEDDINGS_KNN:
            k = 6
            ds = datasets.load_dataset(
                "xyzNLP/nza-ct-zoning-codes-text", split="train+test"
            )
            query = " ".join(expand_term(term))
            query_embedding = np.array(
                get_embedding(query, "text-embedding-ada-002")
            )
            result = (
                ds.filter(lambda x: x["Town"] == town)
                .add_faiss_index("embeddings")
                .get_nearest_examples("embeddings", query_embedding, k)
            )
            for i in range(k):
                yield PageSearchOutput(
                    text=result.examples["Text"][i],
                    page_number=result.examples["Page"][i],
                    score=result.scores[i],
                    highlight=[],
                    query=query,
                )


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
