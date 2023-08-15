import json
from enum import Enum
from functools import cache
from pathlib import Path
from typing import cast
import warnings

import datasets
import numpy as np
import pandas as pd
import tiktoken
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from openai.embeddings_utils import get_embedding
from pydantic import BaseModel

from .types import District


@cache
def get_elasticsearch_client():
    return Elasticsearch("http://localhost:9200")  # default client


@cache
def get_thesaurus() -> dict[str, list[str]]:
    # Global thesaurus
    with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
        return json.load(f)
    
@cache
def get_knn_lookup_tables(town: str) -> tuple[datasets.Dataset, pd.DataFrame]:
    ds = cast(
        datasets.Dataset,
        datasets.load_dataset(
            "xyzNLP/nza-ct-zoning-codes-text", split="train+test"
        ),
    )
    df = ds.to_pandas().set_index(["Town", "Page"]).loc[town]
    result = ds.filter(lambda x: x["Town"] == town).add_faiss_index("embeddings")
    return result, df

def fill_to_token_length(start_page, df, max_token_length):
    """
    Given a starting page in the document, add subsequent pages to the document
    until the desired token length is achieved, or until no subsequent pages can
    be found.
    """
    
    enc = tiktoken.encoding_for_model("text-davinci-003")
    page = start_page - 1
    last_page = max(df.index)
    text = ""
    while len(enc.encode(text)) < max_token_length and page <= last_page:
        page += 1
        if page in df.index:
            text += f"\nNEW PAGE {page - 1}\n" + df.loc[page]["Text"]

    tokenized_text = enc.encode(text)

    if page == start_page:
        warnings.warn(f"Page {page} was {len(enc.decode(enc.encode(text))) - max_token_length} tokens longer than the specified max token length of {max_token_length} and will be truncated.")

    return enc.decode(tokenized_text[:max_token_length])


class PageSearchOutput(BaseModel):
    text: str
    page_number: int
    highlight: list[str]
    score: float
    query: str


class SearchMethod(str, Enum):
    NO_SEARCH = "no_search"
    """Don't do anything; just return all pages for this town's zoning document."""
    ELASTICSEARCH = "elasticsearch"
    """Perform keyword-based search using ElasticSearch."""
    EMBEDDINGS_KNN = "embeddings_knn"
    """Perform semantic search using embeddings."""


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
        case SearchMethod.NO_SEARCH:
            ds, df = get_knn_lookup_tables(town)
            for x in iter(ds):
                yield PageSearchOutput(
                    text=fill_to_token_length(x["Page"], df, 1900),
                    page_number=x["Page"],
                    score=0,
                    highlight=[],
                    query="",
                )
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
                should=list(
                    Q("match_phrase", Text=t) for t in expand_term(f"{term} dimensions")
                ),
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
            query = next(expand_term(term))
            query_embedding = np.array(get_embedding(query, "text-embedding-ada-002"))
            ds, df = get_knn_lookup_tables(town)
            result = ds.get_nearest_examples("embeddings", query_embedding, k)
            for i in range(k):
                page = result.examples["Page"][i]
                yield PageSearchOutput(
                    text=fill_to_token_length(page, df, 2000),
                    page_number=page,
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
