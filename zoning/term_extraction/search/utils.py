import warnings
from functools import cache
from typing import cast
from collections import Counter

import datasets
import pandas as pd
import tiktoken

from ..thesaurus import get_thesaurus
from ..types import PageSearchOutput


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


@cache
def get_lookup_tables() -> tuple[datasets.Dataset, pd.DataFrame]:
    ds = cast(
        datasets.Dataset,
        datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text", split="train+test"),
    )
    df = ds.to_pandas().set_index(["Town", "Page"])
    return ds, df


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
        warnings.warn(
            f"Page {page} was {len(enc.decode(enc.encode(text))) - max_token_length} tokens longer than the specified max token length of {max_token_length} and will be truncated."
        )

    return enc.decode(tokenized_text[:max_token_length])


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

def get_top_k_chunks(
    search_result: list[PageSearchOutput], k: int
) -> list[PageSearchOutput]:
    output = get_non_overlapping_chunks(search_result)
    output_indices_set = set([r.page_number for r in output])

    for res in search_result:
        if len(output) >= k:
            break
        if res.page_number not in output_indices_set:
            output.append(res)
            output_indices_set.add(res.page_number)

    return output


def naive_reranking(
    search_result_list: list[list[PageSearchOutput]]
) -> list[PageSearchOutput]:
    flattened_search_results = []
    page_search_dict = {}

    for search_result_sublist in search_result_list:
        for search_result in search_result_sublist:
            flattened_search_results.append(search_result)

            if search_result.page_number not in page_search_dict:
                page_search_dict[search_result.page_number] = search_result
                print(str(search_result.page_number) + ",")

        print("---")


    page_counts = Counter(search_result.page_number for search_result in flattened_search_results)
    sorted_pages = sorted(page_counts.keys(), key=lambda x: page_counts[x], reverse=True)

    res = []
    for page in sorted_pages:
        res.append(page_search_dict[page])
        print(str(page) + ",")
    return res
