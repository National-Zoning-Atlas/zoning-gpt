from enum import StrEnum

from ..types import District
from .base import Searcher
from .dummy import DummySearcher
from .elasticsearch import ElasticSearcher
from .embeddings_knn import EmbeddingsKNNSearcher
from .utils import get_non_overlapping_chunks


class SearchMethod(StrEnum):
    NONE = "none"
    """Don't do anything; just return all pages for this town's zoning document."""
    ELASTICSEARCH = "elasticsearch"
    """Perform keyword-based search using ElasticSearch."""
    EMBEDDINGS_KNN = "embeddings_knn"
    """Perform semantic search using embeddings."""


def search_for_term(
    town: str,
    district: District,
    term: str,
    method: SearchMethod,
    k: int,
):
    searcher: Searcher
    match method:
        case SearchMethod.NONE:
            searcher = DummySearcher()
        case SearchMethod.ELASTICSEARCH:
            searcher = ElasticSearcher()
        case SearchMethod.EMBEDDINGS_KNN:
            # We grossly inflate the K we use for KNN to ensure that, even after
            # removing all overlapping pages, we have at least k pages leftover
            searcher = EmbeddingsKNNSearcher(k * 5)

    results = get_non_overlapping_chunks(list(searcher.search(town, district, term)))
    if k > 0:
        return results[:k]
    else:
        return results
