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
    ELASTIC_AND_EMBEDDINGS = "elastic_and_embeddings"
    """Perform both ElasticSearch and Embeddings Search"""
    SEARCH_TOP_TEN = "search_top_ten"
    """Perform combination of Elastic Search and Embedding Search, but only return top 10 results"""


def search_for_term(
    town: str,
    district: District,
    term: str,
    method: SearchMethod,
    k: int,
):
    searcher: Searcher
    results: []
    match method:
        case SearchMethod.NONE:
            searcher = DummySearcher()
            return execute_search(searcher, town, district, term, k)
        case SearchMethod.ELASTICSEARCH:
            searcher = ElasticSearcher(k)
            return execute_search(searcher, town, district, term, k)
        case SearchMethod.EMBEDDINGS_KNN:
            # We grossly inflate the K we use for KNN to ensure that, even after
            # removing all overlapping pages, we have at least k pages leftover
            searcher = EmbeddingsKNNSearcher(k)
            return execute_search(searcher, town, district, term, k)
        case SearchMethod.ELASTIC_AND_EMBEDDINGS:
            # to reach optimal search results, run both elastic search and embeddings knn
            es_searcher = ElasticSearcher(k)
            embedding_searcher = EmbeddingsKNNSearcher(k)
            es_result = execute_search(es_searcher, town, district, term, k)
            embedding_result = execute_search(embedding_searcher, town, district, term, k)
            # TODO: this may return duplicate results. Could be good to filter them to reduce cost
            return es_result + embedding_result
        case SearchMethod.SEARCH_TOP_TEN:
            es_searcher_nf = ElasticSearcher(k)
            es_searcher_f = ElasticSearcher(k, True)
            embedding_searcher = EmbeddingsKNNSearcher(k)
            es_searcher_nf_res = execute_search(es_searcher_nf, town, district, term, k)
            es_searcher_f_res = execute_search(es_searcher_f, town, district, term, k)
            embedding_res = execute_search(embedding_searcher, town, district, term, k)
            res = get_non_overlapping_chunks(es_searcher_nf_res + es_searcher_f_res + embedding_res)
            return res[:k]


def execute_search(
        searcher: Searcher,
        town: str,
        district: District,
        term: str,
        k: int
): 
    results = get_non_overlapping_chunks(list(searcher.search(town, district, term)))
    if k > 0:
        return results[:k]
    else:
        return results
