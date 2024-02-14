from enum import StrEnum

from ..types import District
from .base import Searcher
from .dummy import DummySearcher
from .elasticsearch import ElasticSearcher
from .embeddings_knn import EmbeddingsKNNSearcher
from .utils import get_non_overlapping_chunks, naive_reranking, get_top_k_chunks


class SearchMethod(StrEnum):
    NONE = "none"
    """Don't do anything; just return all pages for this town's zoning document."""
    BASELINE = "baseline"
    """Perform baseline search using ES Fuzzy District and ES Fuzzy District Term"""
    EXPERIMENT_1 = "experiment_1"
    """ES, ES Fuzzy District and ES Fuzzy District Term"""
    EXPERIMENT_2 = "experiment_2"
    """ES, ES Fuzzy District and Embedding"""
    EXPERIMENT_3 = "experiment_3"
    """Double k ES, Double k ES Fuzzy District and ES Fuzzy District Term"""
    ELASTICSEARCH = "elasticsearch"
    """Perform keyword-based search using ElasticSearch."""
    ES_FUZZY = "es_fuzzy"
    """Perform ElasticSearch Fuzzy"""
    EMBEDDINGS_KNN = "embeddings_knn"
    """Perform semantic search using embeddings."""
    ELASTIC_AND_EMBEDDINGS = "elastic_and_embeddings"
    """Perform both ElasticSearch and Embeddings Search"""


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

        case SearchMethod.BASELINE:
            district_es_searcher = ElasticSearcher(k, True, False)
            district_term_es_searcher = ElasticSearcher(k, True, True)
            district_es_res = list(district_es_searcher.search(town, district, term))
            district_term_es_res = list(district_term_es_searcher.search(town, district, term))
            res = get_top_k_chunks((district_es_res + district_term_es_res), k)
            return res[:k]

        case SearchMethod.EXPERIMENT_1:
            es_searcher = ElasticSearcher(k, False, False)
            district_es_searcher = ElasticSearcher(k, True, False)
            district_term_es_searcher = ElasticSearcher(k, True, True)
            es_res = list(es_searcher.search(town, district, term))
            district_es_res = list(district_es_searcher.search(town, district, term))
            district_term_es_res = list(district_term_es_searcher.search(town, district, term))
            res = get_top_k_chunks((es_res + district_es_res + district_term_es_res), k)
            return res[:k]

        case SearchMethod.EXPERIMENT_2:
            es_searcher = ElasticSearcher(k, False, False)
            district_es_searcher = ElasticSearcher(k, True, False)
            embedding_searcher = EmbeddingsKNNSearcher(k)
            es_res = list(es_searcher.search(town, district, term))
            district_es_res = list(district_es_searcher.search(town, district, term))
            embedding_res = list(embedding_searcher.search(town, district, term))
            res = get_top_k_chunks((es_res + district_es_res + embedding_res), k)
            return res[:k]

        case SearchMethod.EXPERIMENT_3:
            es_searcher = ElasticSearcher(k * 2, False, False)
            district_es_searcher = ElasticSearcher(k * 2, True, False)
            district_term_es_searcher = ElasticSearcher(k, True, True)
            es_res = list(es_searcher.search(town, district, term))
            district_es_res = list(district_es_searcher.search(town, district, term))
            district_term_es_res = list(district_term_es_searcher.search(town, district, term))
            res = get_top_k_chunks((es_res + district_es_res + district_term_es_res), k)
            return res[:k]

        case SearchMethod.ELASTICSEARCH:
            searcher = ElasticSearcher(k)
            res = list(searcher.search(town, district, term))
            return get_non_overlapping_chunks(res)

        case SearchMethod.ES_FUZZY:
            searcher = ElasticSearcher(k, True, False)
            res = list(searcher.search(town, district, term))
            return get_non_overlapping_chunks(res)

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
