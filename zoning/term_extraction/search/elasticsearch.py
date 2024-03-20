import json
import os
import warnings

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search

from ..types import District, PageSearchOutput
from .base import Searcher
from .utils import expand_term
from ...utils import logger


class ElasticSearcher(Searcher):
    def __init__(self, k: int, is_district_fuzzy: bool = False,
                 is_term_fuzzy: bool = False, label: str = "") -> None:
        if os.environ.get("ELASTICSEARCH_URL"):
            self.client = Elasticsearch(os.environ.get("ELASTICSEARCH_URL"))
        else:
            self.client = Elasticsearch("http://localhost:9200")  # default client
        self.k = k
        self.is_district_fuzzy = is_district_fuzzy
        self.is_term_fuzzy = is_term_fuzzy
        self.label = label

    def search(self, town: str, district: District, term: str):
        # Search in town
        s = Search(using=self.client, index=town)

        # Boost factor: Increasing the boost value will make documents matching this query to be ranked higher
        # Reference to Fuzzy: https://blog.mikemccandless.com/2011/03/lucenes-fuzzyquery-is-100-times-faster.html
        boost_value = 1.0

        exact_district_query = (
                Q("match_phrase", Text={"query": district.full_name, "boost": boost_value})
                | Q("match_phrase", Text={"query": district.short_name, "boost": boost_value})
                | Q("match_phrase", Text={"query": district.short_name.replace("-", ""), "boost": boost_value})
                | Q("match_phrase", Text={"query": district.short_name.replace(".", ""), "boost": boost_value})
        )

        fuzzy_district_query = (Q("match", Text={"query": district.short_name, "fuzziness": "AUTO"})
                                | Q("match", Text={"query": district.full_name, "fuzziness": "AUTO"})
                                )

        if self.is_district_fuzzy:
            district_query = Q("bool", should=[exact_district_query, fuzzy_district_query])
        else:
            district_query = exact_district_query
        expanded_term = expand_term(term)
        exact_term_query = Q(
            "bool",
            should=list(Q("match_phrase", Text=t) for t in expanded_term),
            minimum_should_match=1,
        )

        if self.is_term_fuzzy:
            term_query = Q(
                "bool",
                should=[
                           Q("match_phrase", Text=t) for t in expanded_term
                       ] + [
                           Q("match", Text={"query": t, "fuzziness": "AUTO"}) for t in expanded_term
                       ],
                minimum_should_match=1,
            )
        else:
            term_query = exact_term_query
        dimensions_expanded_term = expand_term(f"{term} dimensions")
        dim_query = Q(
            "bool",
            should=list(
                Q("match_phrase", Text=t) for t in dimensions_expanded_term
            ),
            minimum_should_match=1,
        )

        s.query = district_query & term_query & dim_query
        logger.info(f"Query: {s.query.to_dict()}")
        # ensure that we have a maximum of k results 
        s = s.extra(size=self.k)

        s = s.highlight("Text")

        res = s.execute()
        # print(res)
        if len(res) == 0:
            logger.warn(f"No results found for {term} in {town} {district.full_name}")

        yield from (
            PageSearchOutput(
                text=r.Text,
                page_number=r.Page,
                highlight=list(r.meta.highlight.Text),
                score=r.meta.score,
                log={"label": self.label},
                query=json.dumps(s.query.to_dict()),
            )
            for r in res
        )
