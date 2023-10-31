import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search

from ..types import District, PageSearchOutput
from .base import Searcher
from .utils import expand_term


class ElasticSearcher(Searcher):
    def __init__(self, k: int, is_fuzzy = False) -> None:
        self.client = Elasticsearch("http://localhost:9200")  # default client
        self.k = k 
        self.is_fuzzy = is_fuzzy

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

        fuzzy_district_query = Q("match", Text={"query": district.short_name, "fuzziness": "AUTO"})

        if self.is_fuzzy:
            district_query = Q("bool", should=[exact_district_query, fuzzy_district_query])
        else:
            district_query = exact_district_query

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
        # ensure that we have a maximum of k results 
        s = s.extra(size=self.k) 
        
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
