import json

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search

from ..types import District, PageSearchOutput
from .base import Searcher
from .utils import expand_term


class ElasticSearcher(Searcher):
    def __init__(self, k: int) -> None:
        self.client = Elasticsearch("http://localhost:9200")  # default client
        self.k = k 

    def search(self, town: str, district: District, term: str):
        # Search in town
        s = Search(using=self.client, index=town)

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
