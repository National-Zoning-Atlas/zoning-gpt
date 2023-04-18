import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from pydantic import BaseModel

from ..utils import get_project_root

es = Elasticsearch("http://localhost:9200")  # default client

# Global thesaurus
with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

class PageSearchOutput(BaseModel):
    text: str
    page_number: int
    highlight: list[str]

def nearest_pages(town, district, term="min lot size") -> list[PageSearchOutput]:
    # Search in town
    s = Search(using=es, index=town)

    # Search for district
    district_query = (
        Q("match_phrase", Text=district["T"])
        | Q("match_phrase", Text=district["Z"])
        | Q("match_phrase", Text=district["Z"].replace("-", ""))
        | Q("match_phrase", Text=district["Z"].replace(".", ""))
    )

    # Search for term
    term_expansion = [
        Q("match_phrase", Text=query.replace("min", r))
        for query in thesaurus.get(term, [])
        for r in ["min", "minimum", "min."]
    ]

    term_query = Q(
        "bool",
        should=term_expansion,
        minimum_should_match=1,
    )
    dim_query = Q(
        "bool",
        should=[Q("match_phrase", Text=d) for d in thesaurus["dimensions"]],
        minimum_should_match=1,
    )
    # s.query = cell_query
    s.query = district_query & term_query & dim_query
    s = s.highlight("Text")
    res = s.execute()
    return [PageSearchOutput(text=r.Text, page_number=r.Page, highlight=list(r.meta.highlight.Text)) for r in res]


def main():
    districts_file = get_project_root() / "data" / "results" / "districts_matched.jsonl"
    for l in districts_file.open(encoding="utf-8").readlines():
        d = json.loads(l)
        town = d["Town"]
        for district in d["Districts"]:
            print(town)
            print(district)
            print(nearest_pages(town, district))
            # nearest_pages(town, district)
            # break - to only perform search for first district, remove to do search for each district in the town



if __name__ == "__main__":
    main()