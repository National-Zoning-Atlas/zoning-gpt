import json
from pathlib import Path

from manifest import Manifest
from minichain import TemplatePrompt, start_chain

from zoning.prompting.search import nearest_pages
from zoning.utils import get_project_root

manifest = Manifest(client_name = "openai",
                    cache_name = "sqlite",
                    cache_connection = "mycache.sqlite",
                    max_tokens=128)

with Path(__file__).parent.joinpath("thesaurus.json").open() as f:
    thesaurus = json.load(f)

class DistrictMinPrompt(TemplatePrompt):
    template_file = str((get_project_root() / "templates" / "extraction.pmpt.tpl").relative_to(Path.cwd()))

def lookup_term(page_text, district, term="lot size"):
    with start_chain("lookup") as backend:
        extract = DistrictMinPrompt(backend.Manifest(manifest))

        return extract({"passage": page_text,
                        "term": term,
                        "synonyms": " ,".join(thesaurus.get(term, [])),
                        "zone_name" : district["T"],
                        "zone_abbreviation" : district["Z"],
        })


    
if __name__ == "__main__":
    # town_districts = {}
    # for l in open("districts.jsonl"):
    #     d = json.loads(l)
    #     town_districts[d["Town"]] = d["Districts"]
    # towns = set(data["Town"])
    districts_file = "districts_matched.jsonl"
    results_file = "sizes_test.jsonl"

    # town_sizes = {}
    # for l in open(results_file):
    #     d = json.loads(l)
    #     town_sizes[d["Town"]] = d["Districts"]

    with open(results_file, "a") as out_f:
        for l in open(districts_file):
            d = json.loads(l)
            town = d["Town"]

            out = {"Town": town, "Districts": []}

            for district in d["Districts"]:
                sizes = {}                
                print(town)
                print(district)

                for term in ["min lot size", "min unit size"]:
                    pages = nearest_pages(town, district, term)
                    if (len(pages) == 0):
                        continue
                    page_text = next(r[0] for r in pages)
                    print(term)
                    lt = lookup_term(page_text, district, term)
                    print(lt)
                    sizes[term] = lt
                out["Districts"].append({"Name": district, "Sizes": sizes})
                break 
            print(json.dumps(out), file=out_f)
