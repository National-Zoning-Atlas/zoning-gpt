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
    #template_file = str((get_project_root() / "templates" / "extraction.pmpt.tpl").relative_to(Path.cwd()))
    template_file = "extraction.pmpt.tpl"

def lookup_term(page_text, district, term):    
    with start_chain("lookup") as backend:
        extract = DistrictMinPrompt(backend.Manifest(manifest))

        return extract({"passage": page_text,
                        "term": term,
                        "synonyms": " ,".join(thesaurus.get(term, [])),
                        "zone_name" : district["T"],
                        "zone_abbreviation" : district["Z"],
        })


    
if __name__ == "__main__":
    districts_file = str(get_project_root() / "data" / "results" / "districts_matched_2.jsonl")
    results_file =  str(get_project_root() / "data" / "results" / "sizes_test_4.jsonl")
    # town_districts = {}
    # for l in open("districts.jsonl"):
    #     d = json.loads(l)
    #     town_districts[d["Town"]] = d["Districts"]
    # towns = set(data["Town"])

    town_sizes = {}
    for l in open(results_file):
        d = json.loads(l)
        town_sizes[d["Town"]] = d["Districts"]

    with open(results_file, "a") as out_f:
        for l in open(districts_file):
            d = json.loads(l)
            town = d["Town"]
            print(town)
            out = {"Town": town, "Districts": []}

            for district in d["Districts"]:
                sizes = {}                
                print(town)
                print(district)

                for term in ["min lot size", "min unit size"]:
                    pages = nearest_pages(town, district, term)
                    print("PAGES CHECK", [r[1] for r in pages])
                    #import pdb; pdb.set_trace()
                    if (len(pages) != 0):
                        pages = pages[:min(6, len(pages))] #keep top-4 or less only
                        page_text = next(r[0] for r in pages)
                        print(term)
                        page_number = pages[0][1]
                        lt = [lookup_term(page_text, district, term), page_number]
                    else:
                        lt = ["n/a", -1]
                        #continue
                    print(lt)
                    sizes[term] = lt
                    sizes[term + "_pages"] = [r[1] for r in pages]
                out["Districts"].append({"Name": district, "Sizes": sizes})
                break 
            print(json.dumps(out), file=out_f)
