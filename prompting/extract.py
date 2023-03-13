from manifest import Manifest
from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain, Prompt
import json
from search import nearest_pages

manifest = Manifest(client_name = "openai",
                    cache_name = "sqlite",
                    cache_connection = "mycache.sqlite",
                    max_tokens=128)
thesaurus = json.load(open("thesaurus.json"))

class DistrictMinPrompt(TemplatePrompt):
    template_file = "extraction.pmpt.tpl"

def lookup_term(town, district, term="lot size"):
    out = nearest_pages(town, district, term)
    if len(out) == 0: return "n/a", -1
    page_text, page_number, highlight = out[0]
    print([o[1] for o in out])
    print(highlight)
    print(page_number)
    print(page_text)
    with start_chain("lookup") as backend:
        extract = DistrictMinPrompt(backend.Manifest(manifest))

        return extract({"passage": page_text,
                        "term": term,
                        "synonyms": " ,".join(thesaurus[term]),
                        "zone_name" : district["T"],
                        "zone_abbreviation" : district["Z"],
        }), page_number


    
if __name__ == "__main__":
    # town_districts = {}
    # for l in open("districts.jsonl"):
    #     d = json.loads(l)
    #     town_districts[d["Town"]] = d["Districts"]
    # towns = set(data["Town"])

    town_sizes = {}
    for l in open("sizes.jsonl"):
        d = json.loads(l)
        town_sizes[d["Town"]] = d["Districts"]

    with open("sizes.jsonl", "a") as out_f:
        for l in open("districts.jsonl"):
            d = json.loads(l)
            town = d["Town"]
            if town != "bethany": continue
            # if town in town_sizes:
            #     continue

            out = {"Town": town, "Districts": []}

            for district in d["Districts"]:
                sizes = {}                
                print(town)
                print(district)

                for term in ["min lot size", "min unit size"]:
                    print(term)
                    lt = lookup_term(town, district, term)
                    print(lt)
                    sizes[term] = lt
                out["Districts"].append({"Name": district, "Sizes": sizes})
                break 
            print(json.dumps(out), file=out_f)
