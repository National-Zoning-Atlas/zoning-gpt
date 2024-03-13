import datasets
from manifest import Manifest
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, start_chain
import json
from zoning.utils import get_project_root

QUERY = "Districts. The town is divided into the following district zones: * residential (R2-0) * industrial (I-10) * rural overlay (T-190)"

data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")["train"]
manifest = Manifest(client_name = "openai",
                    cache_name = "sqlite",
                    cache_connection = "mycache.sqlite", max_tokens=1024)

def get_town_data(town):
    "Return a dataset for a town zoning code with embedding lookups"
    d = data.filter(lambda x: x["Town"] == town)
    return d

class KNNPrompt(EmbeddingPrompt):
    """
    Looks up k-nearest neighbors for query
    """
    def parse(self, out, inp):
        dataset, k = self.data
        query_embedding = np.array(out)        
        res = dataset.get_nearest_examples("embeddings", query_embedding, k)
        docs = res.examples["Text"]
        # print("pages", res.examples["Page"])
        return {"docs": docs, "pages": res.examples["Page"]}

class DistrictsPrompt(TemplatePrompt):
    template_file = "districts.pmpt.tpl"
    #template_file =  str((get_project_root() / "templates" / "districts.pmpt.tpl"))
    #template_file = "/Users/eeshakhanna/Desktop/nlp_zoning/zoning/templates/districts.pmpt.tpl"
    #template_file =  str((get_project_root() / "templates" / "districts_v2.pmpt.tpl"))

def get_districts(dataset):
    with start_chain("districts") as backend:
        # Find pages about districts
        prefix_dataset = dataset.select(range(min(50, len(dataset))))
        prefix_dataset.add_faiss_index("embeddings")
        prompt = KNNPrompt(backend.OpenAIEmbed(), (prefix_dataset, 4))
        #print("prompt", prompt)
        query = QUERY
        doc_list  = prompt(query)
        #import pdb; pdb.set_trace()
        # Extract district table
        dp = DistrictsPrompt(backend.Manifest(manifest))
        #districts = json.loads(dp({"docs": doc_list["docs"]}))
        districts_per_page = []
        for doc in doc_list["docs"]:
            page_districts = json.loads(dp(dict(docs = [doc])))
            districts_per_page.append(page_districts)
        
        # print(districts_per_page)
        
        # combine answers
        districts_combined = []
        [districts_combined.extend(l) for l in (districts_per_page[0], districts_per_page[1], districts_per_page[2], districts_per_page[3])]
        districts_filtered = []
        for x in districts_combined:
            if x not in districts_filtered:
                try:
                    int(x["Z"])
                except:
                    if x["T"] != x["Z"]:
                        districts_filtered.append(x)
        #import pdb; pdb.set_trace()
        # print(districts_filtered)

        pages = doc_list["pages"]
        #print("districts", districts)

    return districts_filtered, pages


if __name__ == "__main__":
    results_file = str(get_project_root() / "data" / "results" / "districts_test.jsonl")
    town_districts = {}
    for l in open(results_file):
        d = json.loads(l)
        town_districts[d["Town"]] = d["Districts"]
    #towns = set(data["Town"])
    towns = ["bristol", "southington", "hebron", "newington", "south-windsor", "warren", "morris", "east-haddam", "ellington",  "cheshire"]
    #towns = ["bristol"]

    with open(results_file, "w") as out:
        # print(towns)
        for town in towns:
            if town in town_districts:
                continue
            d = get_town_data(town)
            districts, pages = get_districts(d)
            # print(town)
            print(json.dumps({"Town": town, "Districts": districts, "Pages": pages}))
            print(json.dumps({"Town": town, "Districts": districts, "Pages": pages}), file=out)
        
