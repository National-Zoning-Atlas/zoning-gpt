import datasets
from manifest import Manifest
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain, Prompt
import json

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
        print(res.examples["Page"])
        return {"docs": docs}

class DistrictsPrompt(TemplatePrompt):
    template_file = "districts.pmpt.tpl"

    
def get_districts(dataset):
    with start_chain("districts") as backend:
        # Find pages about districts
        prefix_dataset = dataset.select(range(min(50, len(dataset))))
        prefix_dataset.add_faiss_index("embeddings")
        prompt = KNNPrompt(backend.OpenAIEmbed(), (prefix_dataset, 4))
        query = QUERY
        doc_list  = prompt(query)
        # Extract district table
        dp = DistrictsPrompt(backend.Manifest(manifest))
        districts = json.loads(dp(doc_list))
    return districts


if __name__ == "__main__":
    town_districts = {}
    for l in open("districts.jsonl"):
        d = json.loads(l)
        town_districts[d["Town"]] = d["Districts"]
    towns = set(data["Town"])

    with open("districts.jsonl", "w") as out:
        print(towns)
        for town in towns:
            if town in town_districts:
                continue
            d = get_town_data(town)
            districts = get_districts(d)
            print(json.dumps({"Town": town, "Districts": districts}))
            print(json.dumps({"Town": town, "Districts": districts}), file=out)
        
