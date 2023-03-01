import datasets
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain, Prompt
import json
# import elasticsearch
# # es_client = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])
# d.add_elasticsearch_index("Text", "Text", es_client=es_client)

data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")

towns = set()
def collect(examples):
    towns.add(examples["Town"])
data["train"].map(collect)

print(towns)


# d = data["train"].filter(lambda x: x["Town"] == "north-haven")
# d = data["train"].filter(lambda x: x["Town"] == "essex")

d = data["train"].filter(lambda x: x["Town"] == "norfolk")
d.add_faiss_index("embeddings")

class KNNPrompt(EmbeddingPrompt):
    def parse(self, out, inp):
        res = self.data.get_nearest_examples("embeddings", np.array(out), 3)
        # for r in res.examples["Text"]:
        #     print(r)
        docs = [(t.replace("CELL (", f"CELL C{p} ("), p)
                for t, p in zip(res.examples["Text"],
                    res.examples["Page"])]
        return {"docs": docs}

class DistrictsPrompt(TemplatePrompt):
    template_file = "districts.pmpt.tpl"
    
def get_districts(d):
    with start_chain("districts") as backend:
        prompt = KNNPrompt(backend.OpenAIEmbed(), d).chain(DistrictsPrompt(backend.OpenAI(max_tokens=512 + 128)))
        result = prompt("table of contents residential commercial business zoning districts district below 1 family multi family permit")
        print("The Districts in the town are:", json.loads(result))
        
    return json.loads(result)
districts = get_districts(d)

# districts = json.loads("""[{"T": "Business", "Z": "Village Business"}, {"T": "Lodging Activities", "Z": "Village Residential"}, {"T": "Lodging Activities", "Z": "Neighborhood Residential"}, {"T": "Lodging Activities", "Z": "Rural Residential"}, {"T": "Business Activities", "Z": "Village Residential"}, {"T": "Business Activities", "Z": "Neighborhood Residential"}, {"T": "Business Activities", "Z": "Rural Residential"}]""")

class MultiFamilyPrompt(TemplatePrompt):
    template_file = "lot_area.pmpt.tpl"

def get_multifamily(d, districts):
    with start_chain("allowed") as backend:
        knn = KNNPrompt(backend.OpenAIEmbed(), d)
        prompt = MultiFamilyPrompt(backend.OpenAI(max_tokens=256))
        for x in districts:
    
            out = knn(f"Section XX and table about zoning district called {x['T']} with title {x['Z']} Minimum Lot Area Front Setback Maximum Building Height")
            result = prompt({"docs": out["docs"], "zone_name": x['T'], "zone_abbreviation": x['Z']})
            print("District", x['T'], result)

# get_multifamily(d, districts)

class UsagesPrompt(TemplatePrompt):
    template_file = "usages.pmpt.tpl"

# def get_usages(d, districts):
#     with start_chain("allowed") as backend:
#         knn = KNNPrompt(backend.OpenAIEmbed(), d)
#         prompt = UsagesPrompt(backend.OpenAI(max_tokens=256))
#         for x in districts:
#             out = knn(f"Section XX and table about zoning district allowed zoning principal permit family uses business special called {x['T']} with title {x['Z']}")
#             result = prompt({"docs": out["docs"], "zone_name": x['T'], "zone_abbreviation": x['Z']})
#             print("District", x['T'], result)
    
# get_usages(d, districts)

class KNNPrompt2(EmbeddingPrompt):
    def parse(self, out, inp):
        res = self.data.get_nearest_examples("embeddings", np.array(out), 10)
        # for r in res.examples["Text"]:
        #     print(r)
        docs = [(t.replace("CELL (", f"CELL (C{p},"), p)
                for t, p in zip(res.examples["Text"],
                    res.examples["Page"])]
        return {"docs": docs}


def get_usages(d, districts):
    with start_chain("allowed") as backend:
        distr = ", ".join([d["Z"] for d in districts] + [d["T"] for d in districts])
        knn = KNNPrompt2(backend.OpenAIEmbed(), d)
        out = knn(f"Table about homes residential zoning district allowed zoning principal permit multi single family dwelling uses business {distr}")
        prompt = UsagesPrompt(backend.OpenAI(max_tokens=512))
        for doc, page in out["docs"]:
            if page != 18: continue
            result = prompt({"docs": [(doc, page)], "districts": distr })
            print(f"Page {page}", result)
    
get_usages(d, districts)
