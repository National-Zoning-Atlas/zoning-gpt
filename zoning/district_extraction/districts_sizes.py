import datasets
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, start_chain
import json

data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")

d = data["train"].filter(lambda x: x["Town"] == "madison")
d.add_faiss_index("embeddings")

class KNNPrompt(EmbeddingPrompt):
    def parse(self, out, inp):
        res = self.data.get_nearest_examples("embeddings", np.array(out), 3)
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

# districts = get_districts(d)
districts = [{"Z": "Affordable Housing Distrct", "T": "AHD"}]
get_multifamily(d, districts)
