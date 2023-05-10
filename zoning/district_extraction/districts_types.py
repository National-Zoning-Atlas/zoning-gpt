import datasets
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain, Prompt
import json

data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")

d = data["train"].filter(lambda x: x["Town"] == "norfolk")
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

districts = get_districts(d)    
get_usages(d, districts)
