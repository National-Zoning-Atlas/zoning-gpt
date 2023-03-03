import datasets
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain, Prompt
import json

# Load data set
data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")

def get_town_data(town):
    "Return a dataset for a town zoning code with embedding lookups"
    d = data["train"].filter(lambda x: x["Town"] == town)
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
        prompt = KNNPrompt(backend.OpenAIEmbed(), (prefix_dataset, 3))
        query = "Districts. The town is divided into the following district zones: * residential (R2-0) * industrial (I-10) * rural overlay (T-190)"
        doc_list  = prompt(query)
        # print(doc_list)
        # extract district table
        dp = DistrictsPrompt(backend.OpenAI(max_tokens=1024))
        
        districts = json.loads(dp(doc_list))
        print("The Districts in the town are:", districts)
    return districts

# Run an additional query of the name of the district and zone
from numpy import dot
from numpy.linalg import norm

class EmbedPrompt(EmbeddingPrompt):
    def parse(self, out, inp):
        return out

def cos_sim(a,b):
    return a @ b / (norm(a, axis=1) *norm(b))

def nearest_pages(d, districts, town):
    with start_chain("lookup") as backend:
        knn = EmbedPrompt(backend.OpenAIEmbed())
        for x in districts: 
            out = knn(f"{x['T']} {x['Z']}")
            def dist(batch):
                return {f"district {x['Z']}":
                        cos_sim(np.array(batch["embeddings"]), np.array(out))}
            d = d.map(dist, batched=True, load_from_cache_file=False)
            closest = np.flip(np.argsort(np.array(d[f"district {x['Z']}"]))[-10:])
            x["closest"] = closest
            x["Zone"] = x["T"]
            x["ZoneAbbrev"] = x["Z"]
            del x["T"]
            del x["Z"]
            x["Town"] = town
            print(x)

t = list(set(data["train"]["Town"]))
# t = ["middlebury"]
print(t)

all_districts = []
town_districts = {}

for town in t[:20]:
    print("Town", town)
    dataset = get_town_data(town)
    town_districts[town] = get_districts(dataset)
    print(town, town_districts[town])

for town in t[:20]:
    # dataset = get_town_data(town)
    # nearest_pages(dataset, town_districts[town], town)
    all_districts += town_districts[town]
    
new_dataset = datasets.Dataset.from_list(all_districts)
new_dataset.save_to_disk("districts")
new_dataset.push_to_hub("xyzNLP/nza-ct-zoning-codes-district")
