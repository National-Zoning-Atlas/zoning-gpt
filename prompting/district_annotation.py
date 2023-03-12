import datasets
from manifest import Manifest
import numpy as np
from minichain import EmbeddingPrompt, TemplatePrompt, show_log, start_chain, Prompt
import json
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

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
        prompt = KNNPrompt(backend.OpenAIEmbed(), (prefix_dataset, 4))
        query = "Districts. The town is divided into the following district zones: * residential (R2-0) * industrial (I-10) * rural overlay (T-190)"
        doc_list  = prompt(query)
        # print(doc_list)
        # extract district table
        dp = DistrictsPrompt(backend.Manifest(Manifest(client_name = "openai",
                                                     cache_name = "sqlite",
                                                       cache_connection = "mycache.sqlite", max_tokens=1024)))
                           # (prefix_dataset, 3))#backend.OpenAI(max_tokens=1024))
        
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



es = Elasticsearch("http://localhost:9200")  # default client
es_config = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
    },
    "mappings": {"properties": {"text": {"type": "text", "analyzer": "standard", "similarity": "BM25"}}},
}

# print(resp['result'])

# resp = es.get(index="test-index", id=1)
# print(resp['_source'])

def index_dataset(d, town):
    ls = d.to_dict()
    for i in range(len(ls["Text"])):
        page = ls["Page"][i]
        text = ""
        for j in range(10):
            if i + j >= len(ls["Text"]): break
            text += f"\nNEW PAGE {i+j}\n" + ls["Text"][i + j]
        # crop to 2000ish tokens
        text = text[:1000 * 4]
        # if page == 8:
        #     print(text)
        es.index(index=town, id=page,
                 document={"Page": page, "Text": text})
    # d.map(index, with_indices=True, load_from_cache_file=False)

class DistrictMinPrompt(TemplatePrompt):
    template_file = "district_min_lot.pmpt.tpl"

    
def nearest_pages(d, districts, town):
    thesaurus = json.load(open("thesaurus.json"))
    
    with start_chain("lookup") as backend:
        d =d.remove_columns(["embeddings"])
        d_prompt = DistrictMinPrompt(
            backend.Manifest(Manifest(client_name = "openai",
                                      cache_name = "sqlite",
                                      cache_connection = "mycache.sqlite",
                                      max_tokens=512)))
        
        index_dataset(d, town)
        for x in districts:
            # if x["Z"] != "P.R.D.": continue
            s = Search(using=es, index=town)
            q = (Q("match_phrase", Text=x['T']) |
                 Q("match_phrase", Text=x['Z']) |
                 Q("match_phrase", Text=x['Z'].replace("-", "")) |
                 Q("match_phrase", Text=x['Z'].replace(".", ""))
                 )
            should = [Q("match", Text=query.replace("min", r)) for query
                         in thesaurus["lot size"]
                      for r in ["min", "minimum", "min."]]
            q2 = Q('bool',
                  should=should,
                  minimum_should_match=1,
                  )
            q = q & q2
            s.query = q
            res = s.execute()

            out = d_prompt({"passage": res[0].Text,
                            "synonyms": " ,".join(thesaurus["lot size"]),
                            "zone_name" : x["T"],
                            "zone_abbreviation" : x["Z"],
                            })
            print(out)
            x["out"] = out
            x["closest"] = np.array([h.Page for h in res], dtype=int)
            x["Zone"] = x["T"]
            x["ZoneAbbrev"] = x["Z"]
            del x["T"]
            del x["Z"]
            x["Town"] = town
            print(x)

t = list(set(data["train"]["Town"]))
t = ["fairfield"]
print(t)

all_districts = []
town_districts = {}

for town in t[:5]:
    print("Town", town)
    dataset = get_town_data(town)
    town_districts[town] = get_districts(dataset)
    print(town, town_districts[town])
    nearest_pages(dataset, town_districts[town], town)
    all_districts += town_districts[town]

# for town in t[:20]:
    # dataset = get_town_data(town)
    
new_dataset = datasets.Dataset.from_list(all_districts)
new_dataset.save_to_disk("districts")
# new_dataset.push_to_hub("xyzNLP/nza-ct-zoning-codes-district")
