import datasets
from elasticsearch import Elasticsearch
import tiktoken

enc = tiktoken.encoding_for_model("text-davinci-003")

# Load data set
train_data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")["train"]
test_data = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes-text")["test"]
es = Elasticsearch("http://localhost:9200")  # default client

def get_town_data(data, town):
    "Return a dataset for a town zoning code with embedding lookups"
    d = data.filter(lambda x: x["Town"] == town)
    return d


def index_dataset(d, town):
    index_name = town
    ls = d.to_dict()
    es.indices.delete(index=index_name, ignore=[400, 404])


    for i in range(len(ls["Text"])):
        page = ls["Page"][i]
        text = ""
        for j in range(10):
            if i + j >= len(ls["Text"]): break
            text += f"\nNEW PAGE {i+j}\n" + ls["Text"][i + j]
            
        # Truncate to 2000 tokens
        text = enc.decode(enc.encode(text)[:2000])
        es.index(index=index_name, id=page,
                 document={"Page": page, "Text": text})

def index_towns(ds):
    towns = set(ds["Town"])
    for town in towns:
        print(town)
        if es.indices.exists(index=town):
            print("Index already exists, skipping")
            continue
        d = get_town_data(ds, town)
        index_dataset(d, town)

# Make index for every town.
if __name__  == "__main__":
    index_towns(train_data)
    index_towns(test_data)
