# -*- coding: utf-8 -*-
"""
@Time : 3/27/2024 12:48 AM
@Auth : Wang Yuyang
@File : index_towns_test.py
@IDE  : PyCharm
"""
from typing import cast

from datasets import load_from_disk, DatasetDict, Dataset
from elasticsearch import Elasticsearch
import tiktoken
from tqdm.contrib.concurrent import thread_map

from zoning import get_project_root

DATA_ROOT = get_project_root() / "data"

input_hf_dataset_path = DATA_ROOT / "hf_text_dataset"

enc = tiktoken.encoding_for_model("text-davinci-003")


def get_town_data(data: Dataset, town: str):
    "Return a dataset for a town zoning code with embedding lookups"
    d = data.filter(lambda x: x["Town"] == town)
    return d


def index_dataset(d, index_name):
    file_text = ""
    print(f"Indexing {index_name}...")
    for page in d.index:
        text = ""
        # Include 10 pages of forward context in the index
        for j in range(1):
            if page + j not in d.index:
                continue
            text += f"\nNEW PAGE {page + j - 1}\n" + d.loc[page + j]["Text"]

        # Truncate to 2000 tokens
        text = enc.decode(enc.encode(text)[:2500])
        file_text += text
    with open(f"text_data/{index_name}.txt", "w", encoding='utf8') as f:
        f.write(file_text)

def main(st=None):
    ds = cast(DatasetDict, load_from_disk(input_hf_dataset_path))

    for split in ds.keys():
        print(f"Processing {split} split...")
        df = ds[split].to_pandas().set_index(["Town", "Page"])
        if st:
            st.write(f"Processing {split} split...")
            st.write(df)
        towns = set(df.index.get_level_values(0))
        # thread_map(lambda town: index_dataset(df.loc[town], town), towns)
        for town in towns:
            index_dataset(df.loc[town], town)


# Make index for every town.
if __name__ == "__main__":
    main()
