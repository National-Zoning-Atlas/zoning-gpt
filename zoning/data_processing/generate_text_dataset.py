from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, cast

import numpy as np
import openai
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm

from ..utils import get_project_root, load_pipeline_config

DATA_ROOT = get_project_root() / "data"
config = load_pipeline_config()

# Whether or not to publish the resulting dataset to HuggingFace Hub. If False,
# the dataset will be saved locally to disk instead.
PUBLISH_DATASET = config["publish_datasets"]

EMBEDDING_MODEL = "text-embedding-ada-002"

input_hf_dataset_path = DATA_ROOT / "hf_dataset"
input_hf_dataset_name = "xyzNLP/nza-ct-zoning-codes"
output_hf_dataset_name = "xyzNLP/nza-ct-zoning-codes-text"
output_hf_dataset_path = DATA_ROOT / "hf_text_dataset"


@dataclass
class Entity:
    id: str
    text: str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]


@dataclass
class Entities:
    ents: List[Entity]
    seen: Set[str]
    relations: Dict[str, List[Entity]]

    def add(self, entity: Entity):
        if entity.id in self.seen:
            return
        self.ents.append(entity)
        self.seen.add(entity.id)
        for r in entity.relationships:
            self.relations.setdefault(r, [])
            self.relations[r].append(entity)

    def __str__(self) -> str:
        out = ""
        for e in self.ents:
            if e.typ == "LINE":
                in_cell = [
                    o
                    for r in e.relationships
                    for o in self.relations[r]
                    if o.typ == "CELL"
                ]
                if not in_cell:
                    out += e.text + "\n"
            if e.typ == "CELL":
                lines = [
                    o
                    for r in e.relationships
                    for o in self.relations[r]
                    if o.typ == "LINE"
                ]

                out += f"CELL {e.position}: \n"
                seen = set()
                for o in lines:
                    if o.id in seen:
                        continue
                    seen.add(o.id)
                    out += o.text + "\n"
        return out


def collect_relations(w):
    rels = w["Relationships"]
    ids = []
    for r in rels if rels else []:
        for id in r["Ids"]:
            ids.append(id)
    return ids


def linearize(dataset: Dataset):
    entities = Entities([], set(), {})
    rows = []
    for w in tqdm(dataset):
        if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
            e = Entity(
                w["Id"],
                w.get("Text", ""),
                w["BlockType"],
                collect_relations(w),
                (w["RowIndex"], w["ColumnIndex"]),
            )
            entities.add(e)
        elif w["BlockType"] == "PAGE":
            rows.append(
                {"Town": w["Town"], "Page": w["Page"], "Text": str(entities)}
            )
            entities = Entities([], set(), {})
        elif w["BlockType"] == "TABLE":
            pass
        else:
            continue
    return Dataset.from_list(rows)


def embed(x):
    y = []
    for p in x["Text"]:
        if not p:
            p = " "
        if len(p.split()) > 3000:
            p = " ".join(p[:3000])
        y.append(p)
    emb = openai.Embedding.create(input=y, engine=EMBEDDING_MODEL)
    return {
        "embeddings": [
            np.array(emb["data"][i]["embedding"]) for i in range(len(emb["data"]))
        ]
    }


def main():
    ds = cast(DatasetDict, load_dataset(input_hf_dataset_name))

    new_ds = DatasetDict()
    for split in ds.keys():
        print(f"Processing {split} split...")
        new_ds[split] = linearize(ds[split]).map(embed, batch_size=100, batched=True)

    new_ds.save_to_disk(output_hf_dataset_path)

    if PUBLISH_DATASET:
        new_ds.push_to_hub(output_hf_dataset_name, private=True)

    
if __name__ == "__main__":
    main()
