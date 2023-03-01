import datasets
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
d = datasets.load_dataset("xyzNLP/nza-ct-zoning-codes")


@dataclass
class Entity:
    id : str
    text : str
    typ: str
    relationships: List[str]
    position: Tuple[int, int]

@dataclass
class Entities:
    ents : List[Entity]
    seen : Set[str]
    relations : Dict[str, List[Entity]] 
    def add(self, entity: Entity):
        if entity.id in self.seen: return
        self.ents.append(entity)
        self.seen.add(entity.id)
        for r in entity.relationships:
            self.relations.setdefault(r, [])
            self.relations[r].append(entity)
    
    def print(self) -> str:
        out = ""
        for e in self.ents:
            if e.typ == "LINE":
                in_cell = [o for r in e.relationships for o in self.relations[r] if o.typ == "CELL"]
                if not in_cell:
                    out += e.text + "\n"
            if e.typ == "CELL":
                lines = [o for r in e.relationships for o in self.relations[r] if o.typ == "LINE"]

                out += f"CELL {e.position}: \n"
                seen = set()
                for o in lines:
                    if o.id in seen: continue
                    seen.add(o.id)
                    out += o.text + "\n"
        return out

def collect_relations(w):
    rels = w["Relationships"]
    ids = []
    for r in (rels if rels else []):
        for id in r["Ids"]:
            ids.append(id)
    return ids

entities = Entities([], set(), {})
rows = [] 
for w in d["train"]:
    if w["BlockType"] in ["LINE", "WORD", "CELL", "MERGED_CELL"]:
        e = Entity(w["Id"], w.get("Text", ""), w["BlockType"], collect_relations(w), (w["RowIndex"], w["ColumnIndex"]))
        entities.add(e)
    elif w["BlockType"] == "PAGE":
        
        rows.append({"Town": w["Town"], "Page" : w["Page"], "Text": entities.print()})
        entities = Entities([], set(), {})
        print({"Town": w["Town"], "Page" : w["Page"]})
    elif w["BlockType"] == "TABLE":
        pass
    else:
        # print("\t" + w["BlockType"] + str(w))
        continue
    
