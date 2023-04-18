import json
from pathlib import Path

from minichain import OpenAI, prompt

from ..utils import get_project_root
from .search import nearest_pages

with Path(__file__).parent.joinpath("thesaurus.json").open(encoding="utf-8") as f:
    thesaurus = json.load(f)

extraction_tmpl = str(
    (get_project_root() / "templates" / "extraction.pmpt.tpl").relative_to(Path.cwd())
)


@prompt(OpenAI(), template_file=extraction_tmpl)
def lookup_term_prompt(model, page_text, district, term):
    return model(dict(
        passage=page_text,
        term=term,
        synonyms=" ,".join(thesaurus.get(term, [])),
        zone_name=district["T"],
        zone_abbreviation=district["Z"],
    ))


def main():
    districts_file = (
        get_project_root() / "data" / "results" / "districts_matched_2.jsonl"
    )
    results_file = get_project_root() / "data" / "results" / "sizes_test_4.jsonl"

    town_sizes = {}
    for l in results_file.open(encoding="utf-8").readlines():
        d = json.loads(l)
        town_sizes[d["Town"]] = d["Districts"]

    with results_file.open("a", encoding="utf-8") as out_f:
        for l in districts_file.open(encoding="utf-8").readlines():
            d = json.loads(l)
            town = d["Town"]
            out = {"Town": town, "Districts": []}

            for district in d["Districts"]:
                sizes = {}

                for term in ["min lot size", "min unit size"]:
                    pages = nearest_pages(town, district, term)

                    if len(pages) != 0:
                        pages = pages[: min(6, len(pages))]  # keep top-4 or less only
                        page_text = next(r[0] for r in pages)
                        page_number = pages[0][1]
                        lt = [
                            lookup_term_prompt(page_text, district, term).run(), # type: ignore
                            page_number,
                        ]
                    else:
                        lt = ["n/a", -1]
                    sizes[term] = lt
                    sizes[term + "_pages"] = [r[1] for r in pages]
                out["Districts"].append({"Name": district, "Sizes": sizes})
                break
            print(json.dumps(out), file=out_f)


if __name__ == "__main__":
    main()
