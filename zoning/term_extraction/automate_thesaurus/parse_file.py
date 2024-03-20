import fitz
import json
import tiktoken
import asyncio
from ...utils import get_project_root, get_jinja_environment
from ...prompting import prompt
from tqdm.asyncio import tqdm

enc = tiktoken.encoding_for_model("text-davinci-003")

create_thesaurus_tmpl = get_jinja_environment().get_template(
    "create_thesaurus.pmpt.tpl"
)
DOCUMENT_PATH = get_project_root() / "zoning/term_extraction/thesaurus"


async def process_chunk(index, data_input, term):
    unique_terms = set()
    text = await prompt(
        "gpt-4-turbo-preview",
        [
            {
                "role": "system",
                "content": create_thesaurus_tmpl.render(
                    term=term,
                ),
            },
            {
                "role": "user",
                "content": f"Input: \n\n {data_input}\n\n Output:",
            },
        ],
        max_tokens=2500,
    )

    if text is not None and text != "null":
        if text[:7] == "```json":
            text = text[7:-4]
        syn_data = json.loads(text)

        for term in syn_data.get(term, []):
            unique_terms.add(term.lower())

    return unique_terms


async def create_thesaurus(term: str, chunks):
    unique_terms = set()
    async for result in tqdm(
        asyncio.as_completed(
            [process_chunk(index, chunk, term) for index, chunk in enumerate(chunks)]
        ),
        total=len(chunks),
        desc="Processing chunks",
    ):
        unique_terms.update(await result)

    unique_terms_list = list(unique_terms)
    output_data = {term: unique_terms_list}
    with open(DOCUMENT_PATH / "thesaurus.json", "w") as json_file:
        json.dump(output_data, json_file, indent=2)


async def extract_text_from_page():
    zoning_atlas = {}
    text = ""
    doc = fitz.open(DOCUMENT_PATH / "Zoning_Atlas.pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        zoning_atlas[page_num + 1] = text

    with open(DOCUMENT_PATH / "zoning_atlas.json", "w") as f:
        json.dump(zoning_atlas, f)

    chunk_size = 2
    values = list(zoning_atlas.values())
    # chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    chunks = [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]

    await create_thesaurus("floor_to_area_ratio", chunks)


async def main(st=None):
    await extract_text_from_page()


if __name__ == "__main__":
    asyncio.run(main())
