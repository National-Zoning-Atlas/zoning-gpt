import fitz
import json
import tiktoken
import asyncio
from ...utils import get_project_root, get_jinja_environment
from ...prompting import prompt

enc = tiktoken.encoding_for_model("text-davinci-003")

create_thesaurus_tmpl = get_jinja_environment().get_template(
    "create_thesaurus.pmpt.tpl"
)

async def create_thesaurus(term: str, page_content: str):
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
                    "content": f"Input: \n\n {page_content}\n\n Output:",
                },
            ],
        max_tokens=2500,
    )
    print(text)
    

async def extract_text_from_page():
    zoning_atlas = {}
    text = ""
    DOCUMENT_PATH = get_project_root() / "zoning/term_extraction/thesaurus"
    doc = fitz.open(DOCUMENT_PATH / "Zoning_Atlas.pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        zoning_atlas[page_num + 1] = text

    with open(DOCUMENT_PATH / "zoning_atlas.json", "w") as f:
        json.dump(zoning_atlas, f)

    with open(DOCUMENT_PATH / "zoning_atlas.json") as f:
        data = json.load(f)
        #chunk_size = 8000
        data = data["63"] + data["64"]
        #chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        #for data_input in chunks:
        await create_thesaurus("max_height", data)


async def main(st=None):
    await extract_text_from_page()


if __name__ == "__main__":
    asyncio.run(main())
