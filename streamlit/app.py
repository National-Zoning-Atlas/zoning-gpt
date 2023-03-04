import json
from os.path import dirname, realpath, join
from typing import cast

import fitz
import pandas as pd
from PIL import Image
import streamlit as st

DIR = dirname(realpath(__file__))


def generate_sample_query_results(town: str, n: int, random_state: int):
    """
    Given a town, loads n random rows from our parquet dataset, groups the
    results by page, and generates example results using the sampled rows.
    """
    df = pd.read_parquet(join(DIR, "../data/parquet_dataset", f"{town}.parquet"))
    gb = df.sample(n, random_state=random_state)[["Page", "Text", "Geometry"]].groupby("Page")

    return [
        {"Page": p, "References": [dict(zip(v, t)) for t in zip(*v.values())]}
        for p, v in gb.agg(list).T.to_dict().items()
    ]


@st.cache_data
def get_towns():
    with open(join(DIR, "../data/names_all_towns.json")) as f:
        return json.load(f)


@st.cache_data
def get_pdf_page_image(pdf_path: str, page_num: int) -> Image.Image:
    """Given a PDF at `pdf_path`, load only the specified `page_num` as an Image."""
    dpi = 160
    zoom = dpi / 72
    magnify = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=magnify)
        return Image.frombytes("RGB", (pix.w, pix.h), pix.samples)

def main():
    st.title("Warpspeed Document QA")

    st.header("Inputs")

    town = cast(str, st.selectbox(label="Town", options=get_towns(), index=0))

    query = st.text_input(
        label="Query",
        value="The quick brown fox jumps over the lazy dog",
        help="What do you want to ask about your document?",
    )
    document_path = join(DIR, "../data/orig-documents", f"{town}-zoning-code.pdf")

    st.header("Results")
    results = generate_sample_query_results(town, 25, 42)
    result = results[0]
    page = result["Page"]
    st.image(get_pdf_page_image(document_path, page), caption=f"Page {page}")


if __name__ == "__main__":
    main()
