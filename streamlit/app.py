import json
from os.path import dirname, realpath, join
from typing import cast

import fitz
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

DIR = dirname(realpath(__file__))


def generate_sample_query_results(town: str, n: int, random_state: int):
    """
    Given a town, loads n random rows from our parquet dataset, groups the
    results by page, and generates example results using the sampled rows.

    Example:
    [{
        'Page': 20,
        'References': [{
            'Text': 'be',
            'Geometry': {
                'BoundingBox': {
                    'Width': 0.018624797463417053,
                    'Height': 0.011233381927013397,
                    'Left': 0.7655000686645508,
                    'Top': 0.4534885883331299},
                'Polygon': array([{'X': 0.7655000686645508, 'Y': 0.4534885883331299},
                        {'X': 0.7841235995292664, 'Y': 0.45349404215812683},
                        {'X': 0.7841248512268066, 'Y': 0.4647219777107239},
                        {'X': 0.7655012607574463, 'Y': 0.46471652388572693}], dtype=object)}}]},
        {'Page': 28,
        'References': [{'Text': 'the',
            'Geometry': {'BoundingBox': {'Width': 0.026048874482512474,
            'Height': 0.011076398193836212,
            'Left': 0.4396061599254608,
            'Top': 0.2604541480541229},
            'Polygon': array([{'X': 0.43961140513420105, 'Y': 0.2604541480541229},
                    {'X': 0.46565502882003784, 'Y': 0.26048582792282104},
                    {'X': 0.46564990282058716, 'Y': 0.27153053879737854},
                    {'X': 0.4396061599254608, 'Y': 0.27149882912635803}], dtype=object)}}]},
        ...
    }]
    """
    df = pd.read_parquet(join(DIR, "../data/parquet_dataset", f"{town}.parquet"))
    gb = df.sample(n, random_state=random_state)[["Page", "Text", "Geometry"]].groupby(
        "Page"
    )

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


@st.cache_data
def draw_rect_onto_image(
    base_image: Image.Image, left: float, top: float, width: float, height: float, color
) -> Image.Image:
    """Given a base image, draw a rectangle at the desired location on the
    image."""
    draw = ImageDraw.Draw(base_image)
    draw.rectangle((left, top, left + width, top + height), fill=color)

    return base_image


def render_page_results(page_image: Image.Image, page_results: dict) -> Image.Image:
    """For a given page image, renders the bounding boxes for all results onto
    the page."""
    draw = ImageDraw.Draw(page_image, "RGBA")
    for ref in page_results["References"]:
        bbox = ref["Geometry"]["BoundingBox"]
        left = bbox["Left"]
        top = bbox["Top"]
        width = bbox["Width"]
        height = bbox["Height"]
        draw.rectangle(
            (
                left * page_image.width,
                top * page_image.height,
                (left + width) * page_image.width,
                (top + height) * page_image.height,
            ),
            fill=(255, 0, 255, 128),
        )

    return page_image


def main():
    with st.sidebar:
        st.title("Warpspeed Document QA")
        st.header("Inputs")
        town = cast(str, st.selectbox(label="Town", options=get_towns(), index=0))
        results = generate_sample_query_results(town, 25, 42)
        document_path = join(DIR, "../data/orig-documents", f"{town}-zoning-code.pdf")

        query = st.text_area(
            label="Query",
            value="Are accessory dwelling units allowed in R-1 residential districts?",
            help="What do you want to ask about your document?",
        )
        generated_answer = "Yes, accessory dwelling units are allowed in R-1 residential districts."
        st.header("Answer")
        st.write(f":blue[{generated_answer}]")

        st.header("Diagnosis")
        st.caption("The following pages and extracted text were used to generate this answer:")
        page = st.select_slider(f"Page ({len(results)} total)", options=set(r["Page"] for r in results))
        page_image = get_pdf_page_image(document_path, page - 1)
        page_result = next(r for r in results if r["Page"] == page)


        st.table(({f"Text Extracted from Page {page}": t["Text"]} for t in page_result["References"]))

    st.header("Document")

    st.image(render_page_results(page_image, page_result), caption=f"Page {page}")


if __name__ == "__main__":
    main()
