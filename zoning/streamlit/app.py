import json
from os.path import dirname, realpath, join

import fitz
from PIL import Image, ImageDraw
import streamlit as st

from zoning.prompting.search import nearest_pages
from zoning.prompting.extract import lookup_term

DIR = dirname(realpath(__file__))

@st.cache_data
def get_towns():
    with open(join(DIR, "../prompting/districts_matched.jsonl"), encoding="utf-8") as f:
        json_lines = (json.loads(l) for l in f.readlines())
        return sorted(json_lines, key=lambda l: l["Town"])


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

# TODO: Use guardrails for this
def parse_size_answer(a):
    answer_lines = [l.strip() for l in a.split("*")]
    reasons = [l.replace("Reason:", "").strip() for l in answer_lines if l.startswith("Reason")]
    answers = [l for l in answer_lines if not l.startswith("Reason") and len(l) > 0]

    return {
        "answer": answers,
        "reason": reasons,
    }


def main():
    with st.sidebar:
        st.title("Warpspeed Document QA")
        st.header("Inputs")
        town_district_data = get_towns()
        town = st.selectbox(
            label="Town",
            options=town_district_data,
            format_func=lambda t: t["Town"],
            index=0,
        )
        district = st.selectbox(
            label="District",
            options=town["Districts"],
            format_func=lambda d: f"{d['Z']} ({d['T']})",
            index=0,
        )
        document_path = join(
            DIR, "../../data/orig-documents", f"{town['Town']}-zoning-code.pdf"
        )

        if district is None:
            st.write("No districts with answers generated.")
            return

        term = st.text_input("Search Term", "min lot size")
        
        pages = nearest_pages(town["Town"], district, term)

        page_num = st.select_slider(
            f"Page ({len(pages)} total)", options=[page_num for _, page_num, _ in pages]
        )
        page_text = next(r[0] for r in pages if r[1] == page_num)

        result = lookup_term(page_text, district, term)
        result = parse_size_answer(result)

        st.header("Answer")
        st.table(result["answer"])

        st.header("Reason")
        st.caption(
            "The following content was used to generate this answer:"
        )
        st.table(result["reason"])

    page_image = get_pdf_page_image(document_path, page_num)
    page_result = None

    st.header("Document")
    if page_result is not None:
        st.image(render_page_results(page_image, page_result), caption=f"Page {page_num}")
    else:
        st.image(page_image, caption=f"Page {page_num}")


if __name__ == "__main__":
    main()
