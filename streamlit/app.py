import json
from os.path import dirname, realpath, join

import fitz
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

DIR = dirname(realpath(__file__))


@st.cache_data
def get_towns():
    with open(join(DIR, "../prompting/sizes.jsonl"), encoding="utf-8") as f:
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
    if page_results is None:
        return page_image

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
            format_func=lambda d: f"{d['Name']['Z']} ({d['Name']['T']})",
            index=0,
        )
        document_path = join(
            DIR, "../data/orig-documents", f"{town['Town']}-zoning-code.pdf"
        )

        if district is None:
            st.write("No districts with answers generated.")
            return

        sizes = district["Sizes"]

        def parse_size_answer(x):
            k, s = x
            a = s[0]
            page = s[1]
            if page == -1:
                # No answer was found to this question
                return {"type": k, "page": None, "reason": None, "answer": None}

            answer_lines = [l.strip() for l in a.split("*")]
            reasons = [l.replace("Reason:", "").strip() for l in answer_lines if l.startswith("Reason")]
            answers = [l for l in answer_lines if not l.startswith("Reason") and len(l) > 0]

            return {
                "type": k,
                "page": page,
                "answer": answers,
                "reason": reasons,
            }

        answers = pd.DataFrame(map(parse_size_answer, sizes.items()))

        st.caption(
            "The following pages were used to generate this answer:"
        )
        pages = answers["page"].dropna().unique().astype(int)
        pages.sort()
        if pages is None or len(pages) == 0:
            return
        elif len(pages) == 1:
            page = pages[0]
        else:
            page = st.select_slider(
                f"Page ({len(answers)} total)", options=pages
            )
        page_image = get_pdf_page_image(document_path, int(page))
        page_result = None

    st.header("Answer")
    st.dataframe(answers, use_container_width=True)

    st.header("Document")
    st.image(render_page_results(page_image, page_result), caption=f"Page {page}")


if __name__ == "__main__":
    main()
