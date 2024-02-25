# -*- coding: utf-8 -*-
"""
@Time : 2/25/2024 2:19 PM
@Auth : Wang Yuyang
@File : chat.py
@IDE  : PyCharm
"""
import asyncio
import collections
import time
import numpy as np
import pandas as pd
import streamlit as st

from elasticsearch import Elasticsearch

import zoning
from zoning.term_extraction.extract import extract_answer, ExtractionMethod
from zoning.term_extraction.search import SearchMethod
from zoning.term_extraction.search.utils import page_coverage
from zoning.term_extraction.types import District
from zoning.utils import flatten

import asyncio


async def consume_async_gen(pages: list,
                            term: str,
                            town: str,
                            district: District,
                            method: ExtractionMethod,
                            model_name: str,
                            tournament_k: int = 1, ):
    async for result in extract_answer(
            pages=pages,
            term=term,
            town=town,
            district=district,
            method=method,
            model_name=model_name,  # getting better results with this
            tournament_k=tournament_k,
    ):

        if result.output is None:
            pass
        else:
            st_stream_write(f"---")
            st_stream_write(f"**extracted_text:** `{result.output.extracted_text}`")
            st_stream_write(f"**rationale:** `{result.output.rationale}`")
            st_stream_write(f"**actual:** `{result.output.answer}`")
            st_stream_write(f"---")

    print("All values consumed.")


def run_asyncio_coroutine(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result


def parse_input(input_str):
    """
    Parse the input string into a list of tokens.
    """
    data = input_str.split(";")
    town = data[0].strip()
    district_name = data[1].strip()
    district_abb = data[2].strip()
    term = data[3].strip()
    return {
        "town": town,
        "district_name": district_name,
        "district_abb": district_abb,
        "term": term,
    }


def get_stream_data(data_to_write):
    if isinstance(data_to_write, str):
        for word in data_to_write:
            yield word
            time.sleep(0.001)
    else:
        yield data_to_write


def st_stream_write(data_to_write):
    st.write_stream(get_stream_data(data_to_write))


def main():
    with st.sidebar:
        st.title("Pipeline control")
        search_method = st.selectbox(
            "Select the search method to use",
            [
                SearchMethod.EXPERIMENT_3,
                SearchMethod.NONE,
                ExtractionMethod.STUFF,
                ExtractionMethod.MAP,
                ExtractionMethod.TOURNAMENT_REDUCE,
                ExtractionMethod.MULTIPLE_CHOICE,
                ExtractionMethod.REDUCE_AND_CONFIRM,
                SearchMethod.BASELINE,
                SearchMethod.EXPERIMENT_1,
                SearchMethod.EXPERIMENT_2,
                SearchMethod.ELASTICSEARCH,
                SearchMethod.ES_FUZZY,
            ],
        )
        k = st.number_input("ElasticSearch K", 10)
        tournament_k = st.number_input("Tournament K", 10)
        extraction_method = st.selectbox(
            "Select the extraction method to use",
            [
                ExtractionMethod.REDUCE_AND_CONFIRM,
                ExtractionMethod.TOURNAMENT_REDUCE,
                ExtractionMethod.NONE,
                ExtractionMethod.STUFF,
                ExtractionMethod.MAP,
                ExtractionMethod.MULTIPLE_CHOICE,
            ],
        )
        model_name = st.text_input("Model name", "gpt-4-1106-preview")

    st.title("Zoning Document Search")
    st.write("This page allows you to search for terms in zoning documents.")
    st.write("Example input: `andover;Andover Lake;AL;min lot size`")
    chat_input = st.chat_input("Chat with the bot")
    # submit = st.button("Submit")
    if chat_input:
        parsed_input = parse_input(chat_input)

        town = parsed_input["town"]
        district_name = parsed_input["district_name"]
        district_abb = parsed_input["district_abb"]
        term = parsed_input["term"]
        district = District(full_name=district_name, short_name=district_abb)
        st_stream_write(
            f"Searching in town: `{town}`, district: `{district_name}` with abbreviation: `{district_abb}` for term: `{term}`.")
        st_stream_write(f"Searching for term: `{term}`")
        st_stream_write(f"Searching using method: `{search_method}`")
        st_stream_write(f"Returning top `{k}` results")
        st_stream_write(f"---")
        st_stream_write(f"Start 1st stage search (Elasticsearch)...")
        try:
            pages = zoning.term_extraction.search.search_for_term(town, district, term, search_method, k)
            expanded_pages = flatten(page_coverage(pages))
        except Exception as e:
            st_stream_write(f"Elasticsearch search failed. Error: \n```\n{e}\n```")
            return
        st_stream_write(f"1st stage search done. Number of results: `{len(pages)}`")
        for page in pages:
            st_stream_write(
                f"Page: `{page.page_number}`, Score: `{page.score}`, Text: \n```\n{page.text[:100]}...\n```")

        st_stream_write(f"Expanded pages: `{expanded_pages}`")
        st_stream_write(f"---")

        st_stream_write(f"Start 2nd stage search...")
        asyncio.run(consume_async_gen(
            pages=pages,
            term=term,
            town=town,
            district=district,
            method=extraction_method,
            model_name=model_name,  # getting better results with this
            tournament_k=tournament_k,
        ))  # For Python 3.7+

        st_stream_write(f"2nd stage search done.")
        st_stream_write(f"Search completed.")
        st_stream_write(f"---")


if __name__ == '__main__':
    main()
