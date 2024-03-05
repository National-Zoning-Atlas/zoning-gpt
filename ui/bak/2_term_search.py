# -*- coding: utf-8 -*-
"""
@Time : 2/12/2024 12:58 AM
@Auth : Wang Yuyang
@File : 2_term_search.py
@IDE  : PyCharm
"""
import collections
import time
import numpy as np
import pandas as pd
import streamlit as st

from elasticsearch import Elasticsearch

import zoning
from zoning.term_extraction.types import District


# pdm run python -m zoning.data_processing.eval --num-eval-rows 30 --terms min_lot_size --search-method elasticsearch --extraction-method tournament_reduce --k 10


def main():
    st.title("Zoning Document Search")
    st.write("This page allows you to search for terms in zoning documents.")

    # Get the term to search for
    term = st.text_input("Enter the term to search for", "min lot size")

    # Get the town to search in
    town = st.text_input("Enter the town to search in", "andover")

    # Get the district to search in
    district_name = st.text_input("Enter the district to search in", "Andover Lake")
    district_abb = st.text_input("Enter the district abbreviation", "AL")

    district = District(full_name=district_name, short_name=district_abb)

    # Get the search method to use
    search_method = st.selectbox(
        "Select the search method to use",
        [
            "elasticsearch",
        ],
    )

    # Get the number of results to return
    k = st.number_input("Enter the number of results to return", 10)

    # Search for the term
    if st.button("Search"):
        st.write("Searching for term...")
        results = zoning.term_extraction.search.search_for_term(town, district, term, search_method, k)
        st.write(results)




if __name__ == '__main__':
    main()
