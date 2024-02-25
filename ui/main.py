# -*- coding: utf-8 -*-
"""
@Time : 2/11/2024 11:26 PM
@Auth : Wang Yuyang
@File : main.py
@IDE  : PyCharm
"""
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Cornell Tech x National Zoning Atlas: Understanding Zoning Codes with Large Language Models 👋")

st.sidebar.success("Select a action above.")

st.markdown(
    """
This repository is the result of a collaboration between the team led by Sara Bronin at the National Zoning Atlas and a team of researchers under Alexander Rush at Cornell Tech.

The National Zoning Atlas (NZA) is working to depict key aspects of zoning codes in an online, user-friendly map. This currently requires extensive human effort to manually review zoning codes and extract information for each zoning district within each jurisdiction in the country.

The goal of this project is to use Large Language Models (LLMs) in conjunction with other natural language processing (NLP) techniques to automatically extract structured, relevant information from U.S. Zoning Code documents, so as to help the NZA team expand the reach of its atlas.
"""
)

st.page_link("pages/chat.py", label="Start chat")

