# -*- coding: utf-8 -*-
"""
@Time : 2/11/2024 11:26 PM
@Auth : Wang Yuyang
@File : main.py
@IDE  : PyCharm
"""
import streamlit as st
from urllib.request import urlopen

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout='wide',
)

# st.sidebar.success("Select a action above.")
st.page_link("pages/chat.py", label="Start chat")

markdown_url = "https://raw.githubusercontent.com/National-Zoning-Atlas/zoning-gpt/master/README.md"
response = urlopen(markdown_url)
data = response.read()
text = data.decode('utf-8')
st.markdown(text)




