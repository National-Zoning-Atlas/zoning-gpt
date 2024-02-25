# -*- coding: utf-8 -*-
"""
@Time : 2/25/2024 4:13 PM
@Auth : Wang Yuyang
@File : thesaurus_management.py
@IDE  : PyCharm
"""
import streamlit as st
import json

from zoning import get_project_root

import streamlit as st
import json
from pathlib import Path


# Load JSON data from file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Save JSON data to file
def save_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Path to the JSON file
file_path = get_project_root().joinpath("zoning/term_extraction/thesaurus.json")


# Load the data
data = load_data(file_path)


# Function to display and edit a single key
def edit_key_values(key, values):
    with st.expander(f"Edit '{key}':", expanded=True):
        # Edit existing values
        for i, value in enumerate(values):
            edited_value = st.text_input(f"{key} Value {i + 1}", value, key=f"{key}_{i}")
            if edited_value != value:
                data[key][i] = edited_value

        # Add new value to the key
        new_value = st.text_input(f"Add new value to '{key}'", "", key=f"add_{key}")
        if st.button(f"Add to '{key}'", key=f"addbtn_{key}"):
            if new_value:
                data[key].append(new_value)
                st.success(f"Value '{new_value}' added to '{key}'.")


# Sidebar for adding a new key
with st.sidebar:
    st.header("Add New Key")
    new_key = st.text_input("Key Name")
    if st.button("Add Key"):
        if new_key and new_key not in data:
            data[new_key] = []
            save_data(file_path, data)
            st.success(f"Key '{new_key}' added.")

# Main page for editing keys and values
st.title("thesaurus.json Editor")

for key, values in data.items():
    edit_key_values(key, values)

# Button to save changes
if st.button("Save Changes"):
    save_data(file_path, data)
    st.success("Changes saved. Refresh the page to see updated values.")
