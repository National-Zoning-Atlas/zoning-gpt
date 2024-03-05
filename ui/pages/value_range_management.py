# -*- coding: utf-8 -*-
"""
@Time : 2/25/2024 5:35 PM
@Auth : Wang Yuyang
@File : value_range_management.py
@IDE  : PyCharm
"""
import streamlit as st
import json
from pathlib import Path

from zoning import get_project_root


# Load JSON data from file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Save JSON data to file
def save_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Path to the JSON file
file_path = get_project_root().joinpath("zoning/term_extraction/value_ranges.json")

# Load the data
data = load_data(file_path)

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
st.title("value_ranges.json File Editor")

for key in data.keys():
    st.subheader(f"Key: {key}")
    values = data[key]

    # Display existing values
    for i, value in enumerate(values):
        edited_value = st.text_input(f"Value {i + 1}", value, key=f"{key}_{i}")
        if edited_value != value:
            data[key][i] = edited_value

    # Add new value to the key
    new_value = st.text_input(f"Add new value to {key}", "")
    if st.button(f"Add to {key}", key=f"add_{key}"):
        if new_value:
            try:
                # Convert to appropriate type (int or str)
                new_value_converted = int(new_value) if new_value.isdigit() else new_value
                data[key].append(new_value_converted)
                save_data(file_path, data)
                st.success(f"Value added to {key}.")
            except ValueError as e:
                st.error("Error adding value. Please check the input.")

# Button to save changes
if st.button("Save Changes"):
    save_data(file_path, data)
    st.success("Changes saved.")

