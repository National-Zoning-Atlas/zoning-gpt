# -*- coding: utf-8 -*-
"""
@Time : 2/11/2024 11:44 PM
@Auth : Wang Yuyang
@File : 1_loading_elasticsearch.py.py
@IDE  : PyCharm
"""
import time
import numpy as np
import pandas as pd
import streamlit as st

from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")  # default client


def get_cluster_health():
    """Fetches the cluster health."""
    return es.cluster.health()


def get_indices():
    """Fetches a list of indices in the cluster."""
    return es.cat.indices(format='json')


def display_cluster_info():
    """Displays Elasticsearch cluster information."""
    st.header("Elasticsearch Cluster Status")

    # Display Cluster Health
    health = get_cluster_health()
    st.subheader("Cluster Health")
    st.json(health)

    # Display Indices Information
    indices = get_indices()
    st.subheader("Indices")
    if indices:
        df_indices = pd.DataFrame(indices)
        st.dataframe(df_indices)
    else:
        st.write("No indices found.")

def run_query():
    """Runs a simple query against the Elasticsearch cluster."""
    st.header("Elasticsearch Actions")
    st.subheader("Load Data")
    existing_indices = get_indices()
    existing_indices = [index['index'] for index in existing_indices]
    st.write("Existing indices:", existing_indices)

    if st.button("Load Data"):
        st.write("Loading data into Elasticsearch...")
        # Load data into Elasticsearch
        zoning.data_processing.index_towns.main(st)
        st.write("Data loaded.")


    st.subheader("Run a Query")
    town_query = st.text_input("Enter a town name", "andover")
    text_query = st.text_input("Enter a text query", "zoning")
    if st.button("Run Query"):
        res = es.search(index=town_query, body={"query": {"match": {"Text": text_query}}})
        st.subheader("Query Results")
        st.json(res, expanded=False)
        dataFrame = pd.DataFrame(res['hits']['hits'])
        st.write(dataFrame)



def main():
    """Main function to run the Streamlit app."""
    run_query()
    display_cluster_info()


if __name__ == "__main__":
    main()
