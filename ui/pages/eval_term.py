# -*- coding: utf-8 -*-
"""
@Time : 2/12/2024 1:13 AM
@Auth : Wang Yuyang
@File : eval_term.py
@IDE  : PyCharm
"""
import asyncio
import streamlit as st
from zoning.data_processing.eval import main as eval_main  # Adjusted import based on your async function


# Define a function to run asyncio event loop
def run_asyncio_coroutine(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result


def main():
    st.title("Zoning Document Search Evaluation")
    st.write("This page allows you to evaluate the search for a term in zoning documents.")

    # Inputs
    term = st.text_input("Enter the term to evaluate", "min_lot_size")
    search_method = st.selectbox("Select the search method to use", ["elasticsearch"])
    extraction_method = st.selectbox("Select the extraction method to use", ["tournament_reduce"])
    num_eval_rows = st.number_input("Enter the number of rows to evaluate", value=30)
    k = st.number_input("Enter the number of results to return", value=10)

    # Evaluate button
    evaluate_button = st.button("Evaluate")

    # Async evaluation
    if evaluate_button:
        async_result = run_asyncio_coroutine(
            eval_main(search_method, extraction_method, [term], k, num_eval_rows=num_eval_rows)
        )
        # Store the result in session state to be accessible after rerun
        st.session_state['async_result'] = async_result

    # Display results (this part runs on rerun after the session state is updated)
    if 'async_result' in st.session_state and st.session_state['async_result']:
        metrics, new_results_df = st.session_state['async_result']
        st.write(metrics)
        st.write(new_results_df)


if __name__ == '__main__':
    main()
