# -*- coding: utf-8 -*-
"""
@Time : 4/17/2024 3:14 PM
@Auth : Wang Yuyang
@File : search.py
@IDE  : PyCharm
"""
import argparse


# #
# def main():
#     st.title("Zoning Document Search")
#     st.write("This page allows you to search for terms in zoning documents.")
#
#     # Get the term to search for
#     term = st.text_input("Enter the term to search for", "min lot size")
#
#     # Get the town to search in
#     town = st.text_input("Enter the town to search in", "andover")
#
#     # Get the district to search in
#     district_name = st.text_input("Enter the district to search in", "Andover Lake")
#     district_abb = st.text_input("Enter the district abbreviation", "AL")
#
#     district = District(full_name=district_name, short_name=district_abb)
#
#     # Get the search method to use
#     search_method = st.selectbox(
#         "Select the search method to use",
#         [
#             "elasticsearch",
#         ],
#     )
#
#     # Get the number of results to return
#     k = st.number_input("Enter the number of results to return", 10)
#
#     # Search for the term
#     if st.button("Search"):
#         st.write("Searching for term...")
#         results = zoning.term_extraction.search.search_for_term(town, district, term, search_method, k)
#         st.write(f"Found {len(results)} results.")
#         st.markdown("## Results")
#         st.write(results)
#
#
# if __name__ == '__main__':
#     main()

class ZoningModule:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Zoning Document Search')
        parser.add_argument('--term', type=str, help='Enter the term to search for', default='min lot size')
        parser.add_argument('--town', type=str, help='Enter the town to search in', default='andover')
        parser.add_argument('--district_name', type=str, help='Enter the district to search in', default='Andover Lake')
        parser.add_argument('--district_abb', type=str, help='Enter the district abbreviation', default='AL')
        parser.add_argument('--search_method', type=str, help='Select the search method to use',
                            default='elasticsearch')
        parser.add_argument('--k', type=int, help='Enter the number of results to return', default=10)
        self.args = parser.parse_args()
        print(f"Search term: {self.args.term}\n"
              f"Search town: {self.args.town}\n"
              f"Search district name: {self.args.district_name}\n"
              f"Search district abbreviation: {self.args.district_abb}\n"
              f"Search method: {self.args.search_method}\n"
              f"Number of results: {self.args.k}"
              f"Search for term...")

    def search(self):
        pass


if __name__ == '__main__':
    pass
