# -*- coding: utf-8 -*-
"""
@Time : 4/17/2024 3:14 PM
@Auth : Wang Yuyang
@File : search.py
@IDE  : PyCharm
"""
import datetime
import json
import warnings
import zoning
from zoning.main.base.ZoningModule import ZoningModule
from zoning.term_extraction.types import District
import jsonpickle

warnings.filterwarnings("ignore")


class ZoningSearchModule(ZoningModule):
    def __init__(self):
        super().__init__()
        self.term = self.args.term
        self.town = self.args.town
        self.district_name = self.args.district_name
        self.district_abb = self.args.district_abb
        self.district = District(full_name=self.district_name, short_name=self.district_abb)
        self.search_method = self.args.search_method
        self.k = self.args.k

    def config_args(self, parser):
        parser.add_argument('--term', type=str, help='Enter the term to search for', default='min lot size')
        parser.add_argument('--town', type=str, help='Enter the town to search in', default='andover')
        parser.add_argument('--district_name', type=str, help='Enter the district to search in', default='Andover Lake')
        parser.add_argument('--district_abb', type=str, help='Enter the district abbreviation', default='AL')
        parser.add_argument('--search_method', type=str, help='Select the search method to use',
                            default='elasticsearch')
        parser.add_argument('--k', type=int, help='Enter the number of results to return', default=10)

    def get_stdin(self):
        return None

    def main(self):
        results = zoning.term_extraction.search.search_for_term(self.town, self.district, self.term, self.search_method,
                                                                self.k)
        return results

    def output(self, results):
        print(self.DATA_START_FLAG)
        data = {
            'metadata': {
                'term': self.term,
                'town': self.town,
                'district_name': self.district_name,
                'district_abb': self.district_abb,
                'search_method': self.search_method,
                'k': self.k
            },
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': json.loads(jsonpickle.encode(results))
        }
        print(json.dumps(data, indent=4))


if __name__ == '__main__':
    module = ZoningSearchModule()
    module.output(module.main())
