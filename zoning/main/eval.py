# -*- coding: utf-8 -*-
"""
@Time : 4/30/2024 4:05 PM
@Auth : Wang Yuyang
@File : eval.py
@IDE  : PyCharm
"""
import json

import jsonpickle
import polars as pl


from zoning.data_processing.eval import eval_result, DATA_ROOT
from zoning.main.base.ZoningModule import ZoningModule


class ZoningEvalModule(ZoningModule):
    def __init__(self):
        self.term = None
        self.pages_result = None
        self.town = None
        self.district_name = None
        self.district_abb = None
        self.search_method = None
        self.elasticsearch_k = None
        self.extraction_method = None
        self.model_name = None
        self.tournament_k = None
        super().__init__()

    def config_args(self, parser):
        pass

    def parse_stdin(self, input_data):
        data = json.loads(input_data)
        self.pages_result = jsonpickle.decode(json.dumps(data['results']))
        self.term = data['metadata']['term']
        self.town = data['metadata']['town']
        self.district_name = data['metadata']['district_name']
        self.district_abb = data['metadata']['district_abb']
        self.search_method = data['metadata']['search_method']
        self.elasticsearch_k = data['metadata']['k']
        self.extraction_method = data['metadata']['extraction_method']
        self.model_name = data['metadata']['model_name']
        self.tournament_k = data['metadata']['tournament_k']

    def eval(self):
        # def eval_result(extraction_method, gt, results, term):
        gt = pl.read_csv(
            DATA_ROOT / "ground_truth.csv",
            dtypes={
                **{f"{tc}_gt": pl.Utf8 for tc in [self.term]},
                **{f"{tc}_page_gt": pl.Utf8 for tc in [self.term]},
            },
            # n_rows=num_eval_rows,
        )

        outputs = eval_result(
            extraction_method=self.extraction_method,
            gt=self.pages_result,
            results=self.pages_result,
            term=self.term
        )
        return outputs

    def main(self):
        return self.eval()

    def output(self, results):
        print(self.DATA_START_FLAG)

        data = {
            'metadata': {
                'term': self.term,
                'town': self.town,
                'district_name': self.district_name,
                'district_abb': self.district_abb,
                'search_method': self.search_method,
                'k': self.elasticsearch_k,
                'extraction_method': self.extraction_method,
                'model_name': self.model_name,
                'tournament_k': self.tournament_k,
            },
            'results': json.loads(jsonpickle.encode(results))
        }
        print(json.dumps(data, indent=4))
