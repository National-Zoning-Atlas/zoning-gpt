# -*- coding: utf-8 -*-
"""
@Time : 4/22/2024 9:48 PM
@Auth : Wang Yuyang
@File : extract.py
@IDE  : PyCharm
"""
import asyncio
import json
from zoning.main.base.ZoningModule import ZoningModule
from zoning.term_extraction.extract import extract_answer
from zoning.term_extraction.types import District
import jsonpickle


class ZoningExtractModule(ZoningModule):
    def __init__(self):
        self.term = None
        self.pages_result = None
        self.town = None
        self.district_name = None
        self.district_abb = None
        self.search_method = None
        self.elasticsearch_k = None

        super().__init__()
        self.extraction_method = self.args.extraction_method
        self.model_name = self.args.model_name
        self.tournament_k = self.args.tournament_k

    def config_args(self, parser):
        parser.add_argument('--extraction_method', type=str, help='Select the extraction method to use',
                            default='answer_confirm')
        parser.add_argument('--model_name', type=str, help='Enter the model name to use', default='gpt-4-1106-preview')
        parser.add_argument('--tournament_k', type=int, help='Enter the number of results to return', default=10)

    def parse_stdin(self, input_data):
        data = json.loads(input_data)
        self.pages_result = jsonpickle.decode(json.dumps(data['results']))
        self.term = data['metadata']['term']
        self.town = data['metadata']['town']
        self.district_name = data['metadata']['district_name']
        self.district_abb = data['metadata']['district_abb']
        self.search_method = data['metadata']['search_method']
        self.elasticsearch_k = data['metadata']['k']

    async def _async_helper(self):
        outputs = []
        async_gen = extract_answer(
            pages=self.pages_result,
            term=self.term,
            town=self.town,
            district=District(full_name=self.district_name, short_name=self.district_abb),
            method=self.extraction_method,
            model_name=self.model_name,
            tournament_k=self.tournament_k,
        )

        async for output in async_gen:
            outputs.append(output)

        return outputs

    def extract(self):
        loop = asyncio.get_event_loop()
        outputs = loop.run_until_complete(self._async_helper())
        return outputs

    def main(self):
        return self.extract()

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


if __name__ == '__main__':
    module = ZoningExtractModule()
    module.output(module.main())
