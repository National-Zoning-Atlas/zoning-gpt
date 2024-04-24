# -*- coding: utf-8 -*-
"""
@Time : 4/24/2024 12:20 PM
@Auth : Wang Yuyang
@File : table_printer.py
@IDE  : PyCharm
"""
import json

import jsonpickle
import pandas as pd
import rich
from tabulate import tabulate

from zoning.main.base.ZoningModule import ZoningModule


class ZoningTablePrinterModule(ZoningModule):
    def __init__(self):
        self.data = None

        super().__init__()

    def parse_stdin(self, input_data):
        data = json.loads(input_data)
        self.data = data

    def main(self):
        return self.data

    def output(self, results):
        rich.print(f"Metadata:")
        print(tabulate([results['metadata']], headers="keys", tablefmt="grid"))
        results_data = jsonpickle.decode(json.dumps(results['results']))
        results_data = [x.to_dict() for x in results_data]
        for i in range(len(results_data)):
            for key in results_data[i].keys():
                results_data[i][key] = str(results_data[i][key])
                results_data[i][key] = results_data[i][key][:20] + '...'
        rich.print(f"Results:")
        print(tabulate(results_data, headers="keys", tablefmt="grid"))

if __name__ == '__main__':
    module = ZoningTablePrinterModule()
    module.output(module.main())
