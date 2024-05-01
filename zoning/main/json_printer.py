# -*- coding: utf-8 -*-
"""
@Time : 4/22/2024 10:35 PM
@Auth : Wang Yuyang
@File : json_printer.py
@IDE  : PyCharm
"""
import json

import rich

from zoning.main.base.ZoningModule import ZoningModule


class ZoningJSONPrinterModule(ZoningModule):
    def __init__(self):
        self.data = None

        super().__init__()

    def parse_stdin(self, input_data):
        data = json.loads(input_data)
        self.data = data

    def main(self):
        return self.data

    def output(self, results):
        rich.print_json(data=self.data)


if __name__ == '__main__':
    module = ZoningJSONPrinterModule()
    module.output(module.main())
