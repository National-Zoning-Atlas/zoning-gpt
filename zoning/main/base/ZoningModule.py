# -*- coding: utf-8 -*-
"""
@Time : 4/22/2024 6:59 PM
@Auth : Wang Yuyang
@File : ZoningModule.py
@IDE  : PyCharm
"""
import argparse
import pickle
import base64
import sys


class ZoningModule:
    DATA_START_FLAG = "+" * 10 + "-" * 10 + "+" * 10 + "START" + "+" * 10 + "-" * 10 + "+" * 10

    def __init__(self):
        self.args = None
        self.stdin = ""
        self.parser = argparse.ArgumentParser(description='Zoning Document Search')
        self.config_args(self.parser)
        self.args = self.parser.parse_args()
        self.parse_stdin(self.get_stdin())

    def get_stdin(self):
        print(f"{self.__class__.__name__} waiting for stdin...")
        for line in sys.stdin:
            self.stdin += line
        print(f"{self.__class__.__name__} received stdin with length {len(self.stdin)}")
        # Only return the data after the start flag
        if self.DATA_START_FLAG in self.stdin:
            return self.stdin.split(self.DATA_START_FLAG)[1]
        return self.stdin

    def parse_stdin(self, input_data):
        return input_data

    def config_args(self, parser):
        return parser

    def main(self):
        pass

    def output(self, results):
        pass
