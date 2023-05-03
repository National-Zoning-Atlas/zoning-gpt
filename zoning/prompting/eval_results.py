import json
import re
import math
from os.path import dirname, realpath, join
import pandas as pd
from datasets import load_dataset
from fuzzywuzzy import fuzz
from zoning.utils import get_project_root

eval_towns = ["bristol", "southington", "hebron", "newington", "south-windsor", "warren", "morris", "east-haddam", "ellington",  "cheshire"]

def get_results(file):
  with open(file, encoding="utf-8") as f:
      json_lines = (json.loads(l) for l in f.readlines())
      return list(json_lines)

def extract_fraction_decimal(text):
    fraction_pattern = r'\d+\s*\d*\/\d+'
    fractions = re.findall(fraction_pattern, text)
    if fractions:
        fraction = fractions[0]
        if ' ' in fraction:
            whole, fraction = fraction.split()
            numerator, denominator = map(int, fraction.split('/'))
            decimal_value = int(whole) + numerator / denominator
        else:
            numerator, denominator = map(int, fraction.split('/'))
            decimal_value = numerator / denominator
        return decimal_value
    else:
        return None

def clean_string_units(x):
    res = []
    x = x.lower() if isinstance(x, str) else x
    if any(substring in str(x) for substring in ['sq ft', 'square feet', 'sq. ft.', 'sq. ft', 'sqft',
                                               'ft2', 'ft^2', 'sf', 's.f.', 'SF', 'sq.ft.', 'sq-ft', 'sq. Ft.']):
        if '/' in x:
          x_split = x.split('/')
          num = len(x_split) - 1
          res = [extract_fraction_decimal(x) for i in range(num)]
        else:
          res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
          res = [float(a.replace(',', '')) for a in res]
    if any(substring in str(x) for substring in ['acres', 'acre', 'acreage', '-acre']):
        if '/' in x:
          x_split = x.split('/')
          num = len(x_split) - 1
          res = [extract_fraction_decimal(x) * 43560 for i in range(num)]
        else:
          res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
          res = [float(a.replace(',', '')) * 43560 for a in res]
    return res

def clean_gt(x):
  res = []
  res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
  res = [float(a.replace(',', '')) for a in res]
  return res

def parse_results(results):
  res = []
  for x in results:
    res_entry = {}
    if x['Town'] in eval_towns:
      print(x['Town'])
      res_entry['town'] = x['Town']
      if not x['Districts']:
          res_entry['district_full'] = ""
          res_entry['district_abb'] = ""
      else:
          res_entry['district_full'] = x['Districts'][0]['Name']['T'].strip()
          res_entry['district_abb'] = x['Districts'][0]['Name']['Z'].strip()

          res_entry['min_lot_size'] = x['Districts'][0]['Sizes']['min lot size'][0]
          print(res_entry) 
          if res_entry['min_lot_size'] != 'n/a' and res_entry['min_lot_size'] != 'N/A':
            res_entry['min_lot_size'] = res_entry['min_lot_size'].split(':')[1].strip()
          res_entry['min_lot_size_topk_pages'] = x['Districts'][0]['Sizes']['min lot size_pages']
          if 'Reason' in res_entry['min_lot_size']:
            res_entry['min_lot_size'] =  res_entry['min_lot_size'].split('\n*')[0]  
          res_entry['min_lot_size_page'] = x['Districts'][0]['Sizes']['min lot size'][1]

          res_entry['min_unit_size'] = x['Districts'][0]['Sizes']['min unit size'][0]
          if res_entry['min_unit_size'] != 'n/a' and res_entry['min_unit_size'] != "N/A":
            res_entry['min_unit_size'] = res_entry['min_unit_size'].split(':')[1].strip()
          res_entry['min_unit_size_topk_pages'] = x['Districts'][0]['Sizes']['min unit size_pages']
          if 'Reason' in res_entry['min_unit_size']:
            res_entry['min_unit_size'] =  res_entry['min_unit_size'].split('\n*')[0]     
          res_entry['min_unit_size_page'] = x['Districts'][0]['Sizes']['min unit size'][1]
      res.append(res_entry)
  df = pd.DataFrame.from_dict(res)
  df['min_unit_size'] = df['min_unit_size'].apply(lambda x: clean_string_units(x))
  df['min_lot_size'] = df['min_lot_size'].apply(lambda x: clean_string_units(x))
  return df

def score_page(x_pred, x_true):
  if '/' in x_true:
    pages = x_true.split('/')
    pages_lst = [float(a) for a in pages]
    return 1 if x_pred in pages_lst else 0
  if x_true == '-' and x_pred == -1.0:
    return 1
  elif x_true == '-' and x_pred != -1.0:
    return 0
  elif math.isnan(x_pred): # it means district not retrieved 
    return 0
  elif x_true != '-':
    return int(x_pred == float(x_true))

def page_recall(x_true, top_k):
  if '/' in x_true:
    pages = x_true.split('/')
    pages_lst = [float(a) for a in pages]
    return 1 if any(x in pages_lst for x in top_k) else 0
  if isinstance(top_k, list) == False:
    return 0
  elif x_true == '-' and len(top_k) == 0:
    return 1
  elif x_true == '-' and len(top_k) != 0:
    return 0
  elif int(x_true) in top_k:
    return 1
  else:
    return 0

def score(x_pred, x_true):
  if x_true == "-" and len(x_pred) == 0:
    return 1
  elif x_true == "_" and len(x_pred) != 0:
    return 0

  #if x_true.isnumeric(): #will only work for 1 value...
  if len(x_pred) == 1 and len(x_true) == 1:
    return 1 if x_pred[0] == float(x_true[0]) else 0
  # print("set", x_true)
  # x_true_lst = [x for x in x_true]
  if set(x_true) <= set(x_pred):
    return 1
  else:
    total = len(x_true)
    intm_score = 0
    for i in range(total):
      if x_true[i] in x_pred:
        intm_score += 1
    return intm_score/total

def eval_answers(df_gt, df_results):
  df_comb = df_results.merge(df_gt, left_on='town', right_on='town', suffixes=(None, '_gt'))

  df_comb['min_lot_size_gt'] = df_comb['min_lot_size_gt'].apply(lambda x: clean_gt(x))
  df_comb['min_unit_size_gt'] = df_comb['min_unit_size_gt'].apply(lambda x: clean_gt(x))

  print(df_comb[['min_lot_size_page_gt', 'min_unit_size_page_gt', 'min_lot_size_topk_pages', 'min_unit_size_topk_pages']])

  df_comb['min_lot_size_page_score'] = df_comb.apply(lambda x: score_page(x.min_lot_size_page, x.min_lot_size_page_gt), axis=1)
  df_comb['min_unit_size_page_score'] = df_comb.apply(lambda x: score_page(x.min_unit_size_page, x.min_unit_size_page_gt), axis=1)

  df_comb['min_lot_size_page_recall'] = df_comb.apply(lambda x: page_recall(x.min_lot_size_page_gt, x.min_lot_size_topk_pages), axis=1)
  df_comb['min_unit_size_page_recall'] = df_comb.apply(lambda x: page_recall(x.min_unit_size_page_gt, x.min_unit_size_topk_pages), axis=1)
  
  df_comb['min_lot_size_score'] = df_comb.apply(lambda x: score(x.min_lot_size, x.min_lot_size_gt), axis=1)
  df_comb['min_unit_size_score'] = df_comb.apply(lambda x: score(x.min_unit_size, x.min_unit_size_gt), axis=1)
  
  min_lot_size_page_recall = df_comb['min_lot_size_page_recall'].sum()
  min_unit_size_page_recall = df_comb['min_unit_size_page_recall'].sum()
  min_lot_size_page_score = df_comb['min_lot_size_page_score'].sum()
  min_unit_size_page_score = df_comb['min_unit_size_page_score'].sum()
  min_lot_size_score = df_comb['min_lot_size_score'].sum()
  min_unit_size_score = df_comb['min_unit_size_score'].sum()
  print(df_comb[['town','min_lot_size', 'min_lot_size_gt', 'min_lot_size_score']])
  print(df_comb[['town','min_unit_size', 'min_unit_size_gt', 'min_unit_size_score']])
  print(df_comb[['town','min_lot_size_page', 'min_lot_size_page_gt', 'min_lot_size_page_score', 'min_lot_size_topk_pages', 'min_lot_size_page_recall']])
  print(df_comb[['town','min_unit_size_page', 'min_unit_size_page_gt', 'min_unit_size_page_score', 'min_unit_size_topk_pages', 'min_unit_size_page_recall']])
  df_save = df_comb[['min_lot_size_page', 'min_lot_size', 'min_lot_size_topk_pages', 'min_lot_size_score', 'min_lot_size_page_score', 'min_lot_size_page_recall', 
  'min_unit_size_page', 'min_unit_size', 'min_unit_size_topk_pages', 'min_unit_size_score', 'min_unit_size_page_score', 'min_unit_size_page_recall']]
  df_save.to_csv("res_csv/cols_more_pages.csv")
  return {"min_lot_size_page_recall": min_lot_size_page_recall, "min_unit_size_page_recall": min_unit_size_page_recall, 
  "min_lot_size_page_score": min_lot_size_page_score, "min_unit_size_page_score": min_unit_size_page_score, 
  "min_lot_size_score": min_lot_size_score, "min_unit_size_score": min_unit_size_score}


if __name__ == "__main__":
  # load ground truth dataset - answers
  df_gt = pd.read_csv("ground_truth.csv")
  results_file = str(get_project_root() / "data" / "results" / "sizes_test_4.jsonl")
  results = get_results(results_file)
  df_results = parse_results(results)
  score_answers = eval_answers(df_gt, df_results)
  print("score", score_answers)

