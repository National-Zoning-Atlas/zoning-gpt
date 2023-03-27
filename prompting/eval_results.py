import json
import re
import math
from os.path import dirname, realpath, join
import pandas as pd
from datasets import load_dataset
from fuzzywuzzy import fuzz

eval_towns = ["vernon", "milford", "westbrook", "franklin", "westport", "madison", "bethany", "avon", "union", "marlborough"]

def get_results(file):
  with open(file, encoding="utf-8") as f:
      json_lines = (json.loads(l) for l in f.readlines())
      return list(json_lines)

def clean_string_units(x):
  res = []
  if any(substring in str(x) for substring in ['sq ft', 'square feet', 'sq. ft.', 'sq. ft', 'sqft', 'ft2', 'ft^2']):
    res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
    res = [float(a.replace(',', '')) for a in res]

  if any(substring in str(x) for substring in ['acres', 'acre', 'acreage']):
    res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
    res = [float(a.replace(',', '')) * 43650 for a in res]
  return res

def parse_results(results):
  res = []
  for x in results:
    res_entry = {}
    if x['Town'] in eval_towns:
      res_entry['town'] = x['Town']
      if not x['Districts']:
          res_entry['district_full'] = ""
          res_entry['district_abb'] = ""
      else:
          res_entry['district_full'] = x['Districts'][0]['Name']['T'].strip()
          res_entry['district_abb'] = x['Districts'][0]['Name']['Z'].strip()

          res_entry['min_lot_size'] = x['Districts'][0]['Sizes']['min lot size'][0] 
          if res_entry['min_lot_size'] != 'n/a' and res_entry['min_lot_size'] != 'N/A':
              res_entry['min_lot_size'] = res_entry['min_lot_size'].split(':')[1].strip()
          if 'Reason' in res_entry['min_lot_size']:
              res_entry['min_lot_size'] =  res_entry['min_lot_size'].split('\n*')[0]  
          res_entry['min_lot_size_page'] = x['Districts'][0]['Sizes']['min lot size'][1]

          res_entry['min_unit_size'] = x['Districts'][0]['Sizes']['min unit size'][0]
          if res_entry['min_unit_size'] != 'n/a' and res_entry['min_unit_size'] != "N/A":
              res_entry['min_unit_size'] = res_entry['min_unit_size'].split(':')[1].strip()
          if 'Reason' in res_entry['min_unit_size']:
              res_entry['min_unit_size'] =  res_entry['min_unit_size'].split('\n*')[0]     
          res_entry['min_unit_size_page'] = x['Districts'][0]['Sizes']['min unit size'][1]
      res.append(res_entry)
  df = pd.DataFrame.from_dict(res)
  df['min_unit_size'] = df['min_unit_size'].apply(lambda x: clean_string_units(x))
  df['min_lot_size'] = df['min_lot_size'].apply(lambda x: clean_string_units(x))
  return df

def score_page(x_pred, x_true):
  if x_true == '-' and math.isnan(x_pred) == False:
    return 0
  elif x_true == '-' and math.isnan(x_pred):
    return 1
  else:
    return x_pred == float(x_true)

def score(x_pred, x_true):
  if x_true == "-" and len(x_pred) == 0:
    return 1
  elif x_true == "_" and len(x_pred) != 0:
    return 0

  if x_true.isnumeric(): #will only work for 1 value...
    if len(x_pred) == 1 and ", " not in x_true:
      return 1 if float(x_pred[0]) == float(x_true) else 0
  if set(x_pred) == set(x_true):
    return 1
  else:
    total = len(x_true)
    intm_score = 0
    for i in range(total):
      if x_true[i] in x_pred:
        intm_score += 1
    return intm_score/total

def eval_answers(df_gt, df_results):
  df_comb = df_results.merge(df_gt, left_on='town', right_on='Town', suffixes=(None, '_gt'))
  
  df_comb['min_lot_size_page_score'] = df_comb.apply(lambda x: score_page(x.min_lot_size_page, x.min_lot_size_page_gt), axis=1)
  df_comb['min_unit_size_page_score'] = df_comb.apply(lambda x: score_page(x.min_unit_size_page, x.min_unit_size_page_gt), axis=1)
  
  df_comb['min_lot_size_score'] = df_comb.apply(lambda x: score(x.min_lot_size, x.min_lot_size_gt), axis=1)
  df_comb['min_unit_size_score'] = df_comb.apply(lambda x: score(x.min_unit_size, x.min_unit_size_gt), axis=1)
  
  min_lot_size_page_score = df_comb['min_lot_size_page_score'].sum()
  min_unit_size_page_score = df_comb['min_unit_size_page_score'].sum()
  min_lot_size_score = df_comb['min_lot_size_score'].sum()
  min_unit_size_score = df_comb['min_unit_size_score'].sum()
  print(df_comb[['min_lot_size', 'min_lot_size_gt', 'min_unit_size', 'min_unit_size_gt', 'min_lot_size_score', 'min_unit_size_score']])

  return min_lot_size_page_score, min_unit_size_page_score, min_lot_size_score, min_unit_size_score


# load ground truth dataset - answers
df_gt = pd.read_csv("ground_truth.csv")
#df_gt['min_lot_size_gt'] = df_gt['min_lot_size_gt'].apply(lambda x: [float(y) for y in x.split(',')])
#df_gt['min_unit_size_gt'] = df_gt['min_unit_size_gt'].apply(lambda x: [float(y) for y in x.split(',')])


if __name__ == "__main__":
  results_file = "sizes_test.jsonl"
  results = get_results(results_file)
  df_results = parse_results(results)
  #print(df_results)
  #score_districts = eval_districts(df_final, df_dataset_gt)
  #print(score_districts)
  score_answers = eval_answers(df_gt, df_results)
  print("score", score_answers)

