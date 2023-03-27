import json
import re
from os.path import dirname, realpath, join
import pandas as pd
from datasets import load_dataset
from fuzzywuzzy import fuzz

def get_results(file="sizes.jsonl", res_index=10):
  with open(file, encoding="utf-8") as f:
      json_lines = (json.loads(l) for l in f.readlines())
      return list(json_lines)[:res_index] if res_index != -1 else list(json_lines)

def parse_results(results):
  res = []
  for x in results:
    res_entry = {}
    res_entry['town'] = x['Town']
    if not x['Districts']:
        res_entry['district_full'] = ""
        res_entry['district_abb'] = ""
    else:
        res_entry['district_full'] = x['Districts'][0]['Name']['T'].strip()
        rmv_words = ['District', 'district', 'Zone', 'zone']
        for w in rmv_words:
          if w in res_entry['district_full']:
            res_entry['district_full'] = res_entry['district_full'].replace(w, '').strip()
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
        res_entry['min unit size page'] = x['Districts'][0]['Sizes']['min unit size'][1]
    res.append(res_entry)
  return pd.DataFrame.from_dict(res)

def clean_string_units(x):
  res = []
  if any(substring in str(x) for substring in ['sq ft', 'square feet', 'sq. ft.', 'sq. ft', 'sqft', 'ft2', 'ft^2']):
    res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
    res = [float(a.replace(',', '')) for a in res]

  if any(substring in str(x) for substring in ['acres', 'acre', 'acreage']):
    res =  re.findall(r'(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?', x)
    res = [float(a.replace(',', '')) * 43650 for a in res]
  return res[0] if len(res) == 1 else res

def clean_units(df):
  df['min_unit_size'] = df['min_unit_size'].apply(lambda x: clean_string_units(x))
  df['min_lot_size'] = df['min_lot_size'].apply(lambda x: clean_string_units(x))
  return df

def check_district(town, dist, df_dataset_gt):
  districts_all = df_dataset_gt[df_dataset_gt['Jurisdiction'] == town]['Full District Name'].values.tolist()
  print(town, districts_all)
  match_ratio = [fuzz.ratio(dist.lower().strip(), y.lower().strip()) for y in districts_all]
  if max(match_ratio) >= 90:
    return 1
  else:
    return 0

def eval_districts(df, df_dataset_gt):
  # checking for full district name only for now
  #TODO: extend to abbreviated district as well

  df['district score'] = df.apply(lambda x: check_district(x.town, x.district_full, df_dataset_gt), axis=1)
  print(df)
  return df['district score'].sum()

def eval_answers(df, df_gt):
  #TODO: write function
  return

# load ground truth dataset - districts
dataset_gt = load_dataset("xyzNLP/nza-ct-zoning-atlas-metadata")
dataset_gt.set_format("pandas")
df_dataset_gt = dataset_gt['train'][:]
# load ground truth dataset - answers
df_gt = pd.read_csv("ground_truth.csv")

if __name__ == "__main__":
  results = get_results("sizes.jsonl", res_index=10)
  df_parsed = parse_results(results)
  df_final = clean_units(df_parsed)
  score_districts = eval_districts(df_final, df_dataset_gt)
  print(score_districts)

