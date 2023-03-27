import csv
import json
import re
from os.path import dirname, realpath, join
import pandas as pd
from datasets import load_dataset
from fuzzywuzzy import fuzz

eval_towns = ["vernon", "milford", "westbrook", "franklin", "westport", "madison", "bethany", "avon", "union", "marlborough"]

# load ground truth districts
def load_gt(file="ground_truth.csv", headers=["Town", "District Abbrev.", "District"]):
    districts_gt = pd.read_csv(file)[headers]
    districts_gt_dict = []
    for index, row in districts_gt.iterrows():
        dist = {"Town": row["Town"], "Districts": [{"T": row["District"], "Z": row["District Abbrev."]}]}
        districts_gt_dict.append(dist)
    return districts_gt_dict

# load pred districts
def load_preds(file="districts_test.jsonl"):
  with open(file, encoding="utf-8") as f:
      json_lines = (json.loads(l) for l in f.readlines())
      return list(json_lines)

def remove_words(pred):
    #TODO remove words like District and Zone
    pred_edited = pred
    words_to_remove = ['District', 'district', 'Zone', 'zone']
    num_dist = len(pred_edited['Districts'])
    for i in range(num_dist):
        district =  pred_edited["Districts"][i]
        for word in words_to_remove:
            if word in district["T"]:
                district_fn_edited = district["T"].replace(word, "").strip()
                pred_edited["Districts"][i]["T"] = district_fn_edited
            if word in district["Z"]:
                district_abb_edited = district["Z"].replace(word, "").strip()
                pred_edited["Districts"][i]["Z"] = district_abb_edited
    return pred_edited
    

def find_similar_match(gt, pred):
    match = None
    max_similarity_fn = 0
    max_similarity_abb = 0

    for gt_district in gt['Districts']:
        for pred_district in pred['Districts']:
            #print(pred_district)
            similarity_fn = fuzz.token_sort_ratio(gt_district['T'], pred_district['T'])
            similarity_abb = fuzz.token_sort_ratio(gt_district['Z'], pred_district['Z'])
            if similarity_fn >= max_similarity_fn and similarity_abb >= max_similarity_abb:
                max_similarity_fn = similarity_fn
                max_similarity_abb = similarity_abb
                match = pred_district
    return match

def score_match(gt, match):
    threshold = 80
    gt_fn = gt['Districts'][0]['T']
    gt_abb = gt['Districts'][0]['Z']
    if (gt_abb in gt_fn) and (match['Z'] not in match['T']):
            gt_fn_edited = gt_fn.replace(gt_abb, "")
            similarity_fn = fuzz.token_set_ratio(gt_fn_edited, match['T'])
            similarity_abb = fuzz.token_set_ratio(gt_abb, match['Z'])
    else:
        similarity_fn = fuzz.token_set_ratio(gt_fn, match['T'])
        similarity_abb = fuzz.token_set_ratio(gt_abb, match['Z'])
    
    final_score = 0.5 * (similarity_fn > threshold) + 0.5 * (similarity_abb > threshold)
    print("gt: ", gt, " match: ", match, "score: ", final_score)
    return final_score

if __name__ == "__main__":
    
    with open("districts_matched.jsonl", "a") as out:
        #TODO: overwrite the file everytime or just append to it? for now - append
        gt = load_gt(file="ground_truth.csv")
        pred_all = load_preds(file="districts_test.jsonl")
        score_total = 0
        for x in gt:
            town = x["Town"]
            for y in pred_all:
                if y["Town"] == town:
                    score = 0
                    print(town)
                    if not y["Districts"]:
                        score = 0
                        match = None
                    else:
                        pred_edited = remove_words(y)
                        match = find_similar_match(x, pred_edited)
                        score = score_match(x, match)
                    score_total += score
                    match_val = [] if not match else [match]
                    print(json.dumps({"Town": town, "Districts": match_val}), file=out)
        print("final score = ", score_total)
