import json
import re
import pandas as pd
from fuzzywuzzy import fuzz
from zoning.utils import get_project_root

eval_towns = ["bristol", "chester", "southington", "hebron", "newington", "south-windsor", "warren", "morris", "east-haddam", "ellington"]

# load ground truth districts
def load_gt(file="ground_truth.csv", headers=["town", "district_abb", "district"]):
    districts_gt = pd.read_csv(file)[headers]
    districts_gt_dict = []
    for index, row in districts_gt.iterrows():
        dist = {"Town": row["town"], "Districts": [{"T": row["district"], "Z": row["district_abb"]}]}
        districts_gt_dict.append(dist)
    return districts_gt_dict

# load pred districts
def load_preds(file="districts_test.jsonl"):
  with open(file, encoding="utf-8") as f:
      json_lines = (json.loads(l) for l in f.readlines())
      return list(json_lines)

def index_or_minus_one(lst_gt, lst_res):
    top_k = []
    for x in lst_gt:
        try:
            top_k.append(lst_res.index(x))
        except ValueError:
            top_k.append(-1)
    return top_k

def check_page(lst_gt, lst_res):
    num = len(lst_gt)
    inc = 1/num
    score = 0
    for x in lst_gt:
        if x in lst_res:
            score += inc
    return score

def score_page(gt_df, pred):
    comb = gt_df[["town", "district_page"]]
    comb["district_page_int"] = comb["district_page"].apply(lambda x: [int(s) for s in re.findall(r'\b\d+\b', x)])
    # print(comb)
    comb["pages_result"] = comb.apply(lambda x: [0,0,0,0], axis=1)
    comb["pages_result"] = comb["pages_result"].astype(object)
    for x in pred:
        town = x["Town"]
        pages = x["Pages"]
        #comb = comb.set_index("town")
        comb.loc[comb['town'] == town, "pages_result"] = comb.loc[comb['town'] == town, "pages_result"].apply(lambda x: pages)
    comb["page_score"] = comb.apply(lambda x: check_page(x.district_page_int, x.pages_result) , axis=1)
    comb["top-k"] = comb.apply(lambda x: index_or_minus_one(x.district_page_int, x.pages_result), axis=1)
    # print(comb)
    comb.to_csv("res_csv/pages_score_baseline.csv")
    return comb["page_score"].sum()

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
    threshold = 70
    gt_fn = gt['Districts'][0]['T']
    gt_abb = gt['Districts'][0]['Z']
    if (gt_abb in gt_fn) and (match['Z'] not in match['T']):
            gt_fn_edited = gt_fn.replace(gt_abb, "")
            similarity_fn = fuzz.token_set_ratio(gt_fn_edited, match['T'])
            similarity_abb = fuzz.token_set_ratio(gt_abb, match['Z'].replace(".", ""))
    else:
        similarity_fn = fuzz.token_set_ratio(gt_fn, match['T'])
        similarity_abb = fuzz.token_set_ratio(gt_abb, match['Z'].replace(".", ""))
    
    final_score = 0.5 * (similarity_fn > threshold) + 0.5 * (similarity_abb > threshold)
    print("gt: ", gt, " match: ", match, "score: ", final_score)
    return final_score

if __name__ == "__main__":
    read_file = str(get_project_root() / "data" / "results" / "districts_test.jsonl")
    write_file = str(get_project_root() / "data" / "results" / "districts_matched_2.jsonl")
    df_results = pd.DataFrame(columns=["town", "district"])
    with open(write_file, "a") as out:
        #TODO: overwrite the file everytime or just append to it? for now - append
        gt = load_gt(file="ground_truth.csv")
        gt_df = pd.read_csv("ground_truth.csv")
        pred_all = load_preds(file=read_file)
        page_score = score_page(gt_df, pred_all)
        # print(page_score)
        score_total = 0
        for x in gt:
            town = x["Town"]
            for y in pred_all:
                if y["Town"] == town:
                    score = 0
                    if not y["Districts"]:
                        score = 0
                        match = None
                    else:
                        pred_edited = remove_words(y)
                        match = find_similar_match(x, pred_edited)
                        score = score_match(x, match)
                    score_total += score
                    if not match:
                        match_val = []
                        temp_df = pd.DataFrame({"town": town, "district_abb": "", "district": "", "score": score}, index=[0]) 
                    else:
                        match_val = [match]
                        temp_df = pd.DataFrame({"town": town, "district_abb": match["Z"], "district": match["T"], "score": score}, index=[0])
                    print(json.dumps({"Town": town, "Districts": match_val}), file=out)
                    #temp_df = pd.DataFrame({"town": town, "district_abb": match["Z"], "district": match["T"], "score": score})
                    df_results = pd.concat([df_results, temp_df], ignore_index=True)
        print("page score = ", page_score, " districts score = ", score_total)
    df_results.to_csv("res_csv/districts_baseline.csv")