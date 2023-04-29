import yaml

from tqdm.contrib.concurrent import thread_map
import pandas as pd

from ..prompting.eval_results import clean_string_units
from ..prompting.extract import extract_size, ExtractionMethod
from ..utils import get_project_root

DATA_ROOT = get_project_root() / "data"

EVAL_METRICS_PATH = DATA_ROOT / "results" / "eval.yaml"
EVAL_OUTPUT_PATH = DATA_ROOT / "results" / "eval.csv"

def compute_eval_result(town: str, district_name: str, term: str, row):
    try:
        outputs = extract_size(
            town,
            dict(T=district_name, Z=row.district_abb),
            term,
            6,
            method=ExtractionMethod.NONE,
        )
    except Exception as e:
        print(f"Error: {town} {district_name} | {e}")
        return
    gt_page = set(map(int, str(row.min_lot_size_page_gt).split(",")))
    for result in outputs:
        searched_pages = {r.page_number for r in result.search_pages}
        searched_pages_expanded = set(result.search_pages_expanded)

        extracted_pages = (
            set(result.output.pages) if result.output is not None else set()
        )

        yield {
            "town": town,
            "district": district_name,
            "expected": row.min_lot_size_gt,
            "actual": result.output.answer
            if result.output is not None
            else None,
            "correct_page_searched": any(gt_page & searched_pages_expanded),
            "correct_page_extracted": any(gt_page & extracted_pages),
            "gt_page": gt_page,
            "searched_pages": searched_pages,
            "searched_pages_expanded": searched_pages_expanded,
            "extracted_pages": extracted_pages,
        }

def main():
    gt = pd.read_csv(DATA_ROOT / "ground_truth.csv", index_col=["town", "district"])

    # TODO: Run this evaluation on other fields besides min_lot_size as well
    gt_min_lot = gt.query("~min_lot_size_gt.isna() & ~min_lot_size_page_gt.isna()")
    term = "min lot size"

    results = []
    for result in thread_map(lambda x: list(compute_eval_result(x[0][0], x[0][1], term, x[1])), gt_min_lot.iterrows(), total=len(gt_min_lot)):
        results.extend(result)

    results_df = pd.DataFrame(results)

    # Attempt to normalize LLM responses
    results_df = results_df.assign(
        actual_normalized=results_df.actual.apply(clean_string_units)
    ).explode("actual_normalized")
    # Explode expected values so that we have one row per expected-actual-value pair.
    results_df = results_df.assign(
        expected_normalized=results_df.expected.apply(
            lambda s: [float(f.strip()) for f in s.split(",")]
        )
    ).explode("expected_normalized")
    results_df = results_df.assign(correct_answer=results_df.actual_normalized == results_df.expected_normalized)

    results_df.to_csv(EVAL_OUTPUT_PATH, index=False)

    # groupby to calculate search page recall
    search_results_df = results_df.groupby(by=['town', 'district']).agg({"correct_page_searched": "sum"}).reset_index()
    search_results_df["correct"] = search_results_df["correct_page_searched"].apply(lambda x: 1 if x > 0 else 0)

    num_results = len(results_df)
    num_correct_page_searched = float(search_results_df["correct"].sum()) #len(results_df.query("correct_page_searched"))
    num_correct_page_extracted = len(results_df.query("correct_page_extracted"))
    num_correct_answer = len(results_df.query("correct_answer"))

    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        yaml.dump({
            "num_results": num_results,
            "num_correct_page_searched": num_correct_page_searched,
            "num_correct_page_extracted": num_correct_page_extracted,
            "num_correct_answer": num_correct_answer,
            "page_search_recall": num_correct_page_searched / search_results_df.shape[0],
            "page_extract_recall": num_correct_page_extracted / num_results,
            "answer_accuracy": num_correct_answer / num_results,
        }, f)



if __name__ == "__main__":
    main()
