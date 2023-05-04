import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml
from rich.progress import Progress

from ..prompting.eval_results import clean_string_units
from ..prompting.extract import ExtractionMethod, extract_size
from ..utils import get_project_root

DATA_ROOT = get_project_root() / "data"

EVAL_METRICS_PATH = DATA_ROOT / "results" / "eval.yaml"
EVAL_OUTPUT_PATH = DATA_ROOT / "results" / "eval.csv"


def compute_eval_result(town: str, district_name: str, term: str, term_code: str, row):
    outputs = extract_size(
        town,
        dict(T=district_name, Z=row.district_abb),
        term,
        6,
        method=ExtractionMethod.MAP,
        model_name="gpt-4",
    )
    gt_page = set(map(int, str(row[f"{term_code}_page_gt"]).split(",")))
    for result in outputs:
        searched_pages = {r.page_number for r in result.search_pages}
        searched_pages_expanded = set(result.search_pages_expanded)

        extracted_pages = (
            set(result.output.pages) if result.output is not None else set()
        )

        yield {
            "town": town,
            "district": district_name,
            "term": term_code,
            "confidence": result.output.confidence
            if result.output is not None
            else 0.0,
            "expected": row[f"{term_code}_gt"],
            "actual": result.output.answer if result.output is not None else None,
            "correct_page_searched": int(any(gt_page & searched_pages_expanded)),
            "correct_page_extracted": int(any(gt_page & extracted_pages)),
            "gt_page": gt_page,
            "searched_pages": searched_pages,
            "searched_pages_expanded": searched_pages_expanded,
            "extracted_pages": extracted_pages,
        }


def evaluate_term(term: str, term_code: str, gt: pd.DataFrame, progress: Progress):
    gt_term = gt.query(f"~{term_code}_gt.isna() & ~{term_code}_page_gt.isna()")
    eval_task = progress.add_task(f"Evaluating {term}", total=len(gt_term))

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for result in executor.map(
            lambda x: list(
                compute_eval_result(x[0][0], x[0][1], term, term_code, x[1])
            ),
            gt_term.iterrows(),
        ):
            progress.advance(eval_task)
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
    results_df = results_df.assign(
        correct_answer=results_df.actual_normalized == results_df.expected_normalized
    )

    # groupby to calculate search page recall
    search_results_df = (
        results_df.groupby(by=["town", "district"])
        .agg(
            {
                "correct_page_searched": "sum",
                "correct_page_extracted": "sum",
                "correct_answer": "sum",
            }
        )
        .reset_index()
    )

    num_results = len(results_df)
    num_correct_page_searched = float(
        search_results_df["correct_page_searched"]
        .apply(lambda x: 1 if x > 0 else 0)
        .sum()
    )
    num_correct_page_extracted = float(
        search_results_df["correct_page_extracted"]
        .apply(lambda x: 1 if x > 0 else 0)
        .sum()
    )
    num_correct_answer = float(
        search_results_df["correct_answer"].apply(lambda x: 1 if x > 0 else 0).sum()
    )

    return {
        "num_results": num_results,
        "num_correct_page_searched": num_correct_page_searched,
        "num_correct_page_extracted": num_correct_page_extracted,
        "num_correct_answer": num_correct_answer,
        "page_search_recall": num_correct_page_searched / len(search_results_df),
        "page_extract_recall": num_correct_page_extracted / len(search_results_df),
        # This is the answer accuracy conditional on the correct page having been looked up by ES
        "conditional_answer_accuracy": len(
            search_results_df.query("correct_page_searched > 0 & correct_answer > 0")
        )
        / num_correct_page_searched,
        "answer_accuracy": num_correct_answer / len(search_results_df),
    }, results_df


def main():
    gt = pd.read_csv(DATA_ROOT / "ground_truth.csv", index_col=["town", "district"])

    terms = ["min lot size", "min unit size"]  # update to list of terms you want to run

    terms_code = [term.replace(" ", "_") for term in terms]
    first = True

    metrics = {}

    with Progress() as progress:
        term_task = progress.add_task("Terms", total=len(terms))
        for i, term in enumerate(terms):
            metrics[term], results_df = evaluate_term(term, terms_code[i], gt, progress)
            results_df.to_csv(
                EVAL_OUTPUT_PATH, index=False, mode="w" if first else "a", header=first
            )
            progress.advance(term_task)
            first = False

    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    main()
