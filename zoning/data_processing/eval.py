from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..term_extraction.eval_results import clean_string_units
from ..term_extraction.extract import ExtractionMethod, extract_size
from ..term_extraction.semantic_comparison import semantic_comparison
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
    gt_page = row[f"{term_code}_page_gt"]
    if pd.isna(gt_page):
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))

    expected = row[f"{term_code}_gt"]

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
            "expected": expected if not pd.isna(expected) else None,
            "expected_extended": row[f"{term_code}_gt_orig"],
            "actual": result.output.answer if result.output is not None else None,
            # For determining the correct page, we consider the page to be
            # correct if the ground truth was also blank and GPT did not return
            # an answer. Note that search always returns some page, so we ignore
            # that result as long as GPT ignored it.
            "correct_page_searched": int(
                any(gt_page & searched_pages_expanded)
                or (len(gt_page) == 0 and result.output is None)
            ),
            "correct_page_extracted": int(
                any(gt_page & extracted_pages)
                or (len(gt_page) == 0 and len(extracted_pages) == 0)
            ),
            "gt_page": gt_page,
            "searched_pages": searched_pages,
            "searched_pages_expanded": searched_pages_expanded,
            "extracted_pages": extracted_pages,
        }


def compare_results(
    actual_normalized: float | None, actual_raw: str | None, expected: str | None, expected_extended: str | None
) -> bool:
    # Normalize responses to None if they are any pandas empty value.
    actual_raw = None if pd.isna(actual_raw) else actual_raw
    actual_normalized = None if pd.isna(actual_normalized) else actual_normalized
    expected = None if pd.isna(expected) else expected
    expected_extended = None if pd.isna(expected_extended) else expected_extended

    if actual_raw is not None and expected is None and expected_extended is not None:
        # If no normalized expected answer exists, but an extended one does,
        # then compare the un-normalized answer from the LLM with our extended
        # ground truth using an LLM comparison. 

        # TODO: If this returns true, then what we actually want to return to
        # the user is the raw answer, not the normalized one.
        return semantic_comparison(expected_extended, actual_raw)
    else:
        # The correct answer is something simple (or nothing)
        return actual_normalized == expected


def evaluate_term(term: str, term_code: str, gt: pd.DataFrame, progress: Progress):
    eval_task = progress.add_task(f"Evaluating {term}", total=len(gt))

    result_fs = []
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for index, row in gt.iterrows():
            town, district = index
            result_fs.append(
                executor.submit(
                    compute_eval_result, town, district, term, term_code, row
                )
            )

        for result in as_completed(result_fs):
            results.extend(result.result())
            progress.advance(eval_task)

    results_df = pd.DataFrame(results)

    # Attempt to normalize LLM responses
    # Explode all values so that we have one row per expected-actual-value pair.
    results_df = (
        results_df.assign(
            actual_normalized=results_df.actual.apply(clean_string_units),
            expected_normalized=results_df.expected.apply(
                lambda s: [float(f.strip()) for f in s.split(",")]
                if s is not None
                else []
            ),
        )
        .explode("actual_normalized")
        .explode("expected_normalized")
    )
    results_df = results_df.assign(
        # NaN != NaN, so we need to replace this with alternative values to correctly compare fields where no answer is expected.
        # correct_answer=results_df.actual_normalized.fillna("N/A").eq(results_df.expected_normalized.fillna("N/A"))
        correct_answer=results_df.apply(
            lambda row: compare_results(
                row.actual_normalized, row.actual, row.expected_normalized, row.expected_extended
            ),
            axis=1,
        )
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
    num_correct_page_searched = len(
        search_results_df.query("correct_page_searched > 0")
    )
    num_correct_page_extracted = len(
        search_results_df.query("correct_page_extracted > 0")
    )
    num_correct_answer = len(search_results_df.query("correct_answer > 0"))

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

    terms = [
        "min lot size",
        "min unit size",
        "max height",
        "max lot coverage",
        "max lot coverage pavement",
    ]  # update to list of terms you want to run

    terms_code = [term.replace(" ", "_") for term in terms]
    metrics = {}

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        term_task = progress.add_task("Terms", total=len(terms))
        for i, term in enumerate(terms):
            metrics[term], results_df = evaluate_term(term, terms_code[i], gt, progress)
            results_df.to_csv(
                EVAL_OUTPUT_PATH,
                index=False,
                mode="w" if i == 0 else "a",
                header=i == 0,
            )
            progress.advance(term_task)

    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    main()
