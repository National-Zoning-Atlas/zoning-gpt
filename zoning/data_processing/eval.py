import asyncio
from typing import TypeVar

import pandas as pd
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..term_extraction.eval_results import clean_string_units
from ..term_extraction.extract import District, ExtractionMethod, extract_answer
from ..term_extraction.search import (
    nearest_pages,
    get_non_overlapping_chunks,
    SearchMethod,
)
from ..term_extraction.semantic_comparison import semantic_comparison
from ..utils import get_project_root

DATA_ROOT = get_project_root() / "data"

EVAL_METRICS_PATH = DATA_ROOT / "results" / "eval.yaml"
EVAL_OUTPUT_PATH = DATA_ROOT / "results" / "eval.csv"


async def compute_eval_result(
    town: str,
    district: District,
    term: str,
    row,
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
):
    pages = list(nearest_pages(town, district, term, search_method))

    if search_method != SearchMethod.NO_SEARCH:
        pages = get_non_overlapping_chunks(pages)[:6]

    outputs = await extract_answer(
        pages, term, district, method=extraction_method, model_name="gpt-4"
    )
    gt_page = row[f"{term}_page_gt"]
    if pd.isna(gt_page):
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))

    expected = row[f"{term}_gt"]

    for result in outputs:
        searched_pages = {r.page_number for r in result.search_pages}
        searched_pages_expanded = set(result.search_pages_expanded)

        extracted_pages = (
            set(result.output.pages) if result.output is not None else set()
        )

        yield {
            "town": town,
            "district": district.full_name,
            "term": term,
            "expected": expected if not pd.isna(expected) else None,
            "expected_extended": row[f"{term}_gt_orig"],
            "actual": result.output.answer if result.output is not None else None,
            "confidence": result.output.confidence
            if result.output is not None
            else 0.0,
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


TVal = TypeVar("TVal")


def standardize_empty_val(val: TVal) -> TVal | None:
    return None if pd.isna(val) else val


def compare_results(
    actual_normalized: float | None,
    actual_raw: str | None,
    expected: str | None,
    expected_extended: str | None,
) -> bool:
    # Normalize responses to None if they are any pandas empty value.
    actual_raw = standardize_empty_val(actual_raw)
    actual_normalized = standardize_empty_val(actual_normalized)
    expected = standardize_empty_val(expected)
    expected_extended = standardize_empty_val(expected_extended)

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


async def evaluate_term(
    term: str,
    gt: pd.DataFrame,
    progress: Progress,
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
):
    eval_task = progress.add_task(f"Evaluating {term}", total=len(gt))

    # Generate results for the given term in parallel, showing progress along
    # the way.
    results = []
    for index, row in gt.iterrows():
        town, district = index
        progress.update(eval_task, description=f"Evaluating {term}, {town}, {district}")
        district = District(full_name=district, short_name=row.district_abb)
        async for result in compute_eval_result(
            town, district, term, row, search_method, extraction_method
        ):
            results.append(result)
        progress.advance(eval_task)
    progress.update(eval_task, description=f"Evaluated {term}")

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
        correct_answer=results_df.apply(
            lambda row: compare_results(
                row.actual_normalized,
                row.actual,
                row.expected_normalized,
                row.expected_extended,
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


async def main():
    terms = [
        "min_lot_size",
        # "min_unit_size",
        # "max_height",
        "max_lot_coverage",
        # "max_lot_coverage_pavement",
        "min_parking_spaces",
    ]  # update to list of terms you want to run

    search_method = SearchMethod.NO_SEARCH
    extract_method = ExtractionMethod.MAP

    metrics = {}

    # Load Ground Truth
    gt = pd.read_csv(
        DATA_ROOT / "ground_truth.csv",
        index_col=["town", "district"],
        dtype={
            **{f"{tc}_gt": str for tc in terms},
            **{f"{tc}_page_gt": str for tc in terms},
        },
        nrows=5,
    )

    # Run evaluation against entire ground truth for each term and aggregate all
    # results into one object.
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        term_task = progress.add_task("Terms", total=len(terms))
        for i, term in enumerate(terms):
            metrics[term], results_df = await evaluate_term(
                term, gt, progress, search_method, extract_method
            )
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
    asyncio.run(main())
