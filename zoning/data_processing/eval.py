from typing import Annotated, Optional, TypeVar

import pandas as pd
import typer
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..term_extraction.eval_results import clean_string_units
from ..term_extraction.extract import District, ExtractionMethod, extract_answer
from ..term_extraction.search import (
    SearchMethod,
    search_for_term,
)
from ..term_extraction.semantic_comparison import semantic_comparison
from ..utils import get_project_root
from .utils import AsyncTyper

DATA_ROOT = get_project_root() / "data"

EVAL_METRICS_PATH = DATA_ROOT / "results" / "eval.yaml"
EVAL_OUTPUT_PATH = DATA_ROOT / "results" / "eval.csv"

TVal = TypeVar("TVal")


def standardize_empty_val(val: TVal) -> TVal | None:
    return None if pd.isna(val) else val


async def compute_eval_result(
    town: str,
    district: District,
    term: str,
    ground_truth,
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
    k: int,
):
    pages = search_for_term(town, district, term, search_method, k)

    outputs = extract_answer(
        pages, term, district, method=extraction_method, model_name="gpt-4", k=k
    )
    gt_page = ground_truth[f"{term}_page_gt"]
    if pd.isna(gt_page):
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))

    expected = standardize_empty_val(ground_truth[f"{term}_gt"])

    async for result in outputs:
        searched_pages = {r.page_number for r in result.search_pages}
        searched_pages_expanded = set(result.search_pages_expanded)

        base_output = {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": gt_page,
            "searched_pages": searched_pages,
            "searched_pages_expanded": searched_pages_expanded,
        }

        if result.output is None:
            yield {
                **base_output,
                # For determining the correct page, we consider the page to be
                # correct if the ground truth was also blank and GPT did not return
                # an answer. Note that search always returns some page, so we ignore
                # that result as long as GPT ignored it.
                "correct_page_searched": len(gt_page) == 0,
            }
        else:
            yield {
                **base_output,
                "rationale": result.output.rationale,
                "extracted_text": result.output.extracted_text,
                "actual": result.output.answer,
                "expected": expected,
                "expected_extended": ground_truth[f"{term}_gt_orig"],
                "correct_page_searched": any(gt_page & searched_pages_expanded),
            }


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
    k: int,
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
            town, district, term, row, search_method, extraction_method, k
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
                "correct_answer": "sum",
            }
        )
        .reset_index()
    )

    num_results = len(results_df)
    num_correct_page_searched = len(
        search_results_df.query("correct_page_searched > 0")
    )
    num_correct_answer = len(search_results_df.query("correct_answer > 0"))

    return {
        "num_results": num_results,
        "num_correct_page_searched": num_correct_page_searched,
        "num_correct_answer": num_correct_answer,
        "page_search_recall": num_correct_page_searched / len(search_results_df),
        # This is the answer accuracy conditional on the correct page having
        # been looked up by search
        "conditional_answer_accuracy": (
            len(
                search_results_df.query(
                    "correct_page_searched > 0 & correct_answer > 0"
                )
            )
            / num_correct_page_searched
        )
        if num_correct_page_searched != 0
        else 0,
        "answer_accuracy": num_correct_answer / len(search_results_df),
    }, results_df


async def main(
    search_method: Annotated[SearchMethod, typer.Option()],
    extraction_method: Annotated[ExtractionMethod, typer.Option()],
    terms: Annotated[list[str], typer.Option()],
    k: Annotated[int, typer.Option()],
    # We must use Optional here because the "|" syntax can't be used by typer
    # yet for some reason.
    num_eval_rows: Annotated[Optional[int], typer.Option()] = None,
):
    metrics = {}

    # Load Ground Truth
    gt = pd.read_csv(
        DATA_ROOT / "ground_truth.csv",
        index_col=["town", "district"],
        dtype={
            **{f"{tc}_gt": str for tc in terms},
            **{f"{tc}_page_gt": str for tc in terms},
        },
        nrows=num_eval_rows,
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
                term, gt, progress, search_method, extraction_method, k
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
    app = AsyncTyper(add_completion=False)
    app.command()(main)
    app()
