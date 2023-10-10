import asyncio
from typing import Annotated, Any, Optional
import pandas as pd
import polars as pl
import typer
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..term_extraction.eval_results import clean_string_units
from ..term_extraction.extract import ExtractionMethod, extract_answer
from ..term_extraction.search import (
    SearchMethod,
    search_for_term,
)
from ..term_extraction.semantic_comparison import semantic_comparison
from ..term_extraction.types import District
from ..utils import get_project_root
from .utils import AsyncTyper

DATA_ROOT = get_project_root() / "data"

EVAL_METRICS_PATH = DATA_ROOT / "results" / "eval.yaml"
EVAL_OUTPUT_PATH = DATA_ROOT / "results" / "eval.parquet"


async def compute_eval_result(
    town: str,
    district: District,
    term: str,
    ground_truth: dict[str, Any],
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
    k: int,
    tournament_k: int
):
    pages = search_for_term(town, district, term, search_method, k)
    outputs = extract_answer(
        pages, term, district, method=extraction_method, model_name="gpt-4",
        tournament_k=tournament_k
    )

    gt_page = ground_truth[f"{term}_page_gt"]
    if gt_page is None:
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))

    expected = ground_truth[f"{term}_gt"]
    is_empty = True

    async for result in outputs:
        is_empty = False
        searched_pages = {r.page_number for r in result.search_pages}
        searched_pages_expanded = set(result.search_pages_expanded)

        base_output = {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": list(gt_page),
            "searched_pages": list(searched_pages),
            "searched_pages_expanded": list(searched_pages_expanded),
            "expected": expected,
            "expected_extended": ground_truth[f"{term}_gt_orig"],
        }

        if result.output is None:
            yield {
                **base_output,
                "rationale": None,
                "extracted_text": None,
                "actual": None,
                # For determining the correct page, we consider the page to be
                # correct if the ground truth was also blank and GPT did not return
                # an answer. Note that search always returns some page, so we ignore
                # that result as long as GPT ignored it.
                "correct_page_searched": expected is None,
            }
        else:
            yield {
                **base_output,
                "rationale": result.output.rationale,
                "extracted_text": result.output.extracted_text,
                "actual": result.output.answer,
                "correct_page_searched": any(gt_page & searched_pages_expanded),
            }

    if is_empty:
        yield {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": list(gt_page),
            "searched_pages": None,
            "searched_pages_expanded": None,
            "expected": expected,
            "expected_extended": ground_truth[f"{term}_gt_orig"],
            "rationale": None,
            "extracted_text": None,
            "actual": None,
            "correct_page_searched": expected is None,
        }

def compare_results(
    actual_normalized: float | None,
    actual_raw: str | None,
    expected: str | None,
    expected_extended: str | None,
) -> bool:
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
    gt: pl.DataFrame,
    progress: Progress,
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
    k: int,
    tournament_k: int
):
    eval_task = progress.add_task(f"Evaluating {term}", total=len(gt))

    # Generate results for the given term in parallel, showing progress along
    # the way.
    results = []
    row_count = 0
    for row in gt.iter_rows(named=True):
        town = row["town"]
        district = District(full_name=row["district"], short_name=row["district_abb"])
        progress.update(
            eval_task, description=f"Evaluating {term}, {town}, {district.full_name}"
        )
        async for result in compute_eval_result(
            town, district, term, row, search_method, extraction_method,
            k, tournament_k
        ):
            results.append(result)
        progress.advance(eval_task)
        row_count += 1
    progress.update(eval_task, description=f"Evaluated {term}")

    results_df = (
        pl.from_dicts(results, schema_overrides={"expected_extended": pl.Utf8})
        # Attempt to normalize LLM responses
        .with_columns(
            pl.col("actual").apply(clean_string_units).alias("actual_normalized"),
            pl.col("expected")
            .apply(
                lambda s: [float(f.strip()) for f in s.split(",")]
                if s is not None
                else [],
                skip_nulls=False
            )
            .alias("expected_normalized"),
        )
        # Explode all values so that we have one row per expected-actual-value pair.
        .explode("actual_normalized")
        .explode("expected_normalized")
        .with_columns(
            pl.struct(
                [
                    "actual",
                    "actual_normalized",
                    "expected_normalized",
                    "expected_extended",
                ]
            )
            .apply(
                lambda s: compare_results(
                    s["actual_normalized"],
                    s["actual"],
                    s["expected_normalized"],
                    s["expected_extended"],
                )
            )
            .alias("correct_answer")
        )
    )

    # groupby to calculate search page recall
    search_results_df = results_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_page_searched").sum(),
        pl.col("correct_answer").sum(),
    )

    num_results = len(results_df)
    num_correct_page_searched = len(
        search_results_df.filter(pl.col("correct_page_searched") > 0)
    )
    num_correct_answer = len(search_results_df.filter(pl.col("correct_answer") > 0))

    return {
        "num_results": num_results,
        "num_row_processed": len(search_results_df),
        "num_row_input": row_count,
        "num_correct_page_searched": num_correct_page_searched,
        "num_correct_answer": num_correct_answer,
        "row_processed": len(search_results_df) / row_count,
        "page_search_recall": num_correct_page_searched / len(search_results_df),
        # This is the answer accuracy conditional on the correct page having
        # been looked up by search
        "conditional_answer_accuracy": (
            len(
                search_results_df.filter(
                    pl.all_horizontal(pl.col("correct_page_searched", "correct_answer"))
                    > 0
                )
            )
            / num_correct_page_searched
        )
        if num_correct_page_searched != 0
        else 0,
        "answer_accuracy": num_correct_answer / len(search_results_df),
        "accuracy": (len(
                search_results_df.filter(
                    pl.all_horizontal(pl.col("correct_page_searched", "correct_answer"))
                    > 0
                )
            ) / len(search_results_df))
    }, results_df


async def main(
    search_method: Annotated[SearchMethod, typer.Option()],
    extraction_method: Annotated[ExtractionMethod, typer.Option()],
    terms: Annotated[list[str], typer.Option()],
    k: Annotated[int, typer.Option()],
    # We must use Optional here because the "|" syntax can't be used by typer
    # yet for some reason.
    num_eval_rows: Annotated[Optional[int], typer.Option()] = None,
    tournament_k: Annotated[int, typer.Option()] = 1,
):


    metrics = {}

    # Load Ground Truth
    gt = pl.read_csv(
        DATA_ROOT / "ground_truth.csv",
        dtypes={
            **{f"{tc}_gt": pl.Utf8 for tc in terms},
            **{f"{tc}_page_gt": pl.Utf8 for tc in terms},
        },
        n_rows=num_eval_rows,
    )
    results_df = None
    # Run evaluation against entire ground truth for each term and aggregate all
    # results into one object.
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        term_task = progress.add_task("Terms", total=len(terms))
        for term in terms:
            metrics[term], new_results_df = await evaluate_term(
                term, gt, progress, search_method, extraction_method,
                k, tournament_k
            )
            if results_df is not None:
                results_df = pl.concat((results_df, new_results_df))
            else:
                results_df = new_results_df

            progress.advance(term_task)

    # Compute metrics aggregated across terms
    metrics["answer_accuracy"] = sum(
        metrics[term]["answer_accuracy"] for term in terms
    ) / len(terms)

    metrics["page_search_recall"] = sum(
        metrics[term]["page_search_recall"] for term in terms
    ) / len(terms)

    metrics["conditional_answer_accuracy"] = sum(
        metrics[term]["conditional_answer_accuracy"] for term in terms
    ) / len(terms)

    metrics["accuracy"] = sum(
        metrics[term]["accuracy"] for term in terms
    ) / len(terms)

    metrics["row_processed"] = sum(
        metrics[term]["row_processed"] for term in terms
    ) / len(terms)

    assert results_df is not None

    results_df.write_parquet(EVAL_OUTPUT_PATH)

    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(metrics, f)

    SNAPSHOT_PATH = str(search_method) + "_" + str(extraction_method) + "_" + str(k) + "_" + str(tournament_k) + ".csv"
    SNAPSHOT_METRICS_PATH = str(search_method) + "_" + str(extraction_method) + "_" + str(k) + "_" + str(tournament_k) + ".yaml"
    df = pd.read_parquet(EVAL_OUTPUT_PATH, engine='pyarrow')
    df.to_csv(SNAPSHOT_PATH, index=False)

    with open(SNAPSHOT_METRICS_PATH, "w") as file:
        yaml.dump(metrics, file)


if __name__ == "__main__":
    app = AsyncTyper(add_completion=False)
    app.command()(main)
    app()


