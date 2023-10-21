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

from ..term_extraction.extract import TournamentTester 
import pickle
import os 

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
    if os.path.exists(district.short_name+"map_results.dat"):
        # for each district, we read in the outputs 
        with open(district.short_name+"map_results.dat", "rb") as f:
                outputs = pickle.load(f)
    else:
        # need to run map 
        pages = search_for_term(town, district, term, search_method, k)
        outputs = extract_answer(
        pages, term, district, method="map", model_name="gpt-4",
        tournament_k=tournament_k
    )
        with open(district.short_name+"map_results.dat", "wb") as f:
            pickle.dump(outputs, f)

    print("DISTRICT IS: ", str(district), "TOWN IS: ", town, "NUM MAP RESULTS IS: ", str(len(outputs)) )
    gt_page = ground_truth[f"{term}_page_gt"]
    if gt_page is None:
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))


    expected = ground_truth[f"{term}_gt"]
    is_empty = True

    correct_answer = None
    # for result in map outputs :
        # get that result, and get the value from it 
        # do semantic comparison between that and the expected value 
        # if the semantic comparison is true, use that result object as a ~ correct answer ~ 
    if expected:
        for result in outputs:
            if result.output:
                gpt_answer = result.output.answer
                if semantic_comparison(expected, gpt_answer):
                    correct_answer = result 


        # for result in map outputs : 
            # pass that result and the ~correct answer~ in to the tournament prompt 
        if correct_answer:
            tournament_test = TournamentTester("gpt-4", tournament_k)
            indices = await tournament_test.extract(outputs, district, term, correct_answer)

            print("hi im hereeeee")
            return indices
    return []

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
    row_count = len(gt)
    for row in gt.iter_rows(named=True):
        town = row["town"]
        district = District(full_name=row["district"], short_name=row["district_abb"])
        progress.update(
            eval_task, description=f"Evaluating {term}, {town}, {district.full_name}"
        )
        for result in await compute_eval_result(
            town, district, term, row, search_method, extraction_method,
            k, tournament_k
        ):
            results.append(result)
        progress.advance(eval_task)
    progress.update(eval_task, description=f"Evaluated {term}")


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

    # Page Recall: #Correct Page / #Unique town
    # Answer Accuracy: #Correct Answer / #Unique town
    # Conditional Accuracy: #(Correct Page & Correct Answer)/ #Correct page
    # Accuracy: #(Correct Page & Correct Answer) / #Unique town
    metrics["answer_accuracy"] = sum(
        metrics[term]["answer_accuracy"] for term in terms
    ) / len(terms)

    metrics["page_search_recall"] = sum(
        metrics[term]["page_search_recall"] for term in terms
    ) / len(terms)

    metrics["conditional_answer_accuracy"] = sum(
        metrics[term]["conditional_answer_accuracy"] for term in terms
    ) / len(terms)

    metrics["answer_page_accuracy"] = sum(
        metrics[term]["answer_page_accuracy"] for term in terms
    ) / len(terms)

    metrics["row_processed"] = sum(
        metrics[term]["row_processed"] for term in terms
    ) / len(terms)

    assert results_df is not None

    results_df.write_parquet(EVAL_OUTPUT_PATH)

    with EVAL_METRICS_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(metrics, f)


    # Save snapshot locally
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


