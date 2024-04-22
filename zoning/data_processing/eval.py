import asyncio
from collections import defaultdict
import logging
from typing import Annotated, Any, Optional
import pandas as pd
import polars as pl
import typer
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from zoning.term_extraction.search.utils import page_coverage
from ..term_extraction.eval_results import clean_string_units
from ..term_extraction.extract import ExtractionMethod, extract_answer
from ..term_extraction.search import (
    SearchMethod,
    search_for_term,
)
from ..term_extraction.semantic_comparison import semantic_comparison
from ..term_extraction.types import District, LookupOutputConfirmed
from ..utils import get_project_root, flatten, logger
from .utils import AsyncTyper

DATA_ROOT = get_project_root() / "data"

EVAL_METRICS_PATH = DATA_ROOT / "results" / "eval.yaml"
EVAL_OUTPUT_PATH = DATA_ROOT / "results" / "eval.parquet"
SNAPSHOTS_DIR = DATA_ROOT / "results" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# If DEBUG=True, do not print rich tracking information
DEBUG = True


def calculate_verification_metrics(true_positives, false_positives, false_negatives):
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return recall, precision, f1


#async def compute_eval_result(
def compute_eval_result(
    town: str,
    district: District,
    districts: list[District],
    term: str,
    ground_truth: dict[str, Any],
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
    k: int,
    tournament_k: int,
):
    pages = search_for_term(town, district, term, search_method, k)
    expanded_pages = flatten(page_coverage(pages))
    outputs = extract_answer(
        pages=pages,
        term=term,
        town=town,
        district=district,
        districts=districts,
        method=extraction_method,
        # model_name="gpt-4",
        #model_name="gpt-4-1106-preview",  # getting better results with this
        model_name="gpt-4-turbo",
        tournament_k=tournament_k,
    )

    gt_page = ground_truth[f"{term}_page_gt"]
    if gt_page is None:
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))

    expected = ground_truth[f"{term}_gt"]
    is_empty = True

    #async for result in outputs:
    for result in outputs:
        is_empty = False
        extracted_pages = {r.page_number for r in result.search_pages}
        extracted_pages_expanded = set(result.search_pages_expanded)
        logger.info(f"Term {term} in {town} in {district.full_name} has searched_pages_expanded: {extracted_pages_expanded}")
        is_correct_page_searched = any(gt_page & set(expanded_pages))
        expected_extended = ground_truth[f"{term}_gt_orig"]
        label = result.search_pages[0].log["label"] if result.search_pages else ""

        base_output = {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": list(gt_page),
            "correct_page_searched": is_correct_page_searched,
            "expanded_pages": list(expanded_pages),
            "extracted_pages": list(extracted_pages),
            "extracted_pages_expanded": list(extracted_pages_expanded),
            "expected": expected,
            "expected_extended": expected_extended,
            "label": label,
            "pages": [p.page_number for p in pages],
            "confirmed_flag": None,
            "confirmed_raw": None,
            "actual_before_confirmation": None,
        }
        if extraction_method == ExtractionMethod.REDUCE_AND_CONFIRM and isinstance(result, LookupOutputConfirmed):
            # If we are using the REDUCE_AND_CONFIRM method, then the result class is LookupOutputConfirmed
            base_output["confirmed_flag"] = result.confirmed
            base_output["confirmed_raw"] = result.confirmed_raw
            base_output['actual_before_confirmation'] = result.original_output.answer
            base_output['rational_before_confirmation'] = result.original_output.rationale
            base_output['subquestions'] = result.subquestions
            base_output['extracted_district'] = result.subquestions['extracted_district']
            base_output['is_district_presented'] = result.subquestions['is_district_presented']
            base_output['extracted_term'] = result.subquestions['extracted_term']
            base_output['is_term_presented'] = result.subquestions['is_term_presented']
            base_output['is_correct_value_present'] = result.subquestions['is_correct_value_present']

        if result.output is None:
            yield {
                **base_output,
                "rationale": None,
                "extracted_text": None,
                "actual": None,

            }
        else:
            yield {
                **base_output,
                "rationale": result.output.rationale,
                "extracted_text": result.output.extracted_text,
                "actual": result.output.answer,
            }

    # Handle case when elastic search return 0 results
    if is_empty:
        yield {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": list(gt_page),
            "correct_page_searched": False,
            "expanded_pages": None,
            "searched_pages": None,
            "searched_pages_expanded": None,
            "expected": expected,
            "expected_extended": ground_truth[f"{term}_gt_orig"],
            "rationale": None,
            "extracted_text": None,
            "actual": None,
            "pages": None,
            "confirmed_flag": None,
            "confirmed_raw": None,
            "actual_before_confirmation": None,
        }


def compare_results(
    actual_normalized: float | None,
    actual_raw: str | None,
    expected: str | None,
    expected_extended: str | None,
    expected_extended_normalized: float | None,
) -> bool:
    if actual_raw is not None and expected is None and expected_extended is not None:
        # If no normalized expected answer exists, but an extended one does,
        # then compare the un-normalized answer from the LLM with our extended
        # ground truth using an LLM comparison.

        # TODO: If this returns true, then what we actually want to return to
        # the user is the raw answer, not the normalized one.
        # expected_extended_normalized is not none if expected_extended is not none
        return semantic_comparison(expected_extended, actual_raw) or actual_normalized == expected_extended_normalized
    else:
        # The correct answer is something simple (or nothing)
        return actual_normalized == expected


#async def evaluate_term(
def evaluate_term(
    term: str,
    gt: pl.DataFrame,
    progress: Progress,
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
    k: int,
    tournament_k: int,
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
        #async for result in compute_eval_result(
        for result in compute_eval_result(
            town, district, districts[town], term, row, search_method, extraction_method, k, tournament_k,
        ):
            results.append(result)
        progress.advance(eval_task)
    progress.update(eval_task, description=f"Evaluated {term}")

    # Load the data with schema overrides
    results_df = pl.from_dicts(results, schema_overrides={"expected_extended": pl.Utf8})

    # Normalize LLM responses
    results_df = results_df.with_columns(
        pl.col("actual").apply(clean_string_units).alias("actual_normalized"),
        pl.col("expected")
        .apply(
            lambda s: [float(f.strip()) for f in s.split(",")]
            if s is not None
            else [],
            skip_nulls=False,
        )
        .alias("expected_normalized"),
        pl.col("expected_extended").apply(clean_string_units).alias("expected_extended_normalized"),
    )

    # Explode values for one row per expected-actual-value pair
    results_df = results_df.explode("actual_normalized").explode("expected_normalized").explode(
        "expected_extended_normalized")

    # Apply comparison function to check for correct answers
    results_df = results_df.with_columns(
        pl.struct(
            [
                "actual",
                "actual_normalized",
                "expected_normalized",
                "expected_extended",
                "expected_extended_normalized"
            ]
        )
        .apply(
            lambda s: compare_results(
                s["actual_normalized"],
                s["actual"],
                s["expected_normalized"],
                s["expected_extended"],
                s["expected_extended_normalized"]
            )
        )
        .alias("correct_answer")
    )

    # Normalize 'actual_before_confirmation' and check if 'expected' exists
    results_df = results_df.with_columns(
        pl.col("actual_before_confirmation").apply(clean_string_units, skip_nulls=False).alias("actual_normalized"),
        pl.col("expected").apply(lambda x: x is not None, skip_nulls=False).alias("expected_exists"),
        pl.col("expected")
        .apply(
            lambda s: [float(f.strip()) for f in s.split(",")]
            if s is not None
            else [],
            skip_nulls=False,
        )
        .alias("expected_normalized"),
        pl.col("expected_extended").apply(clean_string_units, skip_nulls=False).alias("expected_extended_normalized"),
    )

    # Explode values again for one row per expected-actual-value pair after confirmation
    results_df = results_df.explode("actual_normalized").explode("expected_normalized").explode(
        "expected_extended_normalized")

    # Apply comparison function to check for correct answers before confirmation
    results_df = results_df.with_columns(
        pl.struct(
            [
                "actual_before_confirmation",
                "actual_normalized",
                "expected_normalized",
                "expected_extended",
                "expected_extended_normalized"
            ]
        )
        .apply(
            lambda s: compare_results(
                s["actual_normalized"],
                s["actual_before_confirmation"],
                s["expected_normalized"],
                s["expected_extended"],
                s["expected_extended_normalized"]
            ), skip_nulls=False
        )
        .alias("correct_answer_before_confirmation")
    )

    #
    # results_df = (
    #     pl.from_dicts(results, schema_overrides={"expected_extended": pl.Utf8})
    #     # Attempt to normalize LLM responses
    #     .with_columns(
    #         pl.col("actual").apply(clean_string_units).alias("actual_normalized"),
    #         pl.col("expected")
    #         .apply(
    #             lambda s: [float(f.strip()) for f in s.split(",")]
    #             if s is not None
    #             else [],
    #             skip_nulls=False,
    #         )
    #         .alias("expected_normalized"),
    #         pl.col("expected_extended").apply(clean_string_units).alias("expected_extended_normalized"),
    #     )
    #     # Explode all values so that we have one row per expected-actual-value pair.
    #     .explode("actual_normalized")
    #     .explode("expected_normalized")
    #     .explode("expected_extended_normalized")
    #     .with_columns(
    #         pl.struct(
    #             [
    #                 "actual",
    #                 "actual_normalized",
    #                 "expected_normalized",
    #                 "expected_extended",
    #                 "expected_extended_normalized"
    #             ]
    #         )
    #         .apply(
    #             lambda s: compare_results(
    #                 s["actual_normalized"],
    #                 s["actual"],
    #                 s["expected_normalized"],
    #                 s["expected_extended"],
    #                 s["expected_extended_normalized"]
    #             )
    #         )
    #         .alias("correct_answer")
    #     )
    #     # Only used for the REDUCE_AND_CONFIRM method
    #     .with_columns(
    #         pl.col("actual_before_confirmation").apply(clean_string_units, skip_nulls=False).alias("actual_normalized"),
    #         pl.col("expected").apply(lambda x: x is not None, skip_nulls=False).alias("expected_exists"),
    #         pl.col("expected")
    #         .apply(
    #             lambda s: [float(f.strip()) for f in s.split(",")]
    #             if s is not None
    #             else [],
    #             skip_nulls=False,
    #         )
    #         .alias("expected_normalized"),
    #         pl.col("expected_extended").apply(clean_string_units, skip_nulls=False).alias(
    #             "expected_extended_normalized"),
    #     )
    #     # Explode all values so that we have one row per expected-actual-value pair.
    #     .explode("actual_normalized")
    #     .explode("expected_normalized")
    #     .explode("expected_extended_normalized")
    #     .with_columns(
    #         pl.struct(
    #             [
    #                 "actual_before_confirmation",
    #                 "actual_normalized",
    #                 "expected_normalized",
    #                 "expected_extended",
    #                 "expected_extended_normalized"
    #             ]
    #         )
    #         .apply(
    #             lambda s: compare_results(
    #                 s["actual_normalized"],
    #                 s["actual_before_confirmation"],
    #                 s["expected_normalized"],
    #                 s["expected_extended"],
    #                 s["expected_extended_normalized"]
    #             ), skip_nulls=False
    #         )
    #         .alias("correct_answer_before_confirmation")
    #     )
    # )

    # groupby to calculate search page recall
    search_results_df = results_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_page_searched").sum(),
        pl.col("correct_answer").sum(),
    )

    # groupby to calculate search page recall
    search_results_df_before_confirmation = results_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_page_searched").sum(),
        pl.col("correct_answer_before_confirmation").sum(),
        pl.col("confirmed_flag").sum(),
    )

    # filter entries that have correct page searched and answered
    filtered_answer_page_df = results_df.filter(
        (pl.col("correct_page_searched") == True) & (pl.col("correct_answer") == True)
    )

    # groupby to calculate accuracy
    agg_answer_page_df = filtered_answer_page_df.groupby(
        pl.col("town", "district")
    ).agg(pl.col("correct_page_searched").sum(), pl.col("correct_answer").sum())

    num_results = len(results_df)
    num_correct_page_searched = len(
        search_results_df.filter(pl.col("correct_page_searched") > 0)
    )
    num_correct_answer = len(search_results_df.filter(pl.col("correct_answer") > 0))

    number_of_rows_with_gt_page = len(gt.filter(pl.col(f"{term}_page_gt").is_not_null()))
    logger.info(f"Number of rows with ground truth page: {number_of_rows_with_gt_page}")

    # groupby to calculate confirmation f1
    confirmation_df = results_df.groupby(pl.col("town", "district")).agg(
        pl.col("confirmed_flag").sum(),
        pl.col("expected_exists").sum(),
    )
    true_positives = len(confirmation_df.filter(
        (pl.col("expected_exists") > 0) & (pl.col("confirmed_flag") > 0)))
    false_positives = len(confirmation_df.filter(
        (pl.col("expected_exists") == 0) & (pl.col("confirmed_flag") > 0)))
    false_negatives = len(confirmation_df.filter(
        (pl.col("expected_exists") > 0) & (pl.col("confirmed_flag") == 0)))
    true_negatives = len(confirmation_df.filter(
        (pl.col("expected_exists") == 0) & (pl.col("confirmed_flag") == 0)))

    precision, recall, f1 = calculate_verification_metrics(true_positives, false_positives, false_negatives)

    eval_metrics = {
        #
        "num_results": num_results,
        "num_row_processed": len(search_results_df),
        "num_row_input": row_count,
        "num_correct_page_searched": num_correct_page_searched,
        "num_correct_answer": num_correct_answer,
        "row_processed": len(search_results_df) / row_count,
        "page_search_recall": num_correct_page_searched / number_of_rows_with_gt_page,
        # This is the answer accuracy conditional on the correct page having
        # been looked up by search
        "conditional_answer_accuracy": (
                len(agg_answer_page_df) / num_correct_page_searched
        ) if num_correct_page_searched != 0 else 0,
        "answer_accuracy": num_correct_answer / len(search_results_df),
        "answer_page_accuracy": (len(agg_answer_page_df) / len(search_results_df)),
    }
    if extraction_method == ExtractionMethod.REDUCE_AND_CONFIRM:
        eval_metrics.update({
            'e_true_positives': true_positives,
            'e_false_positives': false_positives,
            'e_false_negatives': false_negatives,
            'e_true_negatives': true_negatives,
            'e_confirmed_recall': recall,
            'e_confirmed_precision': precision,
            'e_confirmed_f1': f1,
        })
    return eval_metrics, results_df


def extract_districts():
    data = pd.read_excel(DATA_ROOT / "ct-data.xlsx")
    import pdb; pdb.set_trace()

#async def main(
def main(
    search_method: Annotated[SearchMethod, typer.Option()],
    extraction_method: Annotated[ExtractionMethod, typer.Option()],
    terms: Annotated[list[str], typer.Option()],
    k: Annotated[int, typer.Option()],
    # We must use Optional here because the "|" syntax can't be used by typer
    # yet for some reason.
    num_eval_rows: Annotated[Optional[int], typer.Option()] = None,
    tournament_k: Annotated[int, typer.Option()] = 1,
):
    raw_terms = terms
    terms = [i.split(",") for i in terms]
    terms = [i.strip() for i in flatten(terms)]
    logger.info(f"Term: {raw_terms} -> {terms}")
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

    districts = extract_districts()

    results_df = None
    # Run evaluation against entire ground truth for each term and aggregate all
    # results into one object.
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        disable=DEBUG,
    ) as progress:
        term_task = progress.add_task("Terms", total=len(terms))
        for term in terms:
            #metrics[term], new_results_df = await evaluate_term(
            metrics[term], new_results_df = evaluate_term(
                term, gt, progress, search_method, extraction_method, k, tournament_k,
                districts,
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
    snapshot_name = f"search-{search_method}_{extraction_method}_k={k}_tournament-k={tournament_k}_districts={num_eval_rows}"
    SNAPSHOT_PATH = SNAPSHOTS_DIR / f"{snapshot_name}.csv"
    SNAPSHOT_METRICS_PATH = SNAPSHOTS_DIR / f"{snapshot_name}.yaml"

    df = pd.read_parquet(EVAL_OUTPUT_PATH, engine="pyarrow")
    df.to_csv(SNAPSHOT_PATH, index=False)

    with open(SNAPSHOT_METRICS_PATH, "w") as file:
        yaml.dump(metrics, file)

    return metrics, df


if __name__ == "__main__":
    app = AsyncTyper(add_completion=False)
    app.command()(main)
    app()
