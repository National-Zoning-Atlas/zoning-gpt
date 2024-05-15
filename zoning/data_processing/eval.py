import asyncio
from collections import defaultdict
import logging
from typing import Annotated, Any, Optional
import re
import pandas as pd
import polars as pl
import typer
import yaml
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import json

from zoning.term_extraction.search.utils import page_coverage
from ..term_extraction.eval_results import clean_string_units
from ..term_extraction.extract import ExtractionMethod, extract_answer
from ..term_extraction.search import (
    SearchMethod,
    search_for_term,
)
from ..term_extraction.semantic_comparison import semantic_comparison
from ..term_extraction.to_json import to_json
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


def calculate_f1(true_positives, false_positives, false_negatives):
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return recall, precision, f1


async def compute_eval_result(
        #def compute_eval_result(
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
        #model_name="gpt-4-turbo",
        model_name="gpt-4o",
        tournament_k=tournament_k,
    )

    gt_page = ground_truth[f"{term}_page_gt"]
    if gt_page is None:
        # No ground truth page
        gt_page = set()
    else:
        gt_page = set(map(int, str(gt_page).split(",")))

    # true answer
    true_answer = ground_truth[f"{term}_gt"]
    is_empty = True

    async for result in outputs:
        #for result in outputs:
        is_empty = False
        extracted_pages = {r.page_number for r in result.search_pages}
        extracted_pages_expanded = set(result.search_pages_expanded)
        logger.info(
            f"Term {term} in {town} in {district.full_name} has searched_pages_expanded: {extracted_pages_expanded}")
        # this will be true for all chunks
        is_correct_page_searched = any(gt_page & set(expanded_pages))
        this_correct_page_searched = any(gt_page & set(extracted_pages_expanded))
        true_answer_extended = ground_truth[f"{term}_gt_orig"]
        label = result.search_pages[0].log["label"] if result.search_pages else ""

        base_output = {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": list(gt_page),
            "correct_page_searched": is_correct_page_searched,
            "this_correct_page_searched": this_correct_page_searched,
            "expanded_pages": list(expanded_pages),
            "extracted_pages": list(extracted_pages),
            "extracted_pages_expanded": list(extracted_pages_expanded),
            "true_answer": true_answer,
            "true_answer_extended": true_answer_extended,
            "label": label,
            "pages": [p.page_number for p in pages],
            "confirmed_flag": None,
            "confirmed_raw": None,
            "predicted_answer_before_confirmation": None,
        }
        if extraction_method == ExtractionMethod.REDUCE_AND_CONFIRM and isinstance(result, LookupOutputConfirmed):
            # If we are using the REDUCE_AND_CONFIRM method, then the result class is LookupOutputConfirmed
            base_output["confirmed_flag"] = result.confirmed
            base_output["confirmed_raw"] = result.confirmed_raw
            base_output['predicted_answer_before_confirmation'] = result.original_output.answer
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
                "predicted_answer": None,

            }
        else:
            yield {
                **base_output,
                "rationale": result.output.rationale,
                "extracted_text": result.output.extracted_text,
                "predicted_answer": result.output.answer,
            }

    # Handle case when elastic search return 0 results
    if is_empty:
        yield {
            "town": town,
            "district": district.full_name,
            "term": term,
            "gt_page": list(gt_page),
            "correct_page_searched": False,
            "this_correct_page_searched": False,
            "expanded_pages": None,
            "searched_pages": None,
            "searched_pages_expanded": None,
            "true_answer": true_answer,
            "true_answer_extended": ground_truth[f"{term}_gt_orig"],
            "rationale": None,
            "extracted_text": None,
            "predicted_answer": None,
            "pages": None,
            "confirmed_flag": None,
            "confirmed_raw": None,
            "predicted_answer_before_confirmation": None,
        }

def compare_json(x, y):
    # y: {
    #     "value-type": {
    #         "unit": value
    #     }
    # }
    # every value-type key in y must have a match with a value-type in x
    equals = True
    for k, yvalues in y.items():
        if k not in x:
            return False
        xvalues = x[k]
        # the keys are now the units
        unit_match = False
        for unit, yv in yvalues.items():
            if unit == "none":
                # compare against everything in x
                unit_match |= any(yv == xv for xv in xvalues.values())
            else:
                unit_match |= unit in xvalues and yv == xvalues[unit]
            # TODO: unit conversions
        equals &= unit_match
    return equals


def compare_answers(predicted, true_answer1, true_answer2):
    true_is_none = true_answer1 is None and true_answer2 is None
    all_none = predicted == "None" and true_is_none
    if all_none: return True
    if predicted == "None" and not true_is_none: return False

    # structured check if answers match
    # convert all answers to json, then check json fields
    predicted_json = to_json(predicted)

    pred_equals_answer1 = False
    pred_equals_answer2 = False
    if true_answer1 is not None:
        true_answer1_json = to_json(true_answer1)
        pred_equals_answer1 = compare_json(predicted_json, true_answer1_json)
    if true_answer2 is not None:
        true_answer2_json = to_json(true_answer2)
        pred_equals_answer2 = compare_json(predicted_json, true_answer2_json)

    return pred_equals_answer1 or pred_equals_answer2

def get_metrics(results_df):
    """
    We need to compute
    1. page search recall
    2. answer accuracy
    3. answer + page accuracy
    4. prec/rec/f1 of answers
    5. answer accuracy | correct page
    """

    # 1. page search recall
    search_results_df = results_df.groupby(["town", "district"]).agg([
        pl.col("this_correct_page_searched").sum(),
        pl.col("gt_page").list.lengths().sum()
    ])

    page_search_correct = len(
        search_results_df.filter(pl.col("this_correct_page_searched") > 0)
    )
    page_search_exists = len(
        search_results_df.filter(pl.col("gt_page") > 0)
    )
    page_search_recall = page_search_correct / page_search_exists

    # 2. answer accuracy
    answers_df = results_df.with_columns(
        pl.struct(["predicted_answer", "true_answer_extended", "true_answer"])
        .apply(lambda x:
            #semantic_comparison(x["predicted_answer"], x["true_answer_extended"])
            #or semantic_comparison(x["predicted_answer"], x["true_answer"])
            compare_answers(x["predicted_answer"], x["true_answer_extended"], x["true_answer"])
        )
        .alias("correct_answer")
    )
    answers_results_df = answers_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_answer").sum(),
    )
    answer_correct = len(answers_results_df.filter(pl.col("correct_answer") > 0))

    # 3. answer + page accuracy
    answers_page_df = answers_df.with_columns(
        pl.struct(["correct_answer", "this_correct_page_searched"])
        .apply(lambda x: x["correct_answer"] and x["this_correct_page_searched"])
        .alias("correct_answer_and_page")
    )
    answers_page_results_df = answers_page_df.groupby(pl.col("town", "district")).agg(
        pl.col("correct_answer_and_page").sum(),
    )
    answer_page_correct = len(answers_page_results_df.filter(pl.col("correct_answer_and_page") > 0))

    # 4. answer prec/rec/f1
    # does there exist an answer when the correct page is found?
    pr_answers_df = answers_df.with_columns(
        # is there an answer on this page?
        pl.struct(["this_correct_page_searched", "predicted_answer"])
        .apply(lambda x: x["predicted_answer"] != "None")
        .alias("predicted_positive")
    ).with_columns(
        # did we correctly predict that a page has an answer?
        pl.struct(["this_correct_page_searched", "true_answer_extended", "true_answer", "predicted_answer"])
        .apply(lambda x:
            x["this_correct_page_searched"]
            and (x["true_answer_extended"] is not None or x["true_answer"] is not None)
            and x["predicted_answer"] != "None"
         )
        .alias("true_predicted_positive")
    ).with_columns(
        # was there an answer on the page?
        pl.struct(["this_correct_page_searched", "true_answer"])
        .apply(lambda x: x["this_correct_page_searched"] and x["true_answer"] is not "None")
        .alias("positive")
    ).with_columns(
        # did we incorrectly predict there was no answer?
        pl.struct(["this_correct_page_searched", "predicted_answer"])
        .apply(lambda x: x["this_correct_page_searched"] and x["predicted_answer"] == "None")
        .alias("false_negative")
    ).with_columns(
        # did we incorrectly predict that there was an answer?
        pl.struct(["this_correct_page_searched", "predicted_answer"])
        .apply(lambda x: not x["this_correct_page_searched"] and x["predicted_answer"] != "None")
        .alias("false_positive")
    )
    predicted_positive = pr_answers_df["predicted_positive"].sum()
    true_predicted_positive = pr_answers_df["true_predicted_positive"].sum()
    positive = pr_answers_df["positive"].sum()

    false_positive = pr_answers_df["false_positive"].sum()
    false_negative = pr_answers_df["false_negative"].sum()

    precision = true_predicted_positive / predicted_positive
    recall = true_predicted_positive / positive
    #print(calculate_f1(true_predicted_positive, false_positive, false_negative))
    #print(precision, recall)


    # 5. answer accuracy | correct page
    correct_page_df = answers_df.filter(pl.col("this_correct_page_searched"))
    answer_accuracy_given_correct_page = correct_page_df["correct_answer"].sum() / len(correct_page_df)

    num_rows = len(search_results_df)
    num_rows_with_answers = page_search_exists

    eval_metrics = {
        "num_results": num_rows,
        "num_row_processed": num_rows,
        "num_row_input": num_rows_with_answers,
        "num_correct_page_searched": page_search_correct,
        "num_correct_answer": answer_correct,
        "row_processed": num_rows,
        "page_search_recall": page_search_recall,
        # This is the answer accuracy conditional on the correct page having
        # been looked up by search
        "conditional_answer_accuracy": answer_accuracy_given_correct_page,
        "answer_accuracy": answer_correct / num_rows,
        "answer_page_accuracy": answer_page_correct / num_rows_with_answers,
        "answer_false_positive": false_positive,
        "answer_false_negative": false_negative,
        "answer_precision": precision,
        "answer_recall": recall,
    }
    return eval_metrics, pr_answers_df


async def evaluate_term(
    #def evaluate_term(
    term: str,
    gt: pl.DataFrame,
    progress: Progress,
    search_method: SearchMethod,
    extraction_method: ExtractionMethod,
    k: int,
    tournament_k: int,
    districts,
):
    eval_task = progress.add_task(f"Evaluating {term}", total=len(gt))

    # Generate results for the given term in parallel, showing progress along
    # the way.
    results = []
    for row in gt.iter_rows(named=True):
        town = row["town"]
        district = District(full_name=row["district"], short_name=row["district_abb"])
        progress.update(
            eval_task, description=f"Evaluating {term}, {town}, {district.full_name}"
        )
        async for result in compute_eval_result(
            town, district, districts[town], term, row, search_method, extraction_method, k, tournament_k,
        ):
            results.append(result)
        progress.advance(eval_task)
    progress.update(eval_task, description=f"Evaluated {term}")

    results_df = pl.from_dicts(results, schema_overrides={"true_answer_extended": pl.Utf8})
    eval_metrics, results_df = get_metrics(results_df)
    return eval_metrics, results_df


def normalize_town(x):
    x = x.lower().strip()
    x = re.sub(r"\s*-\s*", "-", x)
    x = re.sub(r"\s*/\s*", "-", x)
    x = re.sub(r"\s+", "-", x)
    return x


def extract_districts():
    data = pd.read_excel(DATA_ROOT / "ct-data.xlsx")
    map = {normalize_town(jurisdiction): [] for jurisdiction in set(data["Jurisdiction"])}
    for i, row in data.iterrows():
        town = normalize_town(row["Jurisdiction"])
        district = District(
            full_name=row["Full District Name"],
            short_name=row["AbbreviatedDistrict"],
        )
        map[town].append(district)
    return map


async def main(
        #def main(
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
            metrics[term], new_results_df = await evaluate_term(
                #metrics[term], new_results_df = evaluate_term(
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
