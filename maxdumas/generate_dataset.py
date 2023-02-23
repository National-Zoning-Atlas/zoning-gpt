import glob
import json
from os import makedirs
from os.path import basename, dirname, join, realpath, splitext

import pandas as pd
import pyarrow as pa
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

DIR = dirname(realpath(__file__))

with open(join(DIR, "..", "params.yaml")) as f:
    config = yaml.safe_load(f)[splitext(basename(__file__))[0]]

# Whether or not to publish the resulting dataset to HuggingFace Hub. If False,
# the dataset will be saved locally to disk instead.
PUBLISH_DATASET = True

RANDOM_STATE = config["seed"]
TEST_SPLIT_FRAC = config["test_split_frac"]

DATA_ROOT = realpath(join(DIR, "..", "data"))

# Path to the directory where Textract results from CT Zoning codes live. A
# folder named "processed-data" should exist at this path that contains the
# Textract results.
input_textract_dataset_path = join(DATA_ROOT, "textract_dataset")

# JSON file listing the full set of towns for which we expect to have data.
input_town_list_path = join(DATA_ROOT, "names_all_towns.json")

output_parquet_dataset_path = join(DATA_ROOT, "parquet_dataset")

output_hf_dataset_name = "xyzNLP/nza-ct-zoning-codes"

output_hf_dataset_path = join(DATA_ROOT, "hf_dataset")

# Create output directories if they don't already exist
makedirs(output_parquet_dataset_path, exist_ok=True)
makedirs(output_hf_dataset_path, exist_ok=True)

SCHEMA = pa.schema(
    [
        pa.field("Town", pa.string()),
        pa.field("BlockType", pa.string()),
        pa.field("ColumnIndex", pa.int32(), nullable=True),
        pa.field("ColumnSpan", pa.int32(), nullable=True),
        pa.field("Confidence", pa.float64(), nullable=True),
        # pa.field("EntityTypes", pa.list_(pa.string()), nullable=True),
        pa.field("Id", pa.string()),
        pa.field("Page", pa.int32()),
        pa.field(
            "Relationships",
            pa.list_(
                pa.struct(
                    [
                        pa.field("Ids", pa.list_(pa.string())),
                        pa.field("Type", pa.string()),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field("RowIndex", pa.int32(), nullable=True),
        pa.field("RowSpan", pa.int32(), nullable=True),
        pa.field("Text", pa.string(), nullable=True),
        pa.field("TextType", pa.string(), nullable=True),
        pa.field(
            "Geometry",
            pa.struct(
                [
                    pa.field(
                        "BoundingBox",
                        pa.struct(
                            [
                                pa.field("Width", pa.float64()),
                                pa.field("Height", pa.float64()),
                                pa.field("Left", pa.float64()),
                                pa.field("Top", pa.float64()),
                            ]
                        ),
                    ),
                    pa.field(
                        "Polygon",
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("X", pa.float64()),
                                    pa.field("Y", pa.float64()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        ),
    ]
)

def import_town(town):
    """
    Inputs:
        town (string): name of town whose text data to import
    Returns: pandas dataframe of cleaned/combined JSONs for a given document with all Textract information
    """

    filename = join(input_textract_dataset_path, f"{town}-zoning-code.json")

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame([b for d in data for b in d["Blocks"]], columns=SCHEMA.names).drop_duplicates(
        subset="Id"
    )
    df["Town"] = town

    output_path = join(output_parquet_dataset_path, f"{town}.parquet")
    df.to_parquet(output_path, schema=SCHEMA)

    return output_path


if __name__ == "__main__":
    with open(input_town_list_path) as f:
        all_towns = json.load(f)

    datafiles = [
        path for path in process_map(import_town, all_towns) if path is not None
    ]
    train, test = train_test_split(
        datafiles, test_size=TEST_SPLIT_FRAC, random_state=RANDOM_STATE
    )

    dataset = load_dataset("parquet", data_files={"train": train, "test": test})

    dataset.save_to_disk(output_hf_dataset_path)

    if PUBLISH_DATASET:
        dataset.push_to_hub(output_hf_dataset_name, private=True)
