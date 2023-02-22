import glob
import json
from os import makedirs
from os.path import basename, dirname, join, realpath, splitext

import pandas as pd
import pyarrow as pa
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

DIR = dirname(realpath(__file__))

with open(join(DIR, "..", "params.yaml")) as f:
    config = yaml.safe_load(f)[splitext(basename(__file__))[0]]

# Whether or not to publish the resulting dataset to HuggingFace Hub. If False,
# the dataset will be saved locally to disk instead.
PUBLISH_DATASET = False

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
        pa.field("EntityTypes", pa.list_(pa.string()), nullable=True),
        pa.field("Hint", pa.string(), nullable=True),
        pa.field("Id", pa.string()),
        pa.field("Page", pa.int32()),
        pa.field("Query", pa.null()),
        # pa.field(
        #     "Relationships",
        #     pa.struct(
        #         [
        #             pa.field("Ids", pa.list_(pa.string())),
        #             pa.field("Type", pa.string()),
        #         ]
        #     ),
        #     nullable=True,
        # ),
        pa.field("RowIndex", pa.int32(), nullable=True),
        pa.field("RowSpan", pa.int32(), nullable=True),
        pa.field("SelectionStatus", pa.string(), nullable=True),
        pa.field("Text", pa.string(), nullable=True),
        pa.field("TextType", pa.string(), nullable=True),
        pa.field("BB_Height", pa.float64()),
        pa.field("BB_Left", pa.float64()),
        pa.field("BB_Top", pa.float64()),
        pa.field("BB_Width", pa.float64()),
        pa.field("Poly_X0", pa.float64()),
        pa.field("Poly_Y0", pa.float64()),
        pa.field("Poly_X1", pa.float64()),
        pa.field("Poly_Y1", pa.float64()),
        pa.field("Poly_X2", pa.float64()),
        pa.field("Poly_Y2", pa.float64()),
        pa.field("Poly_X3", pa.float64()),
        pa.field("Poly_Y3", pa.float64()),
        pa.field("relat_children", pa.list_(pa.string()), nullable=True),
        pa.field("relat_values", pa.list_(pa.string()), nullable=True),
    ]
)


def extract_bbox(df):
    """Extracts Bounding Box columns from nested "Geometry" JSON"""
    geom_list = df["Geometry"].to_list()
    bbox = [geom["BoundingBox"] for geom in geom_list]
    bbox_df = pd.DataFrame(bbox).add_prefix("BB_")
    return bbox_df


def extract_polygon(df):
    """Extracts Polygon columns from nested "Geometry" JSON"""
    geom_list = df["Geometry"].to_list()
    poly_df = pd.DataFrame()
    for point in range(4):
        poly = (
            pd.DataFrame([geom["Polygon"][point] for geom in geom_list])
            .add_prefix("Poly_")
            .add_suffix("{}".format(point))
        )
        poly_df = pd.concat([poly_df, poly], axis=1)
    return poly_df


def extract_relationship(relationship, type="CHILD"):
    """
    Extracts relationship types from nested "Relationships" JSON;
    Can specify either 'CHILD' or 'VALUE' as the type.
    """
    if relationship is None or relationship[0]["Type"] != type:
        return None
    else:
        return relationship[0]["Ids"]


def import_town(town, isMap=False):
    """
    Inputs:
        town (string): name of town whose text data to import
        isMap (boolean): whether to import the town's map (True) or code (False)
    Returns: pandas dataframe of cleaned/combined JSONs for a given document with all Textract information
    """

    local_folder = (
        join(input_textract_dataset_path, f"{town}-map")
        if isMap
        else join(input_textract_dataset_path, town)
    )

    dict_list = []

    # Note that for some folders, the JSON files are invalid and sibling files
    # with no extension are the ones that contain the relevant data. In some
    # folders, it is the JSON files that are valid. So we just load every file
    # and try to parse it, and keep what we can, removing duplicates.
    for file in tqdm(
        sorted(
            glob.glob("*", root_dir=local_folder),
            key=lambda f: int(splitext(f)[0]),
        ),
        desc=town,
    ):
        try:
            with open(join(local_folder, file), encoding="utf-8") as f:
                data = json.load(f)["Blocks"]
        except:
            tqdm.write(
                f"Error encountered while loading JSON file {join(local_folder, file)}."
            )
            continue
        dict_list.extend(data)
    df = pd.DataFrame(dict_list).drop_duplicates(subset="Id")

    if len(df) == 0:
        tqdm.write("No doc exists in the bucket!")
        return None

    bbox = extract_bbox(df)
    poly = extract_polygon(df)

    children = (
        df["Relationships"]
        .apply(extract_relationship, type="CHILD")
        .rename("relat_children")
    )
    values = (
        df["Relationships"]
        .apply(extract_relationship, type="VALUE")
        .rename("relat_values")
    )

    clean_df = pd.concat([df, bbox, poly, children, values], axis=1).drop(
        "Geometry", axis=1
    ).drop(columns=["Relationships"])
    clean_df["Town"] = town

    output_path = join(output_parquet_dataset_path, f"{town}.parquet")
    clean_df.to_parquet(output_path, schema=SCHEMA)

    return output_path


if __name__ == "__main__":
    with open(input_town_list_path) as f:
        all_towns = json.load(f)

    datafiles = [path for path in process_map(import_town, all_towns) if path is not None]
    train, test = train_test_split(datafiles, test_size=TEST_SPLIT_FRAC, random_state=RANDOM_STATE)

    dataset = load_dataset("parquet", data_files={"train": train, "test": test})

    dataset.save_to_disk(output_hf_dataset_path)

    if PUBLISH_DATASET:
        dataset.push_to_hub(output_hf_dataset_name, private=True)
