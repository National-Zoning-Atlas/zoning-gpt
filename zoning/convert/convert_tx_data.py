import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from zoning import get_project_root
# Define the root data directory
DATA_ROOT = Path(get_project_root()) / "data"

# Paths to directories
gt_path = DATA_ROOT / "texas-gt"
annotations_path = DATA_ROOT / "texas-annotations"

# List files in each directory
gt_files = [f for f in os.listdir(gt_path) if f.endswith('.csv')]
annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.csv')]

# Define the columns you need to load from GT and annotations
gt_columns = [
    'Jurisdiction', 'Abbreviated District Name', 'Full District Name',
    '1-Family Min. Lot', '1-Family Max. Height', '1-Family Max. Lot Coverage - Buildings',
    '1-Family Min. # Parking Spaces', '1-Family Floor to Area Ratio'
]
annotation_columns = ['JurisdictionName', 'DistrictAbbrv', 'Pagenumber']

# Prepare an empty list to store data dictionaries
data_list = []

# Process each GT file
for gt_file in gt_files:
    gt_df = pd.read_csv(gt_path / gt_file, usecols=gt_columns)
    town_name = gt_file.split('-')[0]

    # Find the corresponding annotation file
    annotation_file = next((f for f in annotation_files if town_name.lower().startswith(f.split(".")[0].lower())), None)
    if annotation_file:
        ann_df = pd.read_csv(annotations_path / annotation_file, usecols=annotation_columns)

        # Merge data based on matching 'Jurisdiction' and 'DistrictAbbrv'
        merged_df = pd.merge(gt_df, ann_df, left_on=['Jurisdiction', 'Abbreviated District Name'],
                             right_on=['JurisdictionName', 'DistrictAbbrv'])

        # Append each row to data_list
        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc=f"Processing {gt_file} and {annotation_file}"):
            data_list.append({
                'town': town_name,
                'district_abb': row['Abbreviated District Name'],
                'district': row['Full District Name'],
                'district_page': row['Pagenumber'],
                'min_lot_size_gt_orig': row['1-Family Min. Lot'],
                'min_lot_size_gt': row['1-Family Min. Lot'],
                'min_lot_size_page_gt': row['Pagenumber'],
                'min_unit_size_gt_orig': None,  # Example placeholder; specify actual source
                'min_unit_size_gt': None,  # Example placeholder; specify actual source
                'min_unit_size_page_gt': None,  # Example placeholder; specify actual source
                'max_height_gt_orig': row['1-Family Max. Height'],
                'max_height_gt': row['1-Family Max. Height'],
                'max_height_page_gt': row['Pagenumber'],
                'max_lot_coverage_gt_orig': row['1-Family Max. Lot Coverage - Buildings'],
                'max_lot_coverage_gt': row['1-Family Max. Lot Coverage - Buildings'],
                'max_lot_coverage_page_gt': row['Pagenumber'],
                'max_lot_coverage_pavement_gt_orig': None,  # Example placeholder; specify actual source
                'max_lot_coverage_pavement_gt': None,  # Example placeholder; specify actual source
                'max_lot_coverage_pavement_page_gt': None,  # Example placeholder; specify actual source
                'min_parking_spaces_gt_orig': row['1-Family Min. # Parking Spaces'],
                'min_parking_spaces_gt': row['1-Family Min. # Parking Spaces'],
                'min_parking_spaces_page_gt': row['Pagenumber'],
                'floor_to_area_ratio_gt_orig': row['1-Family Floor to Area Ratio'],
                'floor_to_area_ratio_gt': row['1-Family Floor to Area Ratio'],
                'floor_to_area_ratio_page_gt': row['Pagenumber'],
                'review': None,  # Placeholder; specify any processing if necessary
                'notes': None  # Placeholder; specify any processing if necessary
            })

# Create a DataFrame from data_list
merged_data = pd.DataFrame(data_list)

# Save the merged data to a new CSV file
merged_data.to_csv(DATA_ROOT / 'merged_zoning_data.csv', index=False)