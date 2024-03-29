import sys
import pandas as pd
import huggingface_hub
from datasets import load_dataset, DatasetDict, Dataset

# login to huggingface using API key
HF_API_KEY_READ = sys.argv[2]
HF_API_KEY_WRITE = sys.argv[3]

# whether to write to HF or not
WRITE = sys.argv[1]

def load_dataset(url="xyzNLP/nza-ct-zoning-codes"):
    dataset = load_dataset(url)
    return dataset

def process_dataset(dataset):
    # create dfs from dataset
    dataset.set_format("pandas")
    train_df = dataset['train'][:]
    test_df = dataset['test'][:]

    # replace null values
    train_df['Text'].fillna("",inplace=True)
    train_df['TextType'].fillna("0",inplace=True)
    test_df['Text'].fillna("",inplace=True)
    test_df['TextType'].fillna("0",inplace=True)

    train_chunked_df = train_df.groupby(by=['Town', 'Page'], as_index=False).agg({'Text': ' '.join, 'TextType': ' '.join})
    test_chunked_df = test_df.groupby(by=['Town', 'Page'], as_index=False).agg({'Text': ' '.join, 'TextType': ' '.join})

    return train_chunked_df, test_chunked_df


def get_district_names(train_chunked_df, test_chunked_df, metadata_url="xyzNLP/nza-ct-zoning-atlas-metadata"):
    dataset_metadata = load_dataset(metadata_url)
    dataset_metadata.set_format("pandas")
    df_metadata = dataset_metadata['train'][:]

    train_towns = train_chunked_df['Town'].unique()
    test_towns = test_chunked_df['Town'].unique()

    df_metadata['Jurisdiction'] = df_metadata['Jurisdiction'].apply(lambda x: str(x).lower().rstrip())
    df_metadata['Jurisdiction'] = df_metadata['Jurisdiction'].apply(lambda x: str(x).replace(' - ', '-'))
    df_metadata['Jurisdiction'] = df_metadata['Jurisdiction'].apply(lambda x: str(x).replace('/', '-'))
    df_metadata['Jurisdiction'] = df_metadata['Jurisdiction'].apply(lambda x: str(x).replace(' ', '-'))

    train_districts_df = df_metadata[df_metadata['Jurisdiction'].isin(train_towns)][['Jurisdiction', 'AbbreviatedDistrict', 'Full District Name']]
    test_districts_df = df_metadata[df_metadata['Jurisdiction'].isin(test_towns)][['Jurisdiction', 'AbbreviatedDistrict', 'Full District Name']]

    train_districts_dict = (train_districts_df.groupby(by='Jurisdiction')['Full District Name'].apply(list)).to_dict()
    test_districts_dict = (test_districts_df.groupby(by='Jurisdiction')['Full District Name'].apply(list)).to_dict()

    return train_districts_dict, test_districts_dict


# TODO: Create better system where these are automatically generated given the column names + column type (numerical vs categorical)
gpt3_queries_template = [
                        'Categorize xyz district as "Primarily Residential", "Nonresidential", or "Mixed with Residential" based on the information below. Note that a "Primarily Residential" designation should be given to districts allowing: housing only; housing and assorted uses customarily allowed in residential areas, including religious institutions and schools; or housing and agricultural uses. Output “NA” if information is not provided.',
                        'Categorize 1-family housing as "Allowed/Conditional", "Public Hearing", "Prohibited", or "Overlay" for the Residential-Agricultural  district based on the information below. Output just one of the four categories.',
                        'Categorize 2-family housing as "Allowed/Conditional", "Public Hearing", "Prohibited", or "Overlay" for the Residential-Agricultural  district based on the information below. Output just one of the four categories.',
                        'Categorize 3-family housing as "Allowed/Conditional", "Public Hearing", "Prohibited", or "Overlay" for the Residential-Agricultural  district based on the information below. Output just one of the four categories.',
                        'Categorize 4+-family housing as "Allowed/Conditional", "Public Hearing", "Prohibited", or "Overlay" for the Residential-Agricultural  district based on the information below. Output just one of the four categories.',
                        'Categorize Accessory Dwelling Units as "Allowed/Conditional", "Public Hearing", "Prohibited", or "Overlay" for the Residential-Agricultural  district based on the information below. Output just one of the four categories.',
                        'What is the minimum lot size for 1-family homes? Output the number in acres as a numerical answer or "NA" if it is not mentioned.',
                        'What is the minimum lot size for 2-family homes? Output the number in acres as a numerical answer or "NA" if it is not mentioned.',
                        'What is the minimum lot size for 3-family homes? Output the number in acres as a numerical answer or "NA" if it is not mentioned.',
                        'What is the minimum lot size for 4+-family homes? Output the number in acres as a numerical answer or "NA" if it is not mentioned.',
                        'What is the maximum lot coverage for buildings for 1-family homes as a percentage? Just output the numerical answer. Just output the numerical answer. Say "NA" if it is not mentioned.',
                        'What is the maximum lot coverage for buildings and impervious surface for 1-family homes as a percentage? Just output the numerical answer. Say "NA" if it is not mentioned.',
                        'What is the front setback requirement for 1-family housing in feet? Give the numerical output only.',
                        'What is the side setback requirement for 1-family housing in feet? Give the numerical output only.',
                        'What is the rear setback requirement for 1-family housing in feet? Give the numerical output only.',
                        'What is the maximum height allowed for 1-family homes in feet? Output the numerical answer only.',
                        'What is the minimum unit size or minimum floor area for 1-family housing (in sqaure feet)? Output the numerical answer only.'
                        ]


def create_queries(districts_dict, queries_template):
    all_queries = {}
    for jur in districts_dict:
        queries = []
    for district in districts_dict[jur]:
      for query in queries_template:
        custom_query = query.replace('xyz', district)
        queries.append(custom_query)
    all_queries[jur] = queries
  
    return all_queries


if __name__ == "__main__":
    huggingface_hub.login(token=HF_API_KEY_READ)
    huggingface_hub.login(token=HF_API_KEY_WRITE)

    dataset_raw = load_dataset(url="xyzNLP/nza-ct-zoning-codes")

    train_chunked_df, test_chunked_df = process_dataset(dataset_raw)

    train_districts_dict, test_districts_dict = get_district_names(train_chunked_df, test_chunked_df, metadata_url="xyzNLP/nza-ct-zoning-atlas-metadata")

    train_districts_dataset = Dataset.from_pandas(pd.DataFrame(data=train_districts_dict))
    test_districts_dataset =  Dataset.from_pandas(pd.DataFrame(data=test_districts_dict))

    dataset = DatasetDict({"train": train_districts_dataset, "test": test_districts_dataset})

    dataset.push_to_hub("xyzNLP/nza-ct-zoning-codes-text-processed", private=True)