# zoning

General Guide / Instructions

Adding new terms:
- STEP 1: Edit/review thesaurus
    - Ensure that the new term is in the thesaurus (ex. “min unit size”)
    - Ensure that relevant identifier terms like “max” or “min” are included in thesaurus (ex. “min” for “min unit size”)
    - Add expected dimensions (and its variations) for the term to thesaurus (ex. ["square feet", "sq ft", "sf", "s.f.", "sq. ft.", "SF", "sq. ft", "sqft", "sq.ft.”] for “min unit size”)
- STEP 2: Ensure that the two columns corresponding to the term are in the ground_truth.csv file
    - `f”{term}_gt”`: This is the ground truth answer
    - `f”{term}_page_gt”`: This is the ground truth page where you would expect to find the answer
    - Note that evaluation metrics will only be produced if this data is present. The model will run and produce results anyway but will not produce evaluation metrics if the ground truth columns are missing. 
- STEP 3: In order to run the model for a term, add the term as string to terms list in [eval.py](http://eval.py/) (ex. “min unit size”)
    `terms = ["min lot size", "min unit size"] # update to list of terms you want to run`
- STEP 4: Edit/review prompt
    - We are using a few-shot prompting approach by providing the model with various examples of formats in which it may find the answer. Though the generalized prompt should work for all numerical terms, you may edit the prompt or add additional examples to it if you believe that certain formats are missing.
    - The suggested path is to start with the default prompt and edit it while debugging if you find multiple instances of a format that is not captured by the prompt. It is advised to not add very extreme cases as examples in the prompt because if it is just an outlier, it will not affect model performance significantly.
- STEP 5: Run using command `python -m zoning.data_processing.eval` from root directory
- STEP 6: View metrics in `eval.yaml` and results in `eval.csv` in the "data/results" folder
