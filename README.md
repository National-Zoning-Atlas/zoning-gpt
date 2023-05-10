# zoning

### General Guide / Instructions

### Instructions for adding new terms:
- STEP 1: Edit/review thesaurus
    - Ensure that the new term is in the thesaurus (ex. “min unit size”)
    - Ensure that relevant identifier terms like “max” or “min” are included in thesaurus (ex. “min” for “min unit size”)
    - Add expected dimensions (and its variations) for the term to thesaurus (ex. ["square feet", "sq ft", "sf", "s.f.", "sq. ft.", "SF", "sq. ft", "sqft", "sq.ft.”] for “min unit size”)
- STEP 2: Ensure that the two columns corresponding to the term are in the ground_truth.csv file
    - `f”{term}_gt”`: This is the ground truth answer
    - `f”{term}_page_gt”`: This is the ground truth page where you would expect to find the answer
    - Note that evaluation metrics will only be produced if this data is present. The model will run and produce results anyway but will not produce evaluation metrics if the ground truth columns are missing. 
- STEP 3: In order to run the model for a term, add the term as string to terms list in [zoning/data_processing/eval.py](http://eval.py/) (ex. “min unit size”)
    `terms = ["min lot size", "min unit size"] # update to list of terms you want to run`
- STEP 4: Edit/review prompt 
    - This will be under [templates/extraction_chat_completition.pmpt.tpl] for gpt-3.5 and gpt-4 or [templates/extraction_completion.pmpt.tpl] for gpt-3 "text-davinci-003"
    - We are using a few-shot prompting approach by providing the model with various examples of formats in which it may find the answer. Though the generalized prompt should work for all numerical terms, you may edit the prompt or add additional examples to it if you believe that certain formats are missing.
    - The suggested path is to start with the default prompt and edit it while debugging if you find multiple instances of a format that is not captured by the prompt. It is advised to not add very extreme cases as examples in the prompt because if it is just an outlier, it will not affect model performance significantly.
- STEP 5: Run using command `python -m zoning.data_processing.eval` from root directory
- STEP 6: View metrics in `eval.yaml` and results in `eval.csv` in the [data/results] folder

### Debugging and improvement tips: 
- When adding a new term, first establish a baseline by following the steps above.
- Debugging and improving page recall (search):
    - Note that you can only debug search if you have the data for the ground truth page for the term. 
    - The main reason for poor search recall is typically the lack of a fleshed out thesaurus. The best way to debug this is to add more terms to the thesaurus by manually going over the cases where it does not find the correct page and seeing what term is used in that zoning doc.
    - We are currently using a simple elastic search query for search. You may want to experiment with faiss indexing (embeddings based search) as it may work better for cases where a fleshed out thesaurus is not available.
- Debugging and improving answer accuracy (extraction):
    - In order to improve answer accuracy, it is important to understand the types of errors
    - Types of errors:
        - Incorrect answer because correct page not found (this can likely be addressed by improving page recall performance)
        - Incorrect answer despite correct page being found
            - Answer in complex table, model returns NaN/incorrect cell (this is a common error and we have found that even GPT-4 makes such errors when tables are complex and especially when keywords like “Minimum” and “Lot Size” are split up into different cells of the table)
                - This can be addressed either by changing the way tables are represented in text or by providing examples in the prompt that capture such cases
            - Answer is in linked table/appendix that is inaccessible (this is not very common)
            - Text-based answer with not very obvious wording (the answer is in the text but the keywords used are different from what is in the thesaurus)
            - Complex answer with multiple values - the model just returns 1 of the others or something completely incorrect
    - Other things to look out for:
        - Sometimes (very rarely) the answer may be correct (or somewhat correct) but may be marked as wrong by our evaluation pipeline.
            - We have two evaluation workflows:
                - For simple numerical answer, we clean the model output (using regex) and convert it to a numerical value for direct comparison with expected value
                - For complex answers with multiple values, we pass the model output to another prompt that compared the expected and returned answers
            - For the numerical workflow, the evaluation pipeline may sometimes (very rarely) mark a correct answer as incorrect if it is not in the correct format or has an atypical unit. If this happens, you should update the clean_string_units function in zoning/prompting/eval_results.py to include the edge case you found.
            - For the complex answer workflow, our method is not perfect, so you may want to iterate on the prompt to improve evaluation performance.
