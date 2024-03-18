import csv
import pickle

with open("results.pickle", "rb") as f:
    results = pickle.load(f)

with open('tx_results.csv', 'w', newline='') as file:
    # Extract headers (dict keys) for the CSV
    # This assumes all dictionaries have the same structure
    headers = results[0].keys()
    
    # Create a csv.DictWriter object
    writer = csv.DictWriter(file, fieldnames=headers)
    
    # Write the header row
    writer.writeheader()
    
    # Write the rows using dictionaries
    for a_dict in results:
        writer.writerow(a_dict)
