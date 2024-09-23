import pandas as pd
from itertools import combinations

# Load the CSV file
file_path = 'C:/Users/rr010/OneDrive/Desktop/Class/Projects/TruckDelay_Classification/Data/Training_Data/traffic_table.csv'

data = pd.read_csv(file_path)

def check_unique_columns(data):
    unique_columns = []
    for col in data.columns:
        if data[col].is_unique and data[col].notna().all():
            unique_columns.append(col)
    return unique_columns

def check_composite_keys(data, max_columns=3):
    composite_keys = []
    for r in range(2, max_columns+1):
        for cols in combinations(data.columns, r):
            print(cols)
            subset = data[list(cols)].dropna()
            print(subset.duplicated().sum())
            if subset.duplicated().sum() == 0:
                composite_keys.append(cols)
    return composite_keys

def select_best_composite_key(composite_keys, data):
    composite_keys = sorted(composite_keys, key=len)
    
    best_key = None
    max_uniqueness = -1
    for key in composite_keys:
        subset = data[list(key)].dropna()
        uniqueness_ratio = subset.drop_duplicates().shape[0] / data.shape[0]
        
        if uniqueness_ratio > max_uniqueness:
            max_uniqueness = uniqueness_ratio
            best_key = key
    
    return best_key

def find_duplicate_rows(data, max_columns=3):
    duplicate_rows = pd.DataFrame()
    for r in range(2, max_columns+1):
        for cols in combinations(data.columns, r):
            subset = data[list(cols)].dropna()
            duplicates = subset[subset.duplicated(keep=False)]
            if not duplicates.empty:
                duplicate_rows = pd.concat([duplicate_rows, duplicates])
    return duplicate_rows.drop_duplicates()

primary_key_candidates = check_unique_columns(data)

if not primary_key_candidates:
    composite_key_candidates = check_composite_keys(data)
    print(composite_key_candidates)
    best_composite_key = select_best_composite_key(composite_key_candidates, data)
    print(f"Best composite key: {best_composite_key if best_composite_key else 'None found'}")

print("Potential primary key:", primary_key_candidates if primary_key_candidates else "None found")


