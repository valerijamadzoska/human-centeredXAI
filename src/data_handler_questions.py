import pandas as pd
import re

def calculate_metrics(file_path, column_prefix='F'):    
    # Load the data from the CSV file with specified encoding and separator
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)
    
    # Ensure that the data is numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Filtering columns that match the pattern with a prefix and a three-digit number
    columns_of_interest = [col for col in df.columns if re.match(f'{column_prefix}\d{{3}}$', col)]
    
    # Separating odd and even columns
    odd_columns = [col for col in columns_of_interest if int(col[len(column_prefix):]) % 2 != 0]
    even_columns = [col for col in columns_of_interest if int(col[len(column_prefix):]) % 2 == 0]
    
    # Calculating TP, TN, FN, FP
    TP = sum((df[col] == 1).sum() for col in odd_columns) + sum((df[col] == 2).sum() for col in even_columns)
    TN = sum((df[col] == 1).sum() for col in even_columns) + sum((df[col] == 2).sum() for col in odd_columns)
    FN = sum((df[col] == 2).sum() for col in odd_columns)
    FP = sum((df[col] == 2).sum() for col in even_columns)
    
    return TP, TN, FN, FP

def calculate_heatmap(file_path):

    return TP, TN, FN, FP

def calculate_cluster(file_path):

    return TP, TN, FN, FP

def calculate_cluster_overlay(file_path):

    return TP, TN, FN, FP

def calculate_contour(file_path):

    return TP, TN, FN, FP

# Method call
file_path = 'data\data_human-centeredXAI_2023-10-18_22-34.csv'
result = calculate_metrics(file_path)
print(result)

