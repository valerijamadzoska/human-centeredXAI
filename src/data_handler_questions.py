import pandas as pd
import re

def calculate_heatmap(df: pd.DataFrame) -> dict:
    # Load the data from the CSV file with specified encoding and separator
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F101_01', 'F102_02', 'F201_01', 'F202_02', 'F301_01', 'F302_02', 'F401_01', 'F402_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

    # Calculate the metrics
    TP = df[['F101_01', 'F201_01', 'F301_01', 'F401_01']].eq(2).sum().sum()
    TN = df[['F102_02', 'F202_02', 'F302_02', 'F402_02']].eq(2).sum().sum()
    FN = df[['F101_01', 'F201_01', 'F301_01', 'F401_01']].eq(1).sum().sum()
    FP = df[['F102_02', 'F202_02', 'F302_02', 'F402_02']].eq(1).sum().sum()

    metrics = calculate_classification_metrics(TP, TN, FP, FN)
    print("Heatmap Metrics: ", metrics)

    return {
        'True Positive (TP)': TP,
        'True Negative (TN)': TN,
        'False Negative (FN)': FN,
        'False Positive (FP)': FP
    }

def calculate_cluster(df: pd.DataFrame) -> dict:
    # Load the data from the CSV file with specified encoding and separator
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F105_01', 'F106_02', 'F205_01', 'F206_02', 'F305_01', 'F306_02', 'F405_01', 'F406_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

    # Calculate the metrics
    TP = df[['F105_01', 'F205_01', 'F305_01', 'F405_01']].eq(2).sum().sum()
    TN = df[['F106_02', 'F206_02', 'F306_02', 'F406_02']].eq(2).sum().sum()
    FN = df[['F105_01', 'F205_01', 'F305_01', 'F405_01']].eq(1).sum().sum()
    FP = df[['F106_02', 'F206_02', 'F306_02', 'F406_02']].eq(1).sum().sum()

    metrics = calculate_classification_metrics(TP, TN, FP, FN)
    print("Cluster Metrics: ", metrics)

    return {
        'True Positive (TP)': TP,
        'True Negative (TN)': TN,
        'False Negative (FN)': FN,
        'False Positive (FP)': FP
    }

def calculate_cluster_overlay(df: pd.DataFrame) -> dict:
    # Load the data from the CSV file with specified encoding and separator
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F103_01', 'F104_02', 'F203_01', 'F204_02', 'F303_01', 'F304_02', 'F403_01', 'F404_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

    # Calculate the metrics
    TP = df[['F103_01', 'F203_01', 'F303_01', 'F403_01']].eq(2).sum().sum()
    TN = df[['F104_02', 'F204_02', 'F304_02', 'F404_02']].eq(2).sum().sum()
    FN = df[['F103_01', 'F203_01', 'F303_01', 'F403_01']].eq(1).sum().sum()
    FP = df[['F104_02', 'F204_02', 'F304_02', 'F404_02']].eq(1).sum().sum()

    metrics = calculate_classification_metrics(TP, TN, FP, FN)
    print("Cluster Overlay Metrics: ", metrics)

    return {
        'True Positive (TP)': TP,
        'True Negative (TN)': TN,
        'False Negative (FN)': FN,
        'False Positive (FP)': FP
    }

def calculate_contour(df: pd.DataFrame) -> dict:
    # Load the data from the CSV file with specified encoding and separator
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F107_01', 'F108_02', 'F207_01', 'F208_02', 'F307_01', 'F308_02', 'F407_01', 'F408_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

    # Calculate the metrics
    TP = df[['F107_01', 'F207_01', 'F307_01', 'F407_01']].eq(2).sum().sum()
    TN = df[['F108_02', 'F208_02', 'F308_02', 'F408_02']].eq(2).sum().sum()
    FN = df[['F107_01', 'F207_01', 'F307_01', 'F407_01']].eq(1).sum().sum()
    FP = df[['F108_02', 'F208_02', 'F308_02', 'F408_02']].eq(1).sum().sum()

    metrics = calculate_classification_metrics(TP, TN, FP, FN)
    print("Contour Metrics: ", metrics)

    return {
        'True Positive (TP)': TP,
        'True Negative (TN)': TN,
        'False Negative (FN)': FN,
        'False Positive (FP)': FP
    }

def calculate_classification_metrics(TP: int, TN: int, FP: int, FN: int) -> dict:
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else None
    Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else None
    Specificity = TN / (TN + FP) if (TN + FP) != 0 else None
    
    return {
        'Accuracy': Accuracy,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity
    }




# Method call
file_path = 'data\data_human-centeredXAI_2023-10-18_22-34.csv'
result_heatmap = calculate_heatmap(file_path)
print( "Heatmap: ", result_heatmap)

result_cluster = calculate_cluster(file_path)
print("Cluster: ", result_cluster)

result_cluster_overlay = calculate_cluster_overlay(file_path)
print("Cluster Overlay: ", result_cluster_overlay)

result_contour = calculate_contour(file_path)
print("Contour: ", result_contour)

