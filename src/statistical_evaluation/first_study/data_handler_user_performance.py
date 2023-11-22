import pandas as pd

def calculate_row_metrics(row, columns_tp, columns_tn, columns_fp, columns_fn):
    TP = row[columns_tp].eq(2).sum()
    TN = row[columns_tn].eq(2).sum()
    FP = row[columns_fp].eq(1).sum() 
    FN = row[columns_fn].eq(1).sum()

    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else None
    Sensitivity = TP / (TP + FN) if (TP + FN) != 0 else None
    Specificity = TN / (TN + FP) if (TN + FP) != 0 else None

    return pd.Series({
        'Accuracy': Accuracy,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity
    })

# Load the data
file_path = 'data\data_human-centeredXAI_2023-10-31_15-46.csv'
df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

# Clean the data
df_cleaned = df.drop(0).reset_index(drop=True)
df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore')

column_groups = {
    'heatmap': {
        'TP': ['F101_01', 'F201_01', 'F301_01', 'F401_01'],
        'TN': ['F102_02', 'F202_02', 'F302_02', 'F402_02'],
        'FN': ['F101_01', 'F201_01', 'F301_01', 'F401_01'],
        'FP': ['F102_02', 'F202_02', 'F302_02', 'F402_02']
        
    },
    'cluster': {
        'TP': ['F105_01', 'F205_01', 'F305_01', 'F405_01'],
        'TN': ['F106_02', 'F206_02', 'F306_02', 'F406_02'],
        'FN': ['F105_01', 'F205_01', 'F305_01', 'F405_01'],
        'FP': ['F106_02', 'F206_02', 'F306_02', 'F406_02']
    },
        'cluster_overlay': {
        'TP': ['F103_01', 'F203_01', 'F303_01', 'F403_01'],
        'TN': ['F104_02', 'F204_02', 'F304_02', 'F404_02'],
        'FN': ['F103_01', 'F203_01', 'F303_01', 'F403_01'],
        'FP': ['F104_02', 'F204_02', 'F304_02', 'F404_02']
    },
        'contour': {
        'TP': ['F107_01', 'F207_01', 'F307_01', 'F407_01'],
        'TN': ['F108_02', 'F208_02', 'F308_02', 'F408_02'],
        'FN': ['F107_01', 'F207_01', 'F307_01', 'F407_01'],
        'FP': ['F108_02', 'F208_02', 'F308_02', 'F408_02']
    }
}

# Calculate metrics, store the results 
metrics_df = pd.DataFrame()
for metric_type, columns in column_groups.items():
    metrics_df[[f'{metric_type}_Accuracy', f'{metric_type}_Sensitivity', f'{metric_type}_Specificity']] = df_cleaned.apply(
        lambda row: calculate_row_metrics(row, columns['TP'], columns['TN'], columns['FP'], columns['FN']), axis=1)

# Save results
output_file_path = r'C:\Users\icke\Desktop\human-centeredXAI-1\data\userPerformance.csv'
metrics_df.to_csv(output_file_path, index=False)
