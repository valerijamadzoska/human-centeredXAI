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

# mean metrics in dependency on age/ AI knowledge/XAI knowledge/Vision
def calculate_average_metrics_age_group(df, df_metrics, age_column):
    #Alter: 1= unter 30, 2=30-50, 3=51-60, 4=über 61
    #AI Knowledge: 1=keine, 2= etwas gehört, 3= Tools angewandt, 4=programmiert
    #XAI Knowledge: 1=keine, 2= etwas gehört, 3= Tools angewandt, 4=programmiert
    #Farbensehen gestört=1, schlechte Sehkraft= 2, keine Beeinträchtigung=3, sonstige=4
    df_group = df[df[age_column] == 3]
    average_metrics = df_metrics.loc[df_group.index].mean()
    return average_metrics

# Load the data
file_path = '//Users/valerijamadzoska/human-centeredXAI/data/second_study/data_human-centered_XAI_2023-12-01_01-29.csv'
df = pd.read_csv(file_path, encoding='utf-8', sep=';', header=0)

# Clean the data
df_cleaned = df.drop(0).reset_index(drop=True)
df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore')

column_groups = {
'heatmap': {
    'TP': ['C101_01', 'C201_01', 'C301_01', 'C401_01'],
    'TN': ['C102_02', 'C202_02', 'C302_02', 'C402_02'],
    'FN': ['C101_01', 'C201_01', 'C301_01', 'C401_01'],
    'FP': ['C102_02', 'C202_02', 'C302_02', 'C402_02']
},
'cluster': {
    'TP': ['C105_01', 'C205_01', 'C305_01', 'C405_01'],
    'TN': ['C106_02', 'C206_02', 'C306_02', 'C406_02'],
    'FN': ['C105_01', 'C205_01', 'C305_01', 'C405_01'],
    'FP': ['C106_02', 'C206_02', 'C306_02', 'C406_02']
},
'cluster_overlay': {
    'TP': ['C103_01', 'C203_01', 'C303_01', 'C403_01'],
    'TN': ['C104_02', 'C204_02', 'C304_02', 'C404_02'],
    'FN': ['C103_01', 'C203_01', 'C303_01', 'C403_01'],
    'FP': ['C104_02', 'C204_02', 'C304_02', 'C404_02']
},
'contour': {
    'TP': ['C107_01', 'C207_01', 'C307_01', 'C407_01'],
    'TN': ['C108_02', 'C208_02', 'C308_02', 'C408_02'],
    'FN': ['C107_01', 'C207_01', 'C307_01', 'C407_01'],
    'FP': ['C108_02', 'C208_02', 'C308_02', 'C408_02']
}
}

# Calculate metrics, store the results 
metrics_df = pd.DataFrame()
for metric_type, columns in column_groups.items():
    metrics_df[[f'{metric_type}_Accuracy', f'{metric_type}_Sensitivity', f'{metric_type}_Specificity']] = df_cleaned.apply(
        lambda row: calculate_row_metrics(row, columns['TP'], columns['TN'], columns['FP'], columns['FN']), axis=1)

# Save results
output_file_path = '/Users/valerijamadzoska/human-centeredXAI/data/second_study/userPerformanceC.csv'
metrics_df.to_csv(output_file_path, index=False)

##########################
#mean metrics depending on age
##########################


#mean metrics for user under 30 / 30-50 / 51-60 / over 61 
average_accuracy_under30 = calculate_average_metrics_age_group(df_cleaned, metrics_df, 'A001')
print("Durchschnittliche Metriken für Benutzer im Alter über 61:")
print(average_accuracy_under30)


##########################
#mean metrics depending on AI knowledge
##########################

#mean metrics for user with AI knowledge: keine/ etwas gehört/ Tools angewandt/ programmiert
average_metrics_ai = calculate_average_metrics_age_group(df_cleaned, metrics_df, 'A002')
print("Durchschnittliche Metriken für Benutzer die KI programmiert haben:")
print(average_metrics_ai)

##########################
#mean metrics depending on XAI knowledge
##########################

#mean metrics for user with XAI knowledge: keine/ etwas gehört/ Tools angewandt/ programmiert
average_metrics_xai = calculate_average_metrics_age_group(df_cleaned, metrics_df, 'A004')
print("Durchschnittliche Metriken für Benutzer die  XAI programmiert haben:")
print(average_metrics_xai)

##########################
#mean metrics depending on vision
##########################

#mean metrics for user with Farbensehen gestört/schlechte Sehkraft/keine Beeinträchtigung/
average_metrics_vision = calculate_average_metrics_age_group(df_cleaned, metrics_df, 'A005')
print("Durchschnittliche Metriken für Benutzer mit keiner Beeinträchtigung:")
print(average_metrics_vision)