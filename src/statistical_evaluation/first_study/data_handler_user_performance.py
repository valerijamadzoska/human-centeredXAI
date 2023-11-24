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
file_path = '/Users/valerijamadzoska/human-centeredXAI/data/first_study/data_human-centeredXAI_2023-10-31_15-46.csv'
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
output_file_path = '/Users/valerijamadzoska/human-centeredXAI/data/first_study/userPerformance.csv'
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