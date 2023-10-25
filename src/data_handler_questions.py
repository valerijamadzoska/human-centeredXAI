import pandas as pd
import re
from scipy.stats import chi2_contingency

def calculate_heatmap(df: pd.DataFrame) -> dict:
    # Load the data from the CSV 
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
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F105_01', 'F106_02', 'F205_01', 'F206_02', 'F305_01', 'F306_02', 'F405_01', 'F406_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

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
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F103_01', 'F104_02', 'F203_01', 'F204_02', 'F303_01', 'F304_02', 'F403_01', 'F404_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

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
    df = pd.read_csv(file_path, encoding='utf-16', sep='\t', header=0)

    columns_of_interest = ['F107_01', 'F108_02', 'F207_01', 'F208_02', 'F307_01', 'F308_02', 'F407_01', 'F408_02']
    df[columns_of_interest] = df[columns_of_interest].apply(pd.to_numeric, errors='coerce')

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
#######################################
#chi-square-Test
#######################################
def chi_square_test(TP: int, TN: int, FP: int, FN: int) -> None:
    # Creating a contingency table
    contingency_table = [[TP, FP], [FN, TN]]
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi-Square value: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of freedom: {dof}")
    print(f"Expected frequencies table: \n{expected}")
    
    if p < 0.05:
        print("The differences between the categories are statistically significant.")
    else:
        print("The differences between the categories are not statistically significant.")

#######################################
#Calculate Metrics for each Image
#######################################
def calculate_metrics_all_img(dataframe, start_column=1):
    dataframe = pd.read_csv(dataframe, encoding='utf-16', sep='\t', header=0)
    columns = dataframe.columns[start_column:]
    
    # Initialize dictionary for results
    metrics = {}
    for column in columns:
        # Ensure the column name has the expected format
        if '_' in column:
            # Extract object name and type (01 or 02) from the column name
            obj, obj_type = column.split('_')
            
            # Only columns with obj_type 01 or 02
            if obj_type in ['01', '02']:
                
                # Check and create dictionary for each unique object
                if obj not in metrics:
                    metrics[obj] = {'True Positive': 0, 'True Negative': 0, 'False Negative': 0, 'False Positive': 0}
                
                # Calculate metrics 
                if obj_type == '01':
                    metrics[obj]['True Positive'] += (dataframe[column] == 1).sum()
                    metrics[obj]['False Negative'] += (dataframe[column] == 2).sum()
                elif obj_type == '02':
                    metrics[obj]['True Negative'] += (dataframe[column] == 2).sum()
                    metrics[obj]['False Positive'] += (dataframe[column] == 1).sum()

    # Calculate emtrics
    for obj in metrics.keys():
        TP, TN, FP, FN = metrics[obj].values()
        metrics[obj]['Accuracy'] = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else None
        metrics[obj]['Sensitivity'] = TP / (TP + FN) if (TP + FN) != 0 else None
        metrics[obj]['Specificity'] = TN / (TN + FP) if (TN + FP) != 0 else None
         
    return metrics


# Method call
file_path = 'data\data_human-centeredXAI_2023-10-25_09-51.csv'
result_heatmap = calculate_heatmap(file_path)
print( "Heatmap: ", result_heatmap)

result_cluster = calculate_cluster(file_path)
print("Cluster: ", result_cluster)

result_cluster_overlay = calculate_cluster_overlay(file_path)
print("Cluster Overlay: ", result_cluster_overlay)

result_contour = calculate_contour(file_path)
print("Contour: ", result_contour)

metricsForEachImg = calculate_metrics_all_img(file_path)
print("Img: ", metricsForEachImg)

print("Chi-square Test for HeatMap:")
chi_square_test(result_heatmap['True Positive (TP)'], 
               result_heatmap['True Negative (TN)'], 
               result_heatmap['False Positive (FP)'], 
               result_heatmap['False Negative (FN)']) 
print("Chi-square Test for Cluster:")  
chi_square_test(result_cluster['True Positive (TP)'], 
               result_cluster['True Negative (TN)'], 
               result_cluster['False Positive (FP)'], 
               result_cluster['False Negative (FN)'])
print("Chi-square Test for Contour:")
chi_square_test(result_contour['True Positive (TP)'], 
               result_contour['True Negative (TN)'], 
               result_contour['False Positive (FP)'], 
               result_contour['False Negative (FN)'])
print("Chi-square Test for Cluster as Overlay:")
chi_square_test(result_cluster_overlay['True Positive (TP)'], 
               result_cluster_overlay['True Negative (TN)'], 
               result_cluster_overlay['False Positive (FP)'], 
               result_cluster_overlay['False Negative (FN)']) 

######################################
#Test
######################################

def calculate_metrics_101(csv_filepath):
    
    # Load the data from the CSV file
    dataframe = pd.read_csv(csv_filepath, encoding='utf-16', sep='\t', header=0)
    
    # Initialize dictionary for results
    metrics = {'F101_01': {'True Positive': 0, 'False Negative': 0}, 'F101_02': {'True Negative': 0, 'False Positive': 0}}
    
    metrics['F101_01']['True Positive'] = (dataframe["F101_01"] == 1).sum()
    metrics['F101_01']['False Negative'] = (dataframe["F101_01"] == 2).sum()
    metrics['F101_02']['True Negative'] = (dataframe["F101_02"] == 2).sum()
    metrics['F101_02']['False Positive'] = (dataframe["F101_02"] == 1).sum()
                 
    return metrics

metrics = calculate_metrics_101(file_path)
print("TEST: ", metrics)


def calculate_metrics_107(csv_filepath):
    dataframe = pd.read_csv(csv_filepath, encoding='utf-16', sep='\t', header=0)
    metrics = {
        'F107_01': {'True Positive': 0, 'False Negative': 0},
        'F107_02': {'True Negative': 0, 'False Positive': 0}
    }
    metrics['F107_01']['True Positive'] = (dataframe.iloc[:, 41] == 1).sum()
    metrics['F107_01']['False Negative'] = (dataframe.iloc[:, 41] == 2).sum()
    metrics['F107_02']['True Negative'] = (dataframe.iloc[:, 42] == 2).sum()
    metrics['F107_02']['False Positive'] = (dataframe.iloc[:, 42] == 1).sum()        
    return metrics

metrics = calculate_metrics_107(file_path)
print("TEST: ", metrics)






