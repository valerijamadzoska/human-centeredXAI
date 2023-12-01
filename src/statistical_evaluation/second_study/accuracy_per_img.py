
import pandas as pd
#Calculate accuracy for each img

columns2 = {
    "TP": [f"A{num:03d}_01" for num in [101, 201, 301, 401, 103, 203, 303, 403, 105, 205, 305, 405, 107, 207, 307, 407]],
    "TN": [f"A{num:03d}_02" for num in [102, 202, 302, 402, 104, 204, 304, 404, 106, 206, 306, 406, 108, 208, 308, 408]],
    "FN": [f"A{num:03d}_02" for num in [101, 201, 301, 401, 103, 203, 303, 403, 105, 205, 305, 405, 107, 207, 307, 407]],
    "FP": [f"A{num:03d}_01" for num in [102, 202, 302, 402, 104, 204, 304, 404, 106, 206, 306, 406, 108, 208, 308, 408]],
}
columns3 = {
    "TP": [f"B{num:03d}_01" for num in [101, 201, 301, 401, 103, 203, 303, 403, 105, 205, 305, 405, 107, 207, 307, 407]],
    "TN": [f"B{num:03d}_02" for num in [102, 202, 302, 402, 104, 204, 304, 404, 106, 206, 306, 406, 108, 208, 308, 408]],
    "FN": [f"B{num:03d}_02" for num in [101, 201, 301, 401, 103, 203, 303, 403, 105, 205, 305, 405, 107, 207, 307, 407]],
    "FP": [f"B{num:03d}_01" for num in [102, 202, 302, 402, 104, 204, 304, 404, 106, 206, 306, 406, 108, 208, 308, 408]],
}
columns4 = {
    "TP": [f"C{num:03d}_01" for num in [101, 201, 301, 401, 103, 203, 303, 403, 105, 205, 305, 405, 107, 207, 307, 407]],
    "TN": [f"C{num:03d}_02" for num in [102, 202, 302, 402, 104, 204, 304, 404, 106, 206, 306, 406, 108, 208, 308, 408]],
    "FN": [f"C{num:03d}_02" for num in [101, 201, 301, 401, 103, 203, 303, 403, 105, 205, 305, 405, 107, 207, 307, 407]],
    "FP": [f"C{num:03d}_01" for num in [102, 202, 302, 402, 104, 204, 304, 404, 106, 206, 306, 406, 108, 208, 308, 408]],
}
def calculate_accuracy(metrics: dict) -> float:
    TP = metrics['TP']
    TN = metrics['TN']
    FP = metrics['FP']
    FN = metrics['FN']
    
    # Check to avoid division by zero
    if TP + TN + FP + FN == 0:
        return None
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def calculate_metrics_all_Img(data: pd.DataFrame) -> dict:
    data = pd.read_csv(file_path, encoding='utf-8', sep=';', header=0)
    all_metrics = {}
    
    # Looping through the specified range of objects
    for i in range(101, 409):
        obj = f"C{i:03d}"  #ADJUSTMENT HERE!!!!!! A/B/C
        metrics = {"TP": 0, "TN": 0, "FN": 0, "FP": 0}
        
        # Defining the column names based on the object
        #ADJUSTMENT HERE!!!!!! columns
        TP_column = f"{obj}_01" if f"{obj}_01" in columns4['TP'] else 0
        TN_column = f"{obj}_02" if f"{obj}_02" in columns4['TN'] else 0
        FN_column = f"{obj}_02" if f"{obj}_02" in columns4['FN'] else 0
        FP_column = f"{obj}_01" if f"{obj}_01" in columns4['FP'] else 0
        
        # Counting the number of '2's in each specified column
        metrics["TP"] = (data[TP_column] == 2).sum() if TP_column else 0
        metrics["TN"] = (data[TN_column] == 2).sum() if TN_column else 0
        metrics["FN"] = (data[FN_column] == 2).sum() if FN_column else 0
        metrics["FP"] = (data[FP_column] == 2).sum() if FP_column else 0

        # Calculating accuracy
        metrics['Accuracy'] = calculate_accuracy(metrics)

        # Adding the metrics for each object to the overall dictionary
        all_metrics[obj] = metrics
        
    return all_metrics

    # Method call
file_path = '/Users/valerijamadzoska/human-centeredXAI/data/second_study/data_human-centered_XAI_2023-12-01_01-29.csv'
# Calculate and display the metrics for each object
metrics = calculate_metrics_all_Img(file_path)
print(metrics)