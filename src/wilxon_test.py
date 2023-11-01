from scipy.stats import wilcoxon
import pandas as pd

def wilcoxon_signed_rank_test(data_series, comparison_value):
    diff = data_series - comparison_value
    
    # Perform the Wilcoxon signed-rank test
    test_stat, p_value = wilcoxon(diff)
    
    return test_stat, p_value

# call
file = r'C:\Users\icke\Desktop\human-centeredXAI-1\data\userPerformance.csv'
data = pd.read_csv(file)
test_stat, p_value = wilcoxon_signed_rank_test(data['cluster_overlay_Accuracy'], 0.598)
print(f"Test statistic: {test_stat}, p-value: {p_value}")
