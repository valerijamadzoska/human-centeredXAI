from scipy.stats import wilcoxon
import pandas as pd

def wilcoxon_signed_rank_test(data_series, comparison_value):
    diff = data_series - comparison_value
    
    # Perform the Wilcoxon signed-rank test
    test_stat, p_value = wilcoxon(diff)
    
    return test_stat, p_value

# call
file = '/Users/valerijamadzoska/human-centeredXAI/data/first_study/userPerformance.csv'
data = pd.read_csv(file)
test_stat, p_value = wilcoxon_signed_rank_test(data['contour_Specificity'], 0.623)
print(f"Test statistic: {test_stat}, p-value: {p_value}")
