import scipy.stats as stats

def one_sample_t_test(sample_data, mean, alpha=0.05):
    t_stat, p_value = stats.ttest_1samp(sample_data, mean)
    if p_value < alpha:
        print(f"Mit einem p-Wert von {p_value:.4f} wird die Nullhypothese abgelehnt.")
    else:
        print(f"Mit einem p-Wert von {p_value:.4f} gibt es nicht genug Beweise, um die Nullhypothese abzulehnen.")

# Method call
mean = 0.592 #Heatmap Accuracy

#Cluster as Overlay
sample_data = [0.66, 0.83, 0.477, 0.933, 0.551, 0.495, 0.574, 0.585]
one_sample_t_test(sample_data, mean)

#Cluster
sample_data = [0.897, 0.561, 0.745, 0.916, 0.832, 0.915, 0.953, 0.852]
one_sample_t_test(sample_data, mean)

#Contour
sample_data = [0.764, 0.685, 0.393, 0.943, 0.876, 0.99, 0.879, 0.915]
one_sample_t_test(sample_data, mean)