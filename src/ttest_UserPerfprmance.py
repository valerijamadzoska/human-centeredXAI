import pandas as pd
from scipy import stats

def append_column_means(filename):
    data = pd.read_csv(filename)
    #mean of each column
    column_means = data.mean()
    
    # Append the column means as the last row
    data = pd.concat([data, column_means.to_frame().T], ignore_index=True)
    
    # Write the updated data back to the file
    data.to_csv(filename, index=False)

def one_sample_ttest(filename, expected_mean=0.593, column_index=3):
    data = pd.read_csv(filename, skiprows=1, header=None)

    # Ignore the last row (column means), beware if code is run more often!
    data = data.iloc[:-1]

    # Select the data from the specified column
    sample_data = data.iloc[:, column_index]

    # Check for NaN values
    if sample_data.isnull().any():
        return "There are invalid (NaN) values in your data. Please clean your data and try again."

    # Perform a one-sample t-test
    t_stat, p_value = stats.ttest_1samp(sample_data, expected_mean)

    return t_stat, p_value

# call t-test
#p_value = one_sample_ttest(r'C:\Users\icke\Desktop\human-centeredXAI-1\data\userPerformance.csv')
# Print the result
#print("P-Value:")
#print(p_value)
# Call the append_column_means function
#append_column_means(r'C:\Users\icke\Desktop\human-centeredXAI-1\data\userPerformance.csv')
