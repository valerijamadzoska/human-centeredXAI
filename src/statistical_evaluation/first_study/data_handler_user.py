import pandas as pd
import matplotlib.pyplot as plt 


df = pd.read_csv('data\data_human-centeredXAI_2023-10-24_15-13.csv', encoding='utf-16', sep='\t', header=0)

def analyze_data(df, columns):
    counts = {}
    for column in columns:
        # Count the frequency of each unique value in the column
        value_counts = df[column].value_counts().sort_index()
        
        # Add the counts to the dictionary
        counts[column] = value_counts.to_dict()
        
    return counts

#counts = analyze_data(df, ['A001', 'A002', 'A004', 'A005'])
#print(counts)

def plot_result(counts, column, title, xlabel, ylabel, labels):
    """
    This function plots a bar chart with values displayed on top of each bar.
    
    :param counts: Dictionary containing the data to plot
    :param column: Name of the column in the dictionary to use for plotting
    :param title: Title of the chart
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param labels: Labels for the legend
    """
    data = counts[column]
    
    # Create a bar chart
    plt.figure(figsize=(8, 5))
    
    # Create bars and display values on top of each bar
    for i, (group, value) in enumerate(data.items()):
        plt.bar(i, value, label=labels.get(str(group), f"Gruppe: {group}"))
        plt.text(i, value, str(value), ha='center', va='bottom')
    
    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add legend
    plt.legend()
    
    # Display the chart
    plt.xticks(range(len(data)), list(data.keys()))  # Setting xticks labels
    plt.show()



# Assuming counts is the output from your analyze_data function
counts = analyze_data(df, ['A001', 'A002', 'A004', 'A005'])

#Plot Alter
age_labels = {
    "1": "unter 30",
    "2": "30-50",
    "3": "51-60",
    "4": "über 60",
    "5": "keine Angabe"
}
plot_result(counts, 'A001', 'Altersverteilung der Teilnehmer', 'Altersgruppe', 'Gruppengröße', age_labels)

#Plot Erfahrung KI
experience_labels = {
    "1": "keine",
    "2": "Etwas von KI gehört oder gelesen",
    "3": "KI-basierte Tools verwendet oder mit KI gearbeitet",
    "4": "selbst KI programmiert",
}
plot_result(counts, 'A002', 'Erfahrungen der Teilnehmer mit der KI', 'Niveau', 'Gruppengröße', experience_labels)

#Plot Erfahrung XAI
experience_labelsXAI = {
    "1": "keine",
    "2": "Etwas von XAI gehört oder gelesen",
    "3": "XAI-Techniken angewandt",
    "4": "selbst XAI Technik programmiert",
}
plot_result(counts, 'A004', 'Erfahrungen der Teilnehmer mit der XAI', 'Niveau', 'Gruppengröße', experience_labelsXAI)

#Plot Augenleiden
vision_labels = {
    "1": "Störung des Farbensehens",
    "2": "Schlechte Sehkraft",
    "4": "Andere Beeinträchtigungen der Augen",
    "3": "Keine Beeinträchtigungen",
    "5": "Keine Antwort"
}
plot_result(counts, 'A005', 'Beeinträchtigung durch Augenleiden', 'Beeinträchtigung', 'Gruppengröße', vision_labels)
