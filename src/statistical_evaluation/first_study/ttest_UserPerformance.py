import pandas as pd
from scipy import stats

def append_column_means(filename):
    # Daten aus der Datei lesen
    data = pd.read_csv(filename)
    
    # Durchschnitt jeder Spalte berechnen
    column_means = data.mean()
    
    # Durchschnittswerte als letzte Zeile einfügen
    data = data.append(column_means, ignore_index=True)
    
    # Die aktualisierten Daten in die Datei zurückschreiben
    data.to_csv(filename, index=False)


def one_sample_ttest(filename, expected_mean=0.593):
    # Daten aus der Datei lesen
    data = pd.read_csv(filename, skiprows=1, header=None)

    # Die letzte Zeile ignorieren
    data = data.iloc[:-1]
    
    # Die Daten der ersten Spalte auswählen
    sample_data = data.iloc[:, 3]
    
    # Überprüfen, ob die Daten gültig sind
    if sample_data.isnull().any():
        return "Es gibt ungültige (NaN) Werte in Ihren Daten. Bitte bereinigen Sie Ihre Daten und versuchen Sie es erneut."
    
    # One-Sample t-Test durchführen
    t_stat, p_value = stats.ttest_1samp(sample_data, expected_mean)
    
    return t_stat, p_value




#p_value = one_sample_ttest('/Users/valerijamadzoska/Desktop/human-centeredXAI-1/data/metricsEachUser.csv')
#print("Hallo")
#print(p_value)

append_column_means('/Users/valerijamadzoska/Desktop/human-centeredXAI-1/data/metricsEachUser.csv')
