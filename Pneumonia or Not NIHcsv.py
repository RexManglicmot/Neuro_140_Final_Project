import pandas as pd

# Load the CSV file
df = pd.read_csv('/Users/Rex/PycharmProjects/Neuro 140/archive (13)/Data_Entry_2017.csv')

# Display the first few rows of the dataframe
print(df.head())

# Count occurrences where the string 'Pneumonia' appears in the 'Diagnosis' column
pneumonia_count = df['Finding Labels'].str.contains('Pneumonia', na=False).sum()
print("Number of cases with Pneumonia:", pneumonia_count)

# Count occurrences where the string 'No Findings' appears in the 'Diagnosis' column
no_findings_count = df['Finding Labels'].str.contains('No Finding', na=False, case=False).sum()
print("Number of cases with No Findings:", no_findings_count)


# Load the CSV file
df2 = pd.read_csv('/Users/Rex/PycharmProjects/Neuro 140/vinxray/vietnam_train.csv')

# Display the first few rows of the dataframe
print(df2.head())

# Count occurrences where the string 'Pneumonia' appears in the 'Diagnosis' column
pneumonia_count2 = df2['class_name'].str.contains('Pneumonia', na=False).sum()
print("Number of cases with Pneumonia:", pneumonia_count2)

# Count occurrences where the string 'No Findings' appears in the 'Diagnosis' column
no_findings_count2 = df2['class_name'].str.contains('No finding', na=False, case=False).sum()
print("Number of cases with No Findings:", no_findings_count2)


import pandas as pd

# Load the CSV file
df3 = pd.read_csv('/Users/Rex/Downloads/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')

# Display the first few rows of the dataframe
print(df3.head())

# Assuming 1 represents 'Pneumonia' and 0 represents 'No finding'
pneumonia_count2 = (df3['Target'] == 1).sum()
no_findings_count2 = (df3['Target'] == 0).sum()

print("RSNA Number of cases with Pneumonia:", pneumonia_count2)
print("RSNA Number of cases with No Findings:", no_findings_count2)
