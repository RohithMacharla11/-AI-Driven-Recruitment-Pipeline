import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import openpyxl
import sklearn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

file_paths = glob.glob('./Approved_Datasets/*.xlsx')
datasets = [pd.read_excel(path) for path in file_paths]
final_data = pd.concat(datasets, ignore_index=True)

print(final_data.head())

final_data.info()

# Step 2: Data Cleaning
# Standardize column names
final_data.columns = [col.strip().lower().replace(" ", "_") for col in final_data.columns]

# Handle missing values
final_data.fillna("Unknown", inplace=True)

# Remove duplicates
final_data.drop_duplicates(inplace=True)

# Step 3: Data Preprocessing
# Convert decision to categorical type
final_data['decision'] = final_data['decision'].astype('category')

final_data['decision'] = final_data['decision'].str.lower()  # Convert to lowercase
final_data['decision'] = final_data['decision'].replace({
    'select': 'select',
    'selected': 'select',
    'reject': 'reject',
    'rejected': 'reject'
})

# Step 4: Exploratory Data Analysis
# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.countplot(data=final_data, x='decision')
plt.title("Decision Distribution")
plt.show()


# Display all unique roles in the 'roles' column
unique_roles = final_data['role'].unique()
print(unique_roles)

# Mapping dictionary for grouping roles
role_mapping = {
    'Software Developer': 'Software Engineer',
    'Software Engineer': 'Software Engineer',
    'Data Engineer': 'Data Analyst',
    'Data Analyst': 'Data Analyst',
    'UI/UX Designer': 'UI/UX Designer',
    'UI Engineer': 'UI/UX Designer',
    'UI Designer': 'UI/UX Designer',
    'Cybersecurity Specialist': 'Cybersecurity Specialist',
    'Network Engineer': 'Cybersecurity Specialist',
    'Cloud Architect': 'Cloud Architect',
    'DevOps Engineer': 'Cloud Architect',
    'AI Engineer': 'AI Engineer',
    'Machine Learning Engineer': 'AI Engineer',
    'System Administrator': 'System Administrator',
    'Database Administrator': 'System Administrator',
    'Digital Marketing Specialist': 'Digital Marketing Specialist',
    'Content Writer': 'Digital Marketing Specialist',
    'Graphic Designer': 'Mobile Game Developer',
    'Game Developer': 'Mobile Game Developer',
    'Mobile App Developer': 'Mobile Game Developer',
    'HR Specialis': 'Digital Marketing Specialist',
    'HR Specialist' : 'Digital Marketing Specialist',
    'Project Manager' : 'System Administrator'
}


# Apply the mapping to group roles
final_data['role'] = final_data['role'].replace(role_mapping)

# Check the unique values after grouping
print(final_data['role'].unique())


plt.figure(figsize=(10, 6))
sns.countplot(data=final_data, x='role')
plt.title("Role Distribution")
plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees
plt.show()


# Calculate the number of words in the 'Transcript' column
final_data['num_words_in_transcript'] = final_data['transcript'].apply(lambda x: len(str(x).split()))



# Group by Role and decision and calculate mean, median, and standard deviation
aggregated_data = final_data.groupby(['role', 'decision'])['num_words_in_transcript'].agg(['mean', 'median', 'std']).reset_index()

print(aggregated_data)

# Bar plot for mean word count by Role and decision
plt.figure(figsize=(12, 6))
sns.barplot(data=aggregated_data, x='role', y='mean', hue='decision', palette='viridis')
plt.title('Average Word Count in Transcripts by Role and Decision')
plt.xlabel('Role')
plt.ylabel('Mean Word Count')
plt.xticks(rotation=45)
plt.show()


# Box plot for distribution of word count by Role and decision
plt.figure(figsize=(12, 6))
sns.boxplot(data=final_data, x='role', y='num_words_in_transcript', hue='decision', palette='Set2')
plt.title('Word Count Distribution by Role and Decision')
plt.xlabel('Role')
plt.ylabel('Word Count')
plt.xticks(rotation=45)
plt.show()


# Save cleaned data
final_data.to_csv('cleaned_data.csv', index=False)