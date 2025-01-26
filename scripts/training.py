import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import openpyxl
import sklearn
import warnings

# Load cleaned data
final_data = pd.read_csv('Generated_Data/cleaned_data.csv')
print("Data Loaded")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

print("Libraries Imported")
# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Apply preprocessing
for col in ['transcript', 'resume', 'reason_for_decision', 'job_description']:
    final_data[col] = final_data[col].apply(preprocess_text)
    
final_data.head()

# Feature: Word count
final_data['transcript_word_count'] = final_data['transcript'].apply(lambda x: len(str(x).split()))
final_data['resume_word_count'] = final_data['resume'].apply(lambda x: len(str(x).split()))


# Sentiment analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
final_data['transcript_sentiment'] = final_data['transcript'].apply(lambda x: sia.polarity_scores(x)['compound'])


# TF-IDF vectorization and similarity scores
vectorizer = TfidfVectorizer()
job_desc_vectors = vectorizer.fit_transform(final_data['job_description'])
resume_vectors = vectorizer.transform(final_data['resume'])
transcript_vectors = vectorizer.transform(final_data['transcript'])

final_data['resume_job_similarity'] = [cosine_similarity(resume_vectors[i], job_desc_vectors[i])[0][0] for i in range(len(final_data))]
final_data['transcript_job_similarity'] = [cosine_similarity(transcript_vectors[i], job_desc_vectors[i])[0][0] for i in range(len(final_data))]


# Visualization example: Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=final_data, x='decision', y='resume_job_similarity', hue='role')
plt.title("Resume-Job Similarity by Decision and Role")
plt.show()

# Word cloud for selected candidates
selected_transcripts = " ".join(final_data[final_data['decision'] == 'select']['transcript'])
wordcloud = WordCloud(background_color='white').generate(selected_transcripts)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Selected Candidates' Transcripts")
plt.show()

print(final_data.head())  # Check the first few rows
print(final_data.info())  # Check for missing or null values
print(final_data.describe())  # Summary statistics for numerical columns


plt.figure(figsize=(10, 8))
sns.heatmap(final_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=final_data,
    x='resume_job_similarity',
    y='transcript_job_similarity',
    hue='decision',
    style='role',
    palette='coolwarm'
)
plt.title("Resume-Job vs. Transcript-Job Similarity by Decision")
plt.xlabel("Resume-Job Similarity")
plt.ylabel("Transcript-Job Similarity")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(data=final_data, x='resume_job_similarity', hue='decision', fill=True, common_norm=False)
plt.title("Resume-Job Similarity Distribution by Decision")
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=final_data, x='transcript_job_similarity', hue='decision', fill=True, common_norm=False)
plt.title("Transcript-Job Similarity Distribution by Decision")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=final_data, y='num_words_in_transcript', x='role', palette='Set3')
plt.title("Outlier Detection in Transcript Word Counts")
plt.xticks(rotation=45)
plt.show()

# ASSIGNMENT-3

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to extract interviewer comments (assuming they are prefixed with "interviewer")
def interviewer_sentiment(text):
    interviewer_text = " ".join([line for line in text.splitlines() if line.lower().startswith("interviewer")])
    return sia.polarity_scores(interviewer_text)['compound']

# Apply function to calculate sentiment
final_data['interviewer_sentiment'] = final_data['transcript'].apply(interviewer_sentiment)

from nltk.tokenize import sent_tokenize

final_data['transcript_length_sentences'] = final_data['transcript'].apply(
    lambda x: len(sent_tokenize(x)) if pd.notnull(x) else 0
)


# Display summary statistics for the new features
print(final_data[['interviewer_sentiment', 'transcript_length_sentences']].describe())


from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER model
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to calculate compound sentiment score
def calculate_vader_sentiment(text):
    if pd.isnull(text) or text.strip() == '':
        return 0  # Assign 0 sentiment for empty or null text
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score ranges from -1 (negative) to 1 (positive)

# Apply to the 'resume' column
final_data['resume_sentiment'] = final_data['resume'].apply(calculate_vader_sentiment)


import seaborn as sns
import matplotlib.pyplot as plt

# Plot sentiment distribution
sns.histplot(final_data['resume_sentiment'], kde=True, bins=30)
plt.title("Distribution of Resume Sentiment")
plt.xlabel("Resume Sentiment")
plt.ylabel("Frequency")
plt.show()


# Assuming 'transcript_sentiment' and 'resume_sentiment' columns exist
final_data['sentiment_interaction'] = final_data['transcript_sentiment'] * final_data['resume_sentiment']


# Avoid division by zero by replacing 0 in resume_word_count with NaN, then filling with a small value
final_data['resume_word_count'] = final_data['resume_word_count'].replace(0, np.nan).fillna(1)
final_data['word_count_ratio'] = final_data['transcript_word_count'] / final_data['resume_word_count']


from sklearn.feature_extraction.text import CountVectorizer

# Function to calculate overlap
def calculate_overlap(transcript, resume):
    if pd.isnull(transcript) or pd.isnull(resume):
        return 0
    transcript_words = set(transcript.split())
    resume_words = set(resume.split())
    return len(transcript_words.intersection(resume_words))

# Apply the function
final_data['resume_transcript_overlap'] = final_data.apply(
    lambda row: calculate_overlap(row['transcript'], row['resume']), axis=1
)

import seaborn as sns
import matplotlib.pyplot as plt

# Histograms
features_to_plot = ['sentiment_interaction', 'word_count_ratio', 'resume_transcript_overlap']
for feature in features_to_plot:
    sns.histplot(final_data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.show()


# Correlation Matrix with Decision (Numeric)
final_data['decision_numeric'] = final_data['decision'].map({'reject': 0, 'select': 1})
correlation_features = ['sentiment_interaction', 'word_count_ratio', 'resume_transcript_overlap', 'decision_numeric']
correlation_matrix = final_data[correlation_features].corr()

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation with Decision")
plt.show()

final_data.drop(columns=['transcript_length_sentences', 'decision_numeric'], inplace=True)  # Remove the temporary columns

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def calculate_keyword_matching_score_for_dataset(data):
    """
    Calculate the Keyword Matching Score for each row in the dataset by extracting keywords
    from the job description and checking their presence in the transcript.

    Args:
        data (pd.DataFrame): Dataset with 'job_description' and 'transcript' columns.

    Returns:
        pd.DataFrame: Dataset with an additional column 'keyword_matching_score'.
    """
    # Function to extract keywords and calculate the score for a single row
    def calculate_score(row):
        # Extract job description and transcript
        job_description = row['job_description']
        transcript = row['transcript']

        # Extract keywords from the job description
        keywords = job_description.split()  # Splitting by spaces (you can refine this as needed)
        keywords = list(set(keywords))  # Remove duplicates

        # Combine transcript and keywords for vectorization
        texts = [transcript, " ".join(keywords)]

        # Vectorize text
        vectorizer = CountVectorizer(vocabulary=keywords, binary=True)
        keyword_matrix = vectorizer.fit_transform(texts).toarray()

        # Extract keyword counts
        transcript_keywords = keyword_matrix[0].sum()
        total_keywords = keyword_matrix[1].sum()

        # Calculate the transcript score
        transcript_score = transcript_keywords / total_keywords if total_keywords else 0

        return transcript_score

    # Apply the calculation to each row in the dataset
    data['keyword_matching_score'] = data.apply(calculate_score, axis=1)
    return data


# Calculate keyword matching scores
data_with_scores = calculate_keyword_matching_score_for_dataset(final_data)
print(data_with_scores)

import pandas as pd

def calculate_confidence_score(row):

    transcript_sentiment = row['transcript_sentiment']
    interviewer_sentiment = row['interviewer_sentiment']

    # Confidence is derived from the product of the two sentiments
    interaction = transcript_sentiment * interviewer_sentiment

    # Normalization: Scale to [0, 1] assuming sentiment values are in [0, 1]
    # If sentiments are in [-1, 1], adjust accordingly
    confidence_score = max(0, min(interaction, 1))  # Clamp to valid range

    return confidence_score

def apply_confidence_scores(data):

    data['confidence_score'] = data.apply(calculate_confidence_score, axis=1)
    return data


# Apply confidence scores
final_data_with_scores = apply_confidence_scores(final_data)
print(final_data_with_scores)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_interaction_score(resume, transcript):
    if not resume or not transcript:  # Handle missing or empty values
        return 0

    # Combine resume and transcript into a list for vectorization
    documents = [resume, transcript]

    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

def apply_interaction_scores(data):

    data['interaction_score'] = data.apply(
        lambda row: calculate_interaction_score(row['resume'], row['transcript']), axis=1
    )
    return data



final_data_with_interaction_scores = apply_interaction_scores(final_data)
print(final_data_with_interaction_scores[['resume', 'transcript', 'interaction_score']])


final_data.head()

# Check for empty or null values in the transcript column
print(final_data['transcript'].isnull().sum())  # Check nulls
print(final_data['transcript'].apply(len).min())  # Check minimum length

import pandas as pd
import re

# Define a list of self-advocacy keywords/phrases
SELF_ADVOCACY_KEYWORDS = [
    r'\bachieved\b', r'\bled\b', r'\bdeveloped\b', r'\bimproved\b',
    r'\bmanaged\b', r'\binitiated\b', r'\borganized\b', r'\bimplemented\b',
    r'\bdesigned\b', r'\bcreated\b', r'\bexecuted\b', r'\bstrength\b',
    r'\bstrengths\b', r'\bachievement\b', r'\bachievements\b',
    r'\bcontributed\b', r'\binnovated\b', r'\btransformed\b'
]

# Compile regex patterns for efficiency
pattern = re.compile('|'.join(SELF_ADVOCACY_KEYWORDS), re.IGNORECASE)

def calculate_self_advocacy_score(transcript):
    if pd.isnull(transcript) or not isinstance(transcript, str):
        return 0.0

    matches = pattern.findall(transcript)
    word_count = len(transcript.split())

    if word_count == 0:
        return 0.0

    # Normalize the count by word count to get a density
    advocacy_density = len(matches) / word_count

    # Scale to [0,1] with an upper cap (e.g., 0.05)
    scaled_score = min(advocacy_density * 100, 1.0)

    return scaled_score

def apply_self_advocacy_score(data):
    data['self_advocacy_score'] = data['transcript'].apply(calculate_self_advocacy_score)
    return data

final_data_with_self_advocacy = apply_self_advocacy_score(final_data)
print(final_data_with_self_advocacy[['transcript', 'self_advocacy_score']])

import pandas as pd

def calculate_sentiment_alignment(row):
    candidate_sentiment = row.get('transcript_sentiment', 0)
    interviewer_sentiment = row.get('interviewer_sentiment', 0)

    # Compute similarity using the absolute difference
    alignment = 1 - abs(candidate_sentiment - interviewer_sentiment)

    return alignment  # Already between 0 and 1

def apply_sentiment_alignment(data):
    data['sentiment_alignment_score'] = data.apply(calculate_sentiment_alignment, axis=1)
    return data

final_data_with_alignment = apply_sentiment_alignment(final_data)
print(final_data_with_alignment[['transcript_sentiment', 'interviewer_sentiment', 'sentiment_alignment_score']])

final_data.head()

final_data.columns

# Save Featured Final data
final_data.to_csv('Generated_Data/featured_final_data.csv', index=False)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Convert the data into a DataFrame
df = pd.DataFrame(final_data)

# Encode the target variable ('decision') into binary (0 for reject, 1 for select)
df['decision'] = df['decision'].map({'reject': 0, 'select': 1})

# List of features to scale/normalize
features_to_scale = [
    'num_words_in_transcript',
       'transcript_word_count', 'resume_word_count', 'transcript_sentiment',
       'resume_job_similarity', 'transcript_job_similarity',
       'interviewer_sentiment', 'resume_sentiment', 'sentiment_interaction',
       'word_count_ratio', 'resume_transcript_overlap',
       'keyword_matching_score', 'confidence_score', 'interaction_score',
       'self_advocacy_score', 'sentiment_alignment_score'
]

# Using StandardScaler for standardization
scaler = StandardScaler()
df_scaled = df.copy()  # Create a copy to retain original data
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Display scaled features
print("Scaled DataFrame:\n", df_scaled)


# Optionally, use MinMaxScaler for normalization if required
minmax_scaler = MinMaxScaler()
df_normalized = final_data.copy()
df_normalized[features_to_scale] = minmax_scaler.fit_transform(final_data[features_to_scale])

# Display normalized features
print("\nNormalized DataFrame:\n", df_normalized)

# Import necessary libraries
from sklearn.model_selection import train_test_split

# Define the target variable (Y) and features (X)
# Drop irrelevant columns to extract feature variables
X = final_data.drop(columns=['decision', 'id', 'name', 'role', 'transcript', 'resume', 'reason_for_decision', 'job_description'])
Y = final_data['decision']

# Split the data into 80% training and 20% testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Output shapes to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# Logistic Regerssion

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score


# Logistic Regression and its hyperparameters
log_reg = LogisticRegression(random_state=42, max_iter=1000)
param_grid = {
    "C": [0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

# Fit the model on training data
grid_search.fit(X_train, Y_train)

# Get the best model
best_log_reg = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# Make predictions
Y_pred = best_log_reg.predict(X_test)
Y_pred_proba = best_log_reg.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy_lr = accuracy_score(Y_test, Y_pred)
roc_auc_lr = roc_auc_score(Y_test, Y_pred_proba)

print("Accuracy:", accuracy_lr)
print("ROC AUC Score:", roc_auc_lr)

print(f"Best Logistic Regression Model: {best_log_reg}")
print(f"Test Set Accuracy: {accuracy_lr:.4f}")
print(f"Test Set ROC AUC Score: {roc_auc_lr:.4f}")

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
# Decision Tree and its hyperparameters
decision_tree = DecisionTreeClassifier(random_state=42)
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

# Fit the model on training data
grid_search.fit(X_train, Y_train)

# Get the best model
best_decision_tree = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Make predictions
Y_pred = best_decision_tree.predict(X_test)
Y_pred_proba = best_decision_tree.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy_dt = accuracy_score(Y_test, Y_pred)
roc_auc_dt = roc_auc_score(Y_test, Y_pred_proba)

print("Accuracy:", accuracy_dt)
print("ROC AUC Score:", roc_auc_dt)


print(f"Best Decision Tree Model: {best_decision_tree}")
print(f"Test Set Accuracy: {accuracy_dt:.4f}")
print(f"Test Set ROC AUC Score: {roc_auc_dt:.4f}")


# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Encode the target labels
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model directly
rf_model.fit(X_train, Y_train_encoded)

# Predictions
Y_pred = rf_model.predict(X_test)
Y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy_rf = accuracy_score(Y_test_encoded, Y_pred)
roc_auc_rf = roc_auc_score(Y_test_encoded, Y_pred_proba)

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"ROC AUC Score: {roc_auc_rf:.4f}")

# XGBoost

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Encode the target labels
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

# Define the model
xgboost_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train the model directly
xgboost_model.fit(X_train, Y_train_encoded)

# Predictions
Y_pred = xgboost_model.predict(X_test)
Y_pred_proba = xgboost_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy_xgb = accuracy_score(Y_test_encoded, Y_pred)
roc_auc_xgb = roc_auc_score(Y_test_encoded, Y_pred_proba)

print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"ROC AUC Score: {roc_auc_xgb:.4f}")

# LIGHTGBM

import lightgbm as lgb

# Define the model
lgb_model = lgb.LGBMClassifier(random_state=42)

# Train the model
lgb_model.fit(X_train, Y_train_encoded)

# Predictions
Y_pred_lgb = lgb_model.predict(X_test)
Y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy_lgb = accuracy_score(Y_test_encoded, Y_pred_lgb)
roc_auc_lgb = roc_auc_score(Y_test_encoded, Y_pred_proba_lgb)

print(f"LightGBM Accuracy: {accuracy_lgb:.4f}")
print(f"LightGBM ROC AUC Score: {roc_auc_lgb:.4f}")


# SVM

# Import necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# Create an SVM model
svm_model = SVC(probability=True, kernel='rbf', C=1, gamma='scale')  # Directly using parameters

# Train the model
svm_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred_svm = svm_model.predict(X_test)
Y_pred_prob_svm = svm_model.predict_proba(X_test)[:, 1]  # Probability for ROC AUC

# Evaluate model using Accuracy and ROC AUC Score
accuracy_svm = accuracy_score(Y_test, Y_pred_svm)
roc_auc_svm = roc_auc_score(Y_test, Y_pred_prob_svm)

print(f"Accuracy of SVM model: {accuracy_svm:.4f}")
print(f"ROC AUC Score of SVM model: {roc_auc_svm:.4f}")


print("Comparing All the models(POST MODEL ANALYSIS)")

print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}, ROC AUC: {roc_auc_lr:.4f}")
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}, ROC AUC: {roc_auc_dt:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}, ROC AUC: {roc_auc_rf:.4f}")
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}, ROC AUC: {roc_auc_xgb:.4f}")
print(f"LightGBM Accuracy: {accuracy_lgb:.4f}, ROC AUC: {roc_auc_lgb:.4f}")
print(f"SVM Accuracy: {accuracy_svm:.4f}, ROC AUC: {roc_auc_svm:.4f}")


import matplotlib.pyplot as plt

# Model names
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost','LightGBM', 'SVM']

# Accuracy scores for each model
accuracy_scores = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_xgb,accuracy_lgb, accuracy_svm]

# ROC AUC scores for each model
roc_auc_scores = [roc_auc_lr, roc_auc_dt, roc_auc_rf, roc_auc_xgb,roc_auc_lgb, roc_auc_svm]

# Plotting the bar chart
x = range(len(models))
width = 0.4  # bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Accuracy bar chart
ax.bar(x, accuracy_scores, width=width, label='Accuracy', color='royalblue', align='center')

# ROC AUC bar chart
ax.bar([p + width for p in x], roc_auc_scores, width=width, label='ROC AUC', color='orange', align='center')

# Adding labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Models: Accuracy and ROC AUC')
ax.set_xticks([p + width / 2 for p in x])
ax.set_xticklabels(models)

# Show legend
ax.legend()

# Show plot
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# For Random Forest
importances_rf = rf_model.feature_importances_
indices_rf = importances_rf.argsort()

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.barh(range(len(importances_rf)), importances_rf[indices_rf], align="center")
plt.yticks(range(len(importances_rf)), [X_train.columns[i] for i in indices_rf])
plt.xlabel("Relative Importance")
plt.show()

# For XGBoost
importances_xgb = xgboost_model.feature_importances_
indices_xgb = importances_xgb.argsort()

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (XGBoost)")
plt.barh(range(len(importances_xgb)), importances_xgb[indices_xgb], align="center")
plt.yticks(range(len(importances_xgb)), [X_train.columns[i] for i in indices_xgb])
plt.xlabel("Relative Importance")
plt.show()

# ASSIGNMENT_4

import shap

# Explainer and SHAP values
explainer = shap.TreeExplainer(lgb_model)  # Replace `lgb_model` with your trained LightGBM model
shap_values = explainer.shap_values(X_test)

# Convert shap_values[1] to a 2D array if needed
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]  # Select positive class (1)
else:
    shap_values_to_plot = shap_values  # Use directly if not a list

# SHAP Beeswarm Plot
shap.summary_plot(shap_values_to_plot, X_test, plot_type="dot")


import shap

# Verify and extract SHAP values and base values
if isinstance(explainer.expected_value, list):
    base_value = explainer.expected_value[1]  # Base value for the positive class
else:
    base_value = explainer.expected_value  # For models with a single output

if isinstance(shap_values, list):
    shap_values_class = shap_values[1]  # SHAP values for the positive class
else:
    shap_values_class = shap_values  # For models with a single output

# Low prediction example
shap.waterfall_plot(
    shap.Explanation(
        base_values=base_value,
        values=shap_values_class[0],
        feature_names=X_test.columns,
        data=X_test.iloc[0],
    )
)

# Medium prediction example
shap.waterfall_plot(
    shap.Explanation(
        base_values=base_value,
        values=shap_values_class[len(X_test) // 2],
        feature_names=X_test.columns,
        data=X_test.iloc[len(X_test) // 2],
    )
)

# High prediction example
shap.waterfall_plot(
    shap.Explanation(
        base_values=base_value,
        values=shap_values_class[-1],
        feature_names=X_test.columns,
        data=X_test.iloc[-1],
    )
)


import shap
import numpy as np
import pandas as pd

# Ensure you have your SHAP values and test data loaded
# Replace these with your actual variables
# shap_values: The SHAP values from your model explanation
# X_test: The test dataset used for generating SHAP values

# Example: Assuming shap_values and X_test are already defined
# shap_values = explainer.shap_values(X_test)  # For your trained model
# X_test = pd.DataFrame(...)  # Test dataset

# List of features to plot
features_to_plot = ['sentiment_interaction', 'word_count_ratio', 'resume_transcript_overlap']

# Check the structure of shap_values
if isinstance(shap_values, list):  # Multi-class classification
    print("Multi-class model detected. Number of classes:", len(shap_values))
    class_index = 0  # Choose the target class index (modify as needed)
    shap_values_to_use = shap_values[class_index]
else:  # Regression or binary classification
    print("Regression or binary classification model detected.")
    shap_values_to_use = shap_values

# Ensure the shapes match
assert shap_values_to_use.shape[1] == X_test.shape[1], (
    f"Mismatch between SHAP values and X_test. "
    f"SHAP values: {shap_values_to_use.shape[1]}, X_test: {X_test.shape[1]}"
)

# Generate dependence plots
for feature in features_to_plot:
    if feature in X_test.columns:
        print(f"Generating dependence plot for: {feature}")
        shap.dependence_plot(feature, shap_values_to_use, X_test)
    else:
        print(f"Feature {feature} not found in X_test columns.")

# Additional debugging (optional)
print("SHAP values shape:", shap_values_to_use.shape)
print("X_test columns:", X_test.columns.tolist())


from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator

class SklearnCompatibleLGBMClassifier(LGBMClassifier, BaseEstimator):
    pass

# Use the wrapped class
lgb_model = SklearnCompatibleLGBMClassifier()
lgb_model.fit(X_train, Y_train)


from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Define features for PDP
features_to_plot = [0, 1, 2]  # Replace with actual indices or names

# Plot Partial Dependence for each feature
for feature in features_to_plot:
    print(f"Generating PDP for feature index {feature}...")
    PartialDependenceDisplay.from_estimator(
        lgb_model,
        X_test,
        [feature],  # Single feature for 1D PDP
        grid_resolution=50
    )
    plt.show()  # Show the plot for each feature


from sklearn.inspection import PartialDependenceDisplay

# 2D Partial Dependence Plot for the top 2 features (replace indices 0 and 1 with actual feature indices)
PartialDependenceDisplay.from_estimator(
    lgb_model,
    X_test,
    [(0, 1)],  # Indices of the top two features
    grid_resolution=50
)



import pickle
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgboost_model, f)

import joblib
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')



# # Assingmet 5")

# # Import required modules
# from transformers import DistilBertTokenizer, DistilBertModel
# import torch
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping# Load DistilBERT tokenizer and model

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# final_data.head()

# # Function to get BERT embeddings in batches
# def get_bert_embeddings_batch(texts, tokenizer, model, batch_size=32):
#     embeddings = []
#     total_batches = (len(texts) + batch_size - 1) // batch_size
#     print(f"Total Batches: {total_batches}")

#     for i in range(total_batches):
#         print(f"Processing batch {i + 1}/{total_batches}...")
#         batch = texts[i * batch_size:(i + 1) * batch_size]
#         inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=256)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#         embeddings.extend(batch_embeddings)

#     return embeddings

# # Example usage: Generate embeddings for transcripts
# texts = final_data['transcript'].tolist()
# batch_embeddings = get_bert_embeddings_batch(texts, tokenizer, model, batch_size=32)
# final_data['bert_embeddings_trans'] = batch_embeddings

# # Example usage: Generate embeddings for Job Description
# job_texts = final_data['job_description'].tolist()
# job_desc_batch_embeddings = get_bert_embeddings_batch(job_texts, tokenizer, model, batch_size=32)
# final_data['bert_embeddings_job_desc'] = job_desc_batch_embeddings

# # Example usage: Generate embeddings for Resume
# resume_texts = final_data['resume'].tolist()
# resume_batch_embeddings = get_bert_embeddings_batch(resume_texts, tokenizer, model, batch_size=32)
# final_data['bert_embeddings_resume'] = resume_batch_embeddings

# final_data.head()

# final_data.to_excel('Emmbeeding_final_data.xlsx', index=False)


# print("Expanding embeddings into separate columns...")
# trans_expanded = pd.DataFrame(final_data['bert_embeddings_trans'].tolist(), index=final_data.index)
# trans_expanded.columns = [f'trans_emb_{i}' for i in range(trans_expanded.shape[1])]

# resume_expanded = pd.DataFrame(final_data['bert_embeddings_resume'].tolist(), index=final_data.index)
# resume_expanded.columns = [f'resume_emb_{i}' for i in range(resume_expanded.shape[1])]

# jd_expanded = pd.DataFrame(final_data['bert_embeddings_job_desc'].tolist(), index=final_data.index)
# jd_expanded.columns = [f'jd_emb_{i}' for i in range(jd_expanded.shape[1])]

# # Concatenate the expanded embeddings with the original data
# df_expanded = pd.concat([final_data, trans_expanded, resume_expanded, jd_expanded], axis=1)

# # Drop the original embeddings columns
# df_expanded = df_expanded.drop(columns=['bert_embeddings_trans', 'bert_embeddings_resume', 'bert_embeddings_job_desc'])

# # Step 3: Encode the target column
# print("Encoding the decision column...")
# label_encoder = LabelEncoder()
# df_expanded['decision_encoded'] = label_encoder.fit_transform(df_expanded['decision'])


# # Step 4: Split data into features and target
# embedding_columns = [col for col in df_expanded.columns if col.startswith('trans_emb_') or col.startswith('resume_emb_') or col.startswith('jd_emb_')]
# X = df_expanded[embedding_columns]
# y = df_expanded['decision_encoded']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # Step 5: Train the model
# print("Training the XGBoost model...")
# model = XGBClassifier(use_label_encoder=False, max_depth=5, n_estimators=200, learning_rate=0.1, eval_metric='logloss')
# model.fit(X_train, y_train)

# # Step 6: Evaluate the model
# print("Evaluating the model...")
# y_pred = model.predict(X_test)

# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Import necessary libraries
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# import pandas as pd

# # Define the Neural Network model
# model = Sequential([
#     # Input layer
#     Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
#     BatchNormalization(),  # Normalize inputs for faster convergence
#     Dropout(0.1),  # Dropout to prevent overfitting

#     # Hidden layers
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.1),

#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.1),

#     Dense(32, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.1),

#     # Output layer
#     Dense(1, activation='sigmoid')  # For binary classification
# ])


# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',  # For binary classification
#     metrics=['accuracy']
# )

# # Train the neural network
# history = model.fit(
#     X_train,
#     y_train,
#     epochs=100,  # Increase max epochs for deeper models
#     batch_size=8,
#     validation_split=0.2,
#     # callbacks=[early_stopping],  # Use early stopping
#     verbose=1
# )

# # Predict using the trained neural network
# y_test_pred_nn = model.predict(X_test)
# y_test_pred_nn = y_test_pred_nn.flatten()  # Flatten the predictions to 1D array

# test_df = pd.DataFrame()
# test_df['actuals'] = y_test
# test_df['xgb_pred'] = y_pred  # XGBoost predicted probabilities
# test_df['nn_pred'] = y_test_pred_nn  # Neural network predicted probabilities

# # Combine predictions using mean probability
# test_df['mean_prob'] = (test_df['xgb_pred'] + test_df['nn_pred']) / 2
# test_df['new_pred'] = test_df['mean_prob'].round()  # Round mean probability to get final binary predictions

# # Evaluate the combined model
# accuracy = accuracy_score(test_df['actuals'], test_df['new_pred'])
# roc_auc_xgb = roc_auc_score(test_df['actuals'], test_df['xgb_pred'])
# roc_auc_mean = roc_auc_score(test_df['actuals'], test_df['mean_prob'])

# print(f"Accuracy of Combined Model: {accuracy:.4f}")
# print(f"ROC AUC Score (XGBoost): {roc_auc_xgb:.4f}")
# print(f"ROC AUC Score (Mean Probabilities): {roc_auc_mean:.4f}")

# # Optional: Display classification report for combined predictions
# print("\nClassification Report for Combined Model:")
# print(classification_report(test_df['actuals'], test_df['new_pred']))

# # Show the test DataFrame
# print("\nTest DataFrame:")
# print(test_df.head())


