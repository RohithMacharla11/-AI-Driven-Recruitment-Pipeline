import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re


def preprocess_and_analyze_data(prediction_file_path, loaded_vectorizer):
    # Step 1: Read the prediction file
    prediction_df = pd.read_excel(prediction_file_path)

    # Check available columns
    print("Columns in the file:", prediction_df.columns)

    # Step 2: Preprocessing - Convert columns to lowercase
    try:
        prediction_df = prediction_df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)
    except Exception as e:
        raise KeyError(f"Error during preprocessing: {e}")

    # Ensure the required columns are present
    required_columns = ['Transcript', 'Resume', 'Reason for decision', 'Job Description']
    missing_columns = [col for col in required_columns if col not in prediction_df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the input file: {missing_columns}")

    # Ensure 'num_words_in_transcript' is calculated if missing
    if 'num_words_in_transcript' not in prediction_df.columns:
        prediction_df['num_words_in_transcript'] = prediction_df['Transcript'].apply(lambda x: len(str(x).split()))

    # Step 3: Transform textual columns using the vectorizer
    try:
        job_desc_vectors = loaded_vectorizer.transform(prediction_df['Job Description'])
        resume_vectors = loaded_vectorizer.transform(prediction_df['Resume'])
        transcript_vectors = loaded_vectorizer.transform(prediction_df['Transcript'])
    except Exception as e:
        raise ValueError(f"Error during TF-IDF transformation: {e}")

    # Step 4: Feature Engineering
    # Word counts
    prediction_df['transcript_word_count'] = prediction_df['Transcript'].apply(lambda x: len(str(x).split()))
    prediction_df['resume_word_count'] = prediction_df['Resume'].apply(lambda x: len(str(x).split()))
    prediction_df['resume_word_count'] = prediction_df['resume_word_count'].replace(0, np.nan).fillna(1)
    prediction_df['word_count_ratio'] = prediction_df['transcript_word_count'] / prediction_df['resume_word_count']

    # Sentiment analysis
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    prediction_df['transcript_sentiment'] = prediction_df['Transcript'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

    # Cosine similarity
    prediction_df['resume_job_similarity'] = [
        cosine_similarity(resume_vectors[i], job_desc_vectors[i])[0][0] for i in range(len(prediction_df))
    ]
    prediction_df['transcript_job_similarity'] = [
        cosine_similarity(transcript_vectors[i], job_desc_vectors[i])[0][0] for i in range(len(prediction_df))
    ]

    # Binning similarities
    bins = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    prediction_df['similarity_bin'] = pd.cut(prediction_df['resume_job_similarity'], bins=bins, labels=labels, include_lowest=True)

    # Sentiment alignment
    def calculate_sentiment_alignment(row):
        candidate_sentiment = row.get('transcript_sentiment', 0)
        interviewer_sentiment = row.get('interviewer_sentiment', 0)
        alignment = 1 - abs(candidate_sentiment - interviewer_sentiment)
        return alignment

    prediction_df['sentiment_alignment_score'] = prediction_df.apply(calculate_sentiment_alignment, axis=1)

    # Self-advocacy score
    advocacy_keywords = [
        r'\bachieved\b', r'\bled\b', r'\bdeveloped\b', r'\bimproved\b',
        r'\bmanaged\b', r'\binitiated\b', r'\borganized\b', r'\bimplemented\b',
        r'\bdesigned\b', r'\bcreated\b', r'\bexecuted\b', r'\bstrength\b',
        r'\bstrengths\b', r'\bachievement\b', r'\bachievements\b',
        r'\bcontributed\b', r'\binnovated\b', r'\btransformed\b'
    ]
    pattern = '|'.join(advocacy_keywords)

    def calculate_self_advocacy_score(transcript):
        matches = len(re.findall(pattern, str(transcript).lower()))
        total_words = len(str(transcript).split())
        return matches / total_words if total_words > 0 else 0

    prediction_df['self_advocacy_score'] = prediction_df['Transcript'].apply(calculate_self_advocacy_score)
    

    # Function to calculate overlap
    def calculate_overlap(transcript, resume):
        if pd.isnull(transcript) or pd.isnull(resume):
            return 0
        transcript_words = set(transcript.split())
        resume_words = set(resume.split())
        return len(transcript_words.intersection(resume_words))

    # Apply the function
    prediction_df['resume_transcript_overlap'] = prediction_df.apply(
        lambda row: calculate_overlap(row['Transcript'], row['Resume']), axis=1
    )
    
    

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
    prediction_df['resume_sentiment'] = prediction_df['Resume'].apply(calculate_vader_sentiment)
    


    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Function to extract interviewer comments (assuming they are prefixed with "interviewer")
    def interviewer_sentiment(text):
        interviewer_text = " ".join([line for line in text.splitlines() if line.lower().startswith("interviewer")])
        return sia.polarity_scores(interviewer_text)['compound']

    # Apply function to calculate sentiment
    prediction_df['interviewer_sentiment'] = prediction_df['Transcript'].apply(interviewer_sentiment)
    
 
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
    final_data_with_scores = apply_confidence_scores(prediction_df)
    print(final_data_with_scores)
    
    
    
    # Assuming 'transcript_sentiment' and 'resume_sentiment' columns exist
    prediction_df['sentiment_interaction'] = prediction_df['transcript_sentiment'] * prediction_df['resume_sentiment']




    # Interaction scores
    def calculate_interaction_score(resume, transcript):
        documents = [str(resume), str(transcript)]
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(documents)
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    prediction_df['interaction_score'] = prediction_df.apply(lambda row: calculate_interaction_score(row['Resume'], row['Transcript']), axis=1)

    # Keyword matching score
    def calculate_keyword_matching(row):
        job_keywords = set(row['Job Description'].split())
        transcript_words = set(row['Transcript'].split())
        overlap = len(job_keywords.intersection(transcript_words))
        return overlap / len(job_keywords) if len(job_keywords) > 0 else 0

    prediction_df['keyword_matching_score'] = prediction_df.apply(calculate_keyword_matching, axis=1)

    # Grouping results for analysis
    grouped_data = prediction_df.groupby(['similarity_bin', 'Reason for decision']).size().unstack(fill_value=0)

    return prediction_df, grouped_data


if __name__ == "__main__":
    # Load the vectorizer
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("Error: The file 'tfidf_vectorizer.pkl' was not found.")
        exit()

    prediction_file = 'prediction_data.xlsx'

    try:
        processed_df, grouped_data = preprocess_and_analyze_data(prediction_file, vectorizer)
        # Save the processed data to an Excel file
        processed_df.to_excel("processed_data.xlsx", index=False)
        print("Processing complete. Results saved to 'processed_data.xlsx'.")
        
        print()
        processed_df = pd.read_excel('processed_data.xlsx')
        print("Columns in processed_data.xlsx:", processed_df.columns)
    except Exception as e:
        print(f"Error: {e}")
