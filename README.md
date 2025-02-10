# AI-Driven Recruitment Pipeline

This project is a comprehensive solution designed to revolutionize recruitment processes using AI and data-driven insights. It integrates seamlessly to automate, optimize, and enhance every stage of hiring, from initial candidate screening to final selection.

# Presentation
This repository contains a PowerPoint presentation detailing the development of an AI-driven recruitment pipeline with real-time interview insights and cultural fit scoring. The project leverages Machine Learning and AI to automate and optimize the hiring process.
🎯 **View the presentation here:** [Canva Presentation](https://www.canva.com/design/DAGeQDHz53Q/3cVYWZGo-LdQSKC_bDJqfQ/edit?utm_content=DAGeQDHz53Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)  

## Key Features:
1. **AI-Powered Screening**: Leverage advanced natural language processing and machine learning models to analyze resumes, transcripts, and other applicant data for job relevance and skill matching.
2. **Real-Time Interview Insights**: Utilize sentiment analysis and keyword tracking to provide actionable feedback during interviews, ensuring a deeper understanding of candidate responses.
3. **Cultural Fit Scoring**: Evaluate candidates' alignment with organizational values using customized scoring algorithms, promoting better team integration and long-term success.
4. **Data Visualization**: Generate intuitive charts and heatmaps to correlate key metrics like resume-job similarity, transcript quality, and selection outcomes.
5. **Predictive Analytics**: Implement logistic regression and classification models to predict candidate success and improve decision-making.
6. **Role-Specific Analysis**: Tailored recommendations and insights for various roles, ensuring fairness and efficiency in the hiring process.

The pipeline includes:
- **Exploratory Data Analysis (EDA)**: Analyzing the data to understand relationships between different features.
- **Model Training**: Training a machine learning model (e.g., XGBoost) on preprocessed data to make predictions.
- **Resume Screening**: Preprocessing resumes and interview transcripts, extracting relevant features.
- **Prediction**: Using the trained model to predict decisions for new resumes and interview transcripts.
- **Emailing the Results**: Sending the prediction results via email to the specified recipient.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Scripts Description](#scripts-description)
    - [1. Exploratory Data Analysis (`exploratory_data_analysis.py`)](#1-exploratory-data-analysis-exploratory_data_analysispy)
    - [2. Model Training (`training.py`)](#2-model-training-trainingpy)
    - [3. Resume Screening (`resume_screener.py`)](#3-resume-screening-resume_screenerpy)
    - [4. Prediction (`prediction.py`)](#4-prediction-predictionpy)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/resume-screening.git
   cd resume-screening

## Dependencies

The following Python libraries are required to run the project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `xgboost`: For training machine learning models.
- `scikit-learn`: For model evaluation and preprocessing utilities.
- `matplotlib`, `seaborn`: For visualizations.
- `smtplib`, `email`: For sending emails with attachments.
- `openpyxl`: For reading and writing Excel files.
- `pickle`: For saving and loading trained models.

Make sure to install all the required libraries using `pip install -r requirements.txt`.


## Usage

### Preprocessing and Feature Extraction:
The `resume_screener.py` script preprocesses the resumes and interview transcripts, extracting relevant features.

### Training the Model:
Run the `training.py` script to train a machine learning model on the processed data. The model is saved using pickle and can be used for predictions later.
### Making Predictions:
The prediction.py script loads the trained model and processed data, makes predictions, and saves the results into an Excel file. It can also send the prediction results via email to the specified recipient.
### Exploratory Data Analysis:
Run the exploratory_data_analysis.py script to explore and visualize the dataset. This script helps in understanding the relationships between different features and prepares the data for training.

## Scripts Description

### 1. Exploratory Data Analysis (`exploratory_data_analysis.py`)
This script is used to perform an initial analysis of the dataset. The tasks include:
- Loading and cleaning the data.
- Visualizing distributions of key features.
- Identifying correlations between features.
- Detecting any patterns or insights that might help in building the model.

### 2. Model Training (`training.py`)
The `training.py` script is responsible for:
- Splitting the dataset into training and testing sets.
- Training a machine learning model (e.g., XGBoost) using the features extracted from the resumes and interview transcripts.
- Evaluating the model performance using metrics like accuracy, precision, recall, and F1-score.
- Saving the trained model to a file using pickle.

### 3. Resume Screening (`resume_screener.py`)
This script preprocesses the resumes and interview transcripts to extract features such as:
- Word count.
- Sentiment scores.
- Keyword matching scores.
- Job description matching scores.
- Interaction scores.

These features are then saved to a file for model training and prediction.

### 4. Prediction (`prediction.py`)
This script loads the trained model and makes predictions on new data (processed resumes and transcripts). The predictions are saved to an output Excel file and can also be sent via email. The workflow includes:
- Loading the processed data and trained model.
- Making predictions using the trained model.
- Saving the predictions to an output Excel file.
- Sending an email with the results as an attachment.

## File Structure

```
AI-DRIVEN-RECRUITMENT-PIPELINE/
│
├── Approved_Datasets/            # Raw approved datasets
│   ├── dataset2.xlsx
│   ├── dataset3.xlsx
│   ├── ...
│   └── dataset9.xlsx
│
├── backup/                       # Backup folder for interim files
│
├── Generated_Data/               # Processed and intermediate data
│   ├── cleaned_data.csv
│   ├── featured_final_data.csv
│   └── processed_data.xlsx
│
├── models/                       # Saved machine learning models
│   ├── tfidf_vectorizer.pkl
│   └── xgboost_model.pkl
│
├── output/                       # Prediction outputs
│   ├── prediction_results.xlsx
│
├── Prediction_Data/              # Data used for predictions
│   └── prediction_data.xlsx
│
├── Exploratory_Data_Analysis.py  # Data analysis and visualization
├── prediction.py                 # Model prediction script
├── resume_screener.py            # Resume screening logic
└── training.py                   # Model training pipeline
│
├── Exploratory_Data_Analysis.ipynb  # Jupyter Notebook for EDA
├── LICENSE                          # License for the repository
└── README.md                        # Project documentation

```

## Contributing
If you'd like to contribute to this project, feel free to fork the repository, create a branch, and submit a pull request. You can also submit issues if you find bugs or have suggestions for improvements.

### Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes.
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request.
## License
This project is open-source and available under the MIT License.
