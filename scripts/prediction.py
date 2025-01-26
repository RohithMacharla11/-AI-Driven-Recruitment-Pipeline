import pandas as pd
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def load_model_and_predict(processed_data_path, model_path):
    # Load processed data
    processed_df = pd.read_excel(processed_data_path)

    # Step 1: Load the trained model
    with open(model_path, 'rb') as model_file:
        xgb_model = pickle.load(model_file)

    # Get the feature names used during training
    trained_features = xgb_model.get_booster().feature_names

    # Step 2: Ensure we use the correct set of features for prediction
    available_features = [feature for feature in trained_features if feature in processed_df.columns]

    if len(available_features) != len(trained_features):
        print(f"Warning: Some features are missing: {set(trained_features) - set(available_features)}")

    # Perform prediction with the available features
    processed_df['predicted_decision'] = xgb_model.predict(processed_df[available_features])

    return processed_df

def send_email_with_attachment(to_email, subject, body, file_path):
    sender_email = "macharlarohith111@gmai.com"
    sender_password = "Rohith@999r"  # Use an app password

    # Create MIME structure
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Attach the file
    try:
        with open(file_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
        message.attach(part)
    except Exception as e:
        print(f"Error attaching file: {e}")
        return

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
            print("Email sent successfully with attachment!")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    processed_data_file = "Generated_Data/processed_data.xlsx"
    model_file = 'models/xgboost_model.pkl'

    # Step 1: Run predictions
    predicted_df = load_model_and_predict(processed_data_file, model_file)

    # Save final results to an Excel file
    output_file = "output/prediction_results.xlsx"
    predicted_df.to_excel(output_file, index=False)

    # Step 2: Email the results
    send_email_with_attachment(
        to_email="2203a52157@sru.edu.in",
        subject="Prediction Results",
        body="Please find the attached prediction results.",
        file_path=output_file
    )
