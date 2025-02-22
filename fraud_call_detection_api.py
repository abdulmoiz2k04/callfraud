import os
import re
import joblib
import whisper
import numpy as np
import nltk
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load trained scam detection model
try:
    model = joblib.load(r"D:\cybersentinels\Call Fraud\scam_model.pkl")
    vectorizer = joblib.load(r"D:\cybersentinels\Call Fraud\vectorizer.pkl")
except:
    raise FileNotFoundError("❌ Trained model not found. Run 'train_scam_model.py' first.")

# Predefined scam keywords (NLP-based intent analysis)
SCAM_KEYWORDS = [
    "send money", "urgent payment", "bank verification", "account compromised",
    "IRS payment", "gift card payment", "wire transfer", "bitcoin payment",
    "act now", "fraud alert", "reset password"
]

# Initialize Flask app
app = Flask(__name__)

# Load Whisper Model
whisper_model = whisper.load_model("base")

# Function to transcribe audio using Whisper AI
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Function to preprocess text (removing stopwords, special characters)
def preprocess_text(text):
    nltk.download("stopwords")
    nltk.download("punkt")
    
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    return " ".join(tokens)

# Function to detect scam patterns using AI model & NLP
def detect_scam(transcript):
    # Rule-based keyword detection
    for keyword in SCAM_KEYWORDS:
        if keyword in transcript.lower():
            return True, f"Detected scam phrase: '{keyword}'"

    # AI-based text analysis
    cleaned_text = preprocess_text(transcript)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        return True, "AI model detected scam patterns."

    return False, "No scam detected."

# API Route to Detect Fraud Calls
@app.route('/detect_fraud_call', methods=['POST'])
def detect_fraud_call():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files["audio"]
    file_path = "temp_audio.wav"
    audio_file.save(file_path)

    # Transcribe audio
    transcript = transcribe_audio(file_path)
    
    if not transcript:
        return jsonify({"error": "Could not transcribe audio"}), 500

    # Analyze for fraud
    fraud_detected, reason = detect_scam(transcript)

    return jsonify({
        "fraud_detected": fraud_detected,
        "transcript": transcript,
        "reason": reason
    })

# Run Flask API
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
