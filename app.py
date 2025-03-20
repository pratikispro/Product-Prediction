from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import json
import pandas as pd
import gzip
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
# Initialize Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from any frontend


# Load Data
def load_data(gz_file):
    data = []
    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

df = load_data("dataset.json.gz")

# Load vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Define Capsule BiLSTM Model
class CapsuleBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_capsules=10, capsule_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.capsule_layer = nn.Linear(hidden_dim * 2, num_capsules * capsule_dim)
        self.fc = nn.Linear(num_capsules * capsule_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        capsules = self.capsule_layer(lstm_out[:, -1, :])  # Apply capsule transformation
        output = self.fc(capsules)
        return self.sigmoid(output)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsuleBiLSTM(vocab_size=len(vocab), embed_dim=128, hidden_dim=256, output_dim=1).to(device)
model.load_state_dict(torch.load("capsule_bilstm_model.pth", map_location=device))
model.eval()

# Function to fetch product reviews
def fetch_reviews_from_file(product_id, df):
    reviews = df[df['asin'] == product_id]['text'].dropna().tolist()
    return reviews
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Extract key features from reviews
def extract_features(reviews, top_n=5):
    if not reviews:
        return {}

    # Convert reviews into TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(reviews)
    
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(X.toarray().sum(axis=0))[-top_n:]  # Top N features
    
    top_features = feature_array[tfidf_sorting].tolist()
    
    return top_features

# Function to predict product trustworthiness
def predict_product(product_id):
    reviews = fetch_reviews_from_file(product_id, df)
    if not reviews:
        return {"message": "No reviews found for this product."}

    scores = []
    feature_scores = {feature: [] for feature in extract_features(reviews)}

    for review in reviews:
        tokens = word_tokenize(review.lower())
        encoded_review = [vocab.get(word, 1) for word in tokens]
        if not encoded_review:
            continue  # Skip empty reviews
        encoded_review = torch.tensor(encoded_review, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(encoded_review).item()
        scores.append(score)

        # Assign scores to features if they appear in the review
        for feature in feature_scores.keys():
            if feature in review.lower():
                feature_scores[feature].append(score)

    if not scores:
        return {"message": "All reviews contained unknown words. Unable to classify."}

    avg_score = sum(scores) / len(scores)
    label = "Good Product" if avg_score > 0.5 else "Not a Good Product"

    # Compute feature-wise average rating (1-10 scale)
    feature_ratings = {
        feature: round((sum(scores) / len(scores)) * 10, 1) if scores else "No data"
        for feature, scores in feature_scores.items()
    }

    return {
        "product_id": product_id,
        "trustworthiness_score": round(avg_score, 4),
        "label": label,
        "feature_ratings": feature_ratings
    }

# API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    product_id = data.get("product_id")

    if not product_id:
        return jsonify({"error": "Missing product_id"}), 400

    result = predict_product(product_id)
    return jsonify(result)

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
