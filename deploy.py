
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import nltk, re, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import spacy.cli
# Download the model if not already installed
spacy.cli.download("en_core_web_sm")

# Load the model
nlp = spacy.load("en_core_web_sm")


# ============================
# Load NLP resources
# ============================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# Helper functions
# ============================
def lexical_preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w not in string.punctuation]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

def semantic_features(text):
    blob = TextBlob(text)
    return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"

def discourse_features(text):
    sentences = nltk.sent_tokenize(text)
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

def pragmatic_features(text):
    tokens = []
    for w in pragmatic_words:
        count = text.lower().count(w)
        tokens.extend([w] * count)
    return " ".join(tokens)

# ============================
# Load or Train Models
# ============================
# Replace these with actual trained vectorizers and models using pickle
# ============================
# Load trained vectorizers & models
# ============================
import pickle

# Load vectorizers
vec_lexical = pickle.load(open("vec_lexical.pkl","rb"))
vec_syntax = pickle.load(open("vec_syntax.pkl","rb"))
vec_semantic = pickle.load(open("vec_semantic.pkl","rb"))
vec_discourse = pickle.load(open("vec_discourse.pkl","rb"))
vec_pragmatic = pickle.load(open("vec_pragmatic.pkl","rb"))

# Load models
models_dict = {
    "Naive Bayes": pickle.load(open("NaiveBayes.pkl","rb")),
    "SVM": pickle.load(open("SVM.pkl","rb")),
    "Logistic Regression": pickle.load(open("LogisticRegression.pkl","rb")),
    "Decision Tree": pickle.load(open("DecisionTree.pkl","rb"))
}


feature_dict = {
    "Lexical": vec_lexical,
    "Syntactic": vec_syntax,
    "Semantic": vec_semantic,
    "Discourse": vec_discourse,
    "Pragmatic": vec_pragmatic
}

feature_funcs = {
    "Lexical": lexical_preprocess,
    "Syntactic": syntactic_features,
    "Semantic": semantic_features,
    "Discourse": discourse_features,
    "Pragmatic": pragmatic_features
}

# ============================
# Streamlit UI
# ============================
st.title("📰 Fake vs Real Statement Detection")
st.write("Enter a statement and select a model to predict if it is fake or real, along with phase-wise contributions.")

# User input
user_input = st.text_area("Enter Statement:")
selected_model = st.selectbox("Select Model", list(models_dict.keys()))

if st.button("Predict") and user_input.strip() != "":
    phase_preds = {}

    # Compute phase-wise features for visualization (not for model prediction)
    for phase, func in feature_funcs.items():
        feat_text = func(user_input)
        # Count-based "confidence" proxy (you can customize)
        phase_preds[phase] = {"Feature Preview": feat_text, "Confidence": len(feat_text.split())}

    # ----------------------------
    # Combined features for model prediction
    # ----------------------------
    X_combined = hstack([
        vec_lexical.transform([lexical_preprocess(user_input)]),
        vec_syntax.transform([syntactic_features(user_input)]),
        vec_semantic.transform([semantic_features(user_input)]),
        vec_discourse.transform([discourse_features(user_input)]),
        vec_pragmatic.transform([pragmatic_features(user_input)])
    ])

    model = models_dict[selected_model]
    overall_pred = model.predict(X_combined)[0]
    overall_prob = model.predict_proba(X_combined).max() if hasattr(model, "predict_proba") else None

    # ----------------------------
    # Display Results
    # ----------------------------
    st.subheader("✅ Overall Prediction")
    st.write(f"**{overall_pred}**")
    if overall_prob:
        st.write(f"Confidence: {overall_prob:.2f}")

    # Phase-wise visualization (using proxy confidence)
    st.subheader("📊 Phase-wise Feature Overview")
    phase_df = pd.DataFrame.from_dict(phase_preds, orient="index").reset_index()
    phase_df.rename(columns={"index":"Phase"}, inplace=True)
    st.dataframe(phase_df)

    # Bar chart of confidence per phase
    plt.figure(figsize=(8,5))
    sns.barplot(x="Phase", y="Confidence", data=phase_df, palette="viridis")
    plt.title("Phase-wise Feature Length (proxy confidence)")
    st.pyplot(plt)

    # ----------------------------
    # Display Results
    # ----------------------------
    st.subheader("✅ Overall Prediction")
    st.write(f"**{overall_pred}**")
    if overall_prob:
        st.write(f"Confidence: {overall_prob:.2f}")

    # Phase-wise visualization
    st.subheader("📊 Phase-wise Predictions & Confidence")
    phase_df = pd.DataFrame.from_dict(phase_preds, orient="index")
    phase_df.reset_index(inplace=True)
    phase_df.rename(columns={"index":"Phase"}, inplace=True)
    st.dataframe(phase_df)

    # Bar chart of confidence per phase
    plt.figure(figsize=(8,5))
    sns.barplot(x="Phase", y="Confidence", data=phase_df, palette="viridis")
    plt.ylim(0,1)
    plt.title("Phase-wise Confidence Scores")
    st.pyplot(plt)


