import pickle
import pandas as pd
import nltk, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv("merged_final.csv")
X = df['statement']
y = df['BinaryTarget']

# NLP tools
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ============================
# Phase functions
# ============================
def lexical_preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w not in string.punctuation]
    return " ".join(tokens)

def syntactic_features(text):
    doc = nlp(text)
    return " ".join([token.pos_ for token in doc])

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

feature_funcs = {
    "Lexical": lexical_preprocess,
    "Syntactic": syntactic_features,
    "Semantic": semantic_features,
    "Discourse": discourse_features,
    "Pragmatic": pragmatic_features
}

# ============================
# Vectorizers
# ============================
vec_lexical = CountVectorizer().fit(X.apply(lexical_preprocess))
vec_syntax = CountVectorizer().fit(X.apply(syntactic_features))
vec_semantic = TfidfVectorizer().fit(X.apply(semantic_features))
vec_discourse = CountVectorizer().fit(X.apply(discourse_features))
vec_pragmatic = CountVectorizer().fit(X.apply(pragmatic_features))

# Combine all features
X_combined = hstack([
    vec_lexical.transform(X.apply(lexical_preprocess)),
    vec_syntax.transform(X.apply(syntactic_features)),
    vec_semantic.transform(X.apply(semantic_features)),
    vec_discourse.transform(X.apply(discourse_features)),
    vec_pragmatic.transform(X.apply(pragmatic_features))
])

# ============================
# Train models
# ============================
models = {
    "NaiveBayes": MultinomialNB(),
    "SVM": SVC(kernel="linear", probability=True),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_combined, y)
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save vectorizers
pickle.dump(vec_lexical, open("vec_lexical.pkl", "wb"))
pickle.dump(vec_syntax, open("vec_syntax.pkl", "wb"))
pickle.dump(vec_semantic, open("vec_semantic.pkl", "wb"))
pickle.dump(vec_discourse, open("vec_discourse.pkl", "wb"))
pickle.dump(vec_pragmatic, open("vec_pragmatic.pkl", "wb"))
