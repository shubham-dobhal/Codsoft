import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

FILE_NAME = r"C:\Users\dobha\OneDrive\Desktop\codsoft\spam.csv" 

try:
    df = pd.read_csv(FILE_NAME, encoding='latin-1') 
    
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Distribution of Ham vs Spam Messages')
    plt.show()

except FileNotFoundError:
    print(f"Error: '{FILE_NAME}' file not found. Check the file path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}. Check if the file is closed and column names ('v1', 'v2') are correct.")
    exit()
    
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower() 
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['cleaned_message'] = df['message'].apply(clean_text)

df = df.dropna(subset=['label_encoded'])
X = df['cleaned_message']
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Dataset split completed. Training features shape: {X_train_tfidf.shape}")

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

lr_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

print("\n--- MODEL PERFORMANCE SUMMARY ---")
print(f"1. Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
print(f"2. Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")

def report_model_performance(y_true, y_pred, model_name):
    print(f"\n--- {model_name} CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=['HAM (0)', 'SPAM (1)']))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['HAM', 'SPAM'], yticklabels=['HAM', 'SPAM'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

report_model_performance(y_test, nb_pred, "Naive Bayes")
report_model_performance(y_test, lr_pred, "Logistic Regression")

def final_prediction_test(sms_list, model, vectorizer):
    cleaned_sms = [clean_text(sms) for sms in sms_list]
    tfidf_sms = vectorizer.transform(cleaned_sms)
    predictions = model.predict(tfidf_sms)
    
    print("\n--- Final Prediction Test ---")
    for sms, pred in zip(sms_list, predictions):
        result = "SPAM" if pred == 1 else "HAM"
        print(f"Message: '{sms[:45]}...' -> Predicted: {result}")

sample_sms = [
    "URGENT! You have won a free prize. Claim link: http://award.com", 
    "Hey mom, dinner at 7pm tonight?",
    "Congratulations! You are selected for a cash offer of $5000."
]

final_prediction_test(sample_sms, lr_model, tfidf_vectorizer)