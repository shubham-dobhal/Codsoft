
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os


dataset_folder = "movie_dataset" 

train_file = os.path.join(dataset_folder, "train_data.txt")
test_file = os.path.join(dataset_folder, "test_data.txt")
test_solution_file = os.path.join(dataset_folder, "test_data_solution.txt")
desc_file = os.path.join(dataset_folder, "description.txt")

print("📁 Dataset folder path:", os.path.abspath(dataset_folder))


for f in [train_file, test_file, test_solution_file, desc_file]:
 if not os.path.exists(f):
  print(f"⚠️ File not found: {f}")


def load_train_data(file_path):
  df = pd.read_csv(file_path, sep=":::", header=None, engine='python')
  df.columns = ['id', 'movie_name', 'genre', 'description']
  df['description'] = df['description'].astype(str).str.strip()
  df['genre'] = df['genre'].astype(str).str.strip()
  return df

def load_test_data(file_path):
 df = pd.read_csv(file_path, sep=":::", header=None, engine='python')
 df.columns = ['id', 'movie_name', 'description']
 df['description'] = df['description'].astype(str).str.strip()
 return df

def load_test_solution(file_path):
    # ERROR FIX: Assigning 4 column names first and then selecting only 'id' and 'genre'
    df = pd.read_csv(file_path, sep=":::", header=None, engine='python')
    df.columns = ['id', 'movie_name', 'genre', 'description'] # The file likely has 4 columns
    df = df[['id', 'genre']] # Select only the necessary columns
    df['genre'] = df['genre'].astype(str).str.strip()
    return df



try:
    train_df = load_train_data(train_file)
    test_df = load_test_data(test_file)
    test_sol_df = load_test_solution(test_solution_file)

    print("\n✅ Data Loaded Successfully!")
    print("Training samples before cleaning:", len(train_df))
    print("Testing samples before cleaning:", len(test_df))

    
    train_df = train_df.dropna(subset=['description', 'genre'])
    test_df = test_df.dropna(subset=['description'])
    test_sol_df = test_sol_df.dropna(subset=['genre'])

    print("Training samples after cleaning:", len(train_df))
    print("Testing samples after cleaning:", len(test_df))

    
    test_df = test_df[test_df['id'].isin(test_sol_df['id'])]
    test_sol_df = test_sol_df[test_sol_df['id'].isin(test_df['id'])]
    
   
    test_sol_df = test_sol_df.set_index('id').loc[test_df['id']].reset_index()

    X_train = train_df['description']
    y_train = train_df['genre']

    X_test = test_df['description']
    y_test = test_sol_df['genre']

    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    
    y_pred = model.predict(X_test_tfidf)

    print("\n🎯 MODEL PERFORMANCE:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

   
    sample_text = "A superhero saves the world from an alien invasion."
    sample_vec = vectorizer.transform([sample_text])
    sample_pred = model.predict(sample_vec)[0]

    print("\n🎬 Example Prediction:")
    print(f"Description: {sample_text}")
    print(f"Predicted Genre: {sample_pred}")

    print("\n✅ Movie Genre Classification Complete!")

except FileNotFoundError as e:
    print(f"\n❌ Error: One or more data files were not found. Please check the paths. {e}")
except ValueError as e:
    print(f"\n❌ A data loading or processing error occurred. Please check the structure of your data files. {e}")