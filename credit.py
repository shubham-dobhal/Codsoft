 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


print("🔹 Loading datasets...")
train_df = pd.read_csv(r"archive/fraudTrain.csv")
test_df = pd.read_csv(r"archive/fraudTest.csv")


data = pd.concat([train_df, test_df], axis=0)


data = data.sample(n=100000, random_state=42)  # 1 lakh rows for fast test
print(f"✅ Subset Dataset Shape: {data.shape}")


drop_cols = [
    'Unnamed: 0', 'trans_date_trans_time', 'first', 'last',
    'street', 'city', 'state', 'zip', 'dob', 'merchant'
]
data = data.drop(columns=[col for col in drop_cols if col in data.columns])


categorical_cols = data.select_dtypes(include='object').columns
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col].astype(str))


X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n✅ Training Data: {X_train.shape}, Testing Data: {X_test.shape}")


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {X_train_res.shape}, Frauds: {sum(y_train_res==1)}, Legit: {sum(y_train_res==0)}")


print("\n🔹 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  # faster
rf.fit(X_train_res, y_train_res)


y_pred_rf = rf.predict(X_test)


acc = accuracy_score(y_test, y_pred_rf)
print(f"\n✅ Accuracy (Random Forest): {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))


fraud_detected = np.sum((y_pred_rf == 1) & (y_test == 1))
total_frauds = np.sum(y_test == 1)

print("\n==============================")
print("🔍 FRAUD DETECTION SUMMARY")
print("==============================")
print(f"Total Frauds in Test Set: {total_frauds}")
print(f"Frauds Correctly Detected: {fraud_detected}")
print(f"Detection Rate: {(fraud_detected/total_frauds)*100:.2f}%")
print("==============================")


sample_indices = np.random.choice(len(X_test), 10, replace=False)
print("\n🔹 Sample Predictions (0 = Legit, 1 = Fraud):")
for idx in sample_indices:
    print(f"Transaction {idx}: Predicted = {y_pred_rf[idx]}, Actual = {y_test.iloc[idx]}")
