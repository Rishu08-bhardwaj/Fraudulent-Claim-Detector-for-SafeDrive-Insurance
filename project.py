import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 500

# Create mock data
data = {
    "policy_number": np.random.randint(100000, 999999, n_samples),
    "claim_amount": np.round(np.random.uniform(500, 15000, n_samples), 2),
    "incident_type": np.random.choice(["Collision", "Theft", "Fire", "Vandalism"], n_samples),
    "incident_severity": np.random.choice(["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"], n_samples),
    "insured_age": np.random.randint(18, 80, n_samples),
    "vehicle_age": np.random.randint(0, 20, n_samples),
    "insured_education_level": np.random.choice(["High School", "Bachelor", "Masters", "PhD"], n_samples),
    "number_of_past_claims": np.random.randint(0, 5, n_samples),
    "late_payments": np.random.randint(0, 3, n_samples),
    "fraud_reported": np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% fraud
}

# Create DataFrame and save to CSV
df_sample = pd.DataFrame(data)
df_sample.to_csv("insurance_claims.csv", index=False)
print("Sample dataset saved as 'insurance_claims.csv'")
# -----------------------------

# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  # Optional for class imbalance

import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# -----------------------------
# 2. Load and Inspect Dataset
# -----------------------------
# Replace with your actual file path
df = pd.read_csv("insurance_claims.csv")

print("Data Shape:", df.shape)
print("Column Names:", df.columns.tolist())
print(df.head())

# -----------------------------
# 3. Data Preprocessing
# -----------------------------
# Drop irrelevant columns (e.g., claim_id, customer_id if not useful)
drop_cols = ['claim_id', 'customer_id']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Label encoding or one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate target
X = df.drop('fraud_reported', axis=1)  # Target variable should be 'fraud_reported' or similar
y = df['fraud_reported'].map({'Y': 1, 'N': 0}) if df['fraud_reported'].dtype == 'object' else df['fraud_reported']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Optional: Handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------
# 5. Model Training
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# -----------------------------
# 7. Flag Potential Fraud Claims
# -----------------------------
X_test_with_preds = X_test.copy()
X_test_with_preds['fraud_probability'] = y_proba
X_test_with_preds['flagged'] = X_test_with_preds['fraud_probability'] > 0.5

# Export flagged claims for review
X_test_with_preds.to_csv("flagged_claims.csv", index=False)
print("Exported flagged claims to 'flagged_claims.csv'.")

# -----------------------------
# 8. Optional: Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_imp_df.head(15), x='Importance', y='Feature')
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()




# --- Load and preprocess data ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("insurance_claims.csv")

# Drop non-useful columns
df.drop(columns=['policy_number'], inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=[
    'incident_type', 'incident_severity', 'insured_education_level'
], drop_first=True)

# Define features and target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Apply SMOTE ---
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔥 ROC-AUC Score:", roc_auc_score(y_test, y_proba))



import matplotlib.pyplot as plt
import seaborn as sns

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()




# Create DataFrame from test set
df_test = pd.DataFrame(X_test_scaled, columns=X.columns)
df_test['fraud_probability'] = y_proba
df_test['flagged'] = df_test['fraud_probability'] > 0.4  # Adjust threshold

# Show top 10 suspicious claims
top_suspects = df_test.sort_values(by='fraud_probability', ascending=False).head(10)
print("\n🚨 Top 10 Most Suspicious Claims:\n")
print(top_suspects)

# Export full flagged dataset
df_test.to_csv("flagged_claims_improved.csv", index=False)
print("\n✅ Flagged claims exported to 'flagged_claims_improved.csv'")
