import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'pre_match_features.csv')

print("Loading final feature dataset...")
df = pd.read_csv(data_path)

# Smart Feature Transformation
df['team1_won_toss'] = (df['toss_winner'] == df['team_1']).astype(int)

cols_to_drop = ['date', 'toss_winner', 'winner']
df = df.drop(columns=cols_to_drop)

# Separate Features and Target
y = df['target']
X = df.drop(columns=['target'])

# One-Hot Encoding (This will now finally encode the VENUES!)
print("Encoding text features (Teams & Venues) into numerical matrix...")
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"Total features after encoding: {X_train.shape[1]} (Should be higher now because of venues!)")

print("\nTraining XGBoost model...")
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Save the Model and the full Column Structure
saved_models_dir = os.path.join(current_dir, 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

joblib.dump(model, os.path.join(saved_models_dir, 'xgboost_pre_match.joblib'))
joblib.dump(list(X_encoded.columns), os.path.join(saved_models_dir, 'model_columns.joblib'))

print(f"\nSuccess! Model saved.")