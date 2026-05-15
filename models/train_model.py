import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'pre_match_features.csv')

print("Loading V2.0 feature dataset...")
df = pd.read_csv(data_path)

is_toss_winner = (df['toss_winner'] == df['team_1'])
chose_to_bat = (df['toss_decision'] == 'bat')

df['team1_batting_first'] = ((is_toss_winner & chose_to_bat) | (~is_toss_winner & ~chose_to_bat)).astype(int)

cols_to_drop = ['date', 'toss_winner', 'toss_decision', 'winner', 'bat_first_won', 'venue']
df = df.drop(columns=cols_to_drop)

y = df['target']
X = df.drop(columns=['target'])

print("Encoding team names into numerical matrix...")
X_encoded = pd.get_dummies(X, drop_first=True)

split_index = int(len(X_encoded) * 0.8)

X_train = X_encoded.iloc[:split_index]
X_test = X_encoded.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Training on historical matches: {X_train.shape[0]}")
print(f"Testing on future matches: {X_test.shape[0]}")
print(f"Total pure mathematical features: {X_train.shape[1]}")

print("\nTraining V2.0 XGBoost model...")
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== V2.0 MODEL EVALUATION ===")
print(f"Accuracy on Chronological Future Data: {accuracy * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

saved_models_dir = os.path.join(current_dir, 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

joblib.dump(model, os.path.join(saved_models_dir, 'xgboost_pre_match.joblib'))
joblib.dump(list(X_encoded.columns), os.path.join(saved_models_dir, 'model_columns.joblib'))

print(f"\nSuccess! V2.0 Model saved.")