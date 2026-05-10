import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
raw_file_path = os.path.join(current_dir, 'matchwise_data.csv')

print(f"Loading raw data from: {raw_file_path}")
df = pd.read_csv(raw_file_path)

df.columns = df.columns.str.lower().str.strip()

if 'ground_name' in df.columns: 
    df.rename(columns={'ground_name': 'venue'}, inplace=True)
elif 'ground_city' in df.columns:
    df.rename(columns={'ground_city': 'venue'}, inplace=True)

string_cols = df.select_dtypes(include=['object']).columns
for col in string_cols:
    df[col] = df[col].astype(str).str.strip()

columns_to_keep = ['date', 'team_1', 'team_2', 'venue', 'toss_winner', 'toss_decision', 'winner']
existing_columns = [col for col in columns_to_keep if col in df.columns]
df = df[existing_columns]

df = df.dropna(subset=['winner', 'venue'])

invalid_results = ['tie', 'tied', 'no result', 'abandoned', 'cancelled', 'nan']
df = df[~df['winner'].str.lower().isin(invalid_results)]

df['target'] = (df['team_1'] == df['winner']).astype(int)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce') 
    df = df.dropna(subset=['date']) 
    df = df.sort_values(by='date').reset_index(drop=True)

print("\n=== THE NEW BULLETPROOF DATA SHAPE ===")
print(f"Total Matches: {df.shape[0]}")
print(f"Columns Found: {df.columns.tolist()}")

output_path = os.path.join(current_dir, 'cleaned_base_data.csv')
df.to_csv(output_path, index=False)
print(f"\nSuccess! Saved clean data with Venues to: {output_path}")