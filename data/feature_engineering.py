import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'cleaned_base_data.csv')
df = pd.read_csv(file_path)

df['date'] = pd.to_datetime(df['date'])

def get_recent_wins(team, current_date, n=6):
    past_matches = df[(df['date'] < current_date) & 
                      ((df['team_1'] == team) | (df['team_2'] == team))]
    return (past_matches.tail(n)['winner'] == team).sum()

def get_h2h_wins(team_a, team_b, current_date, n=10):
    past_matches = df[(df['date'] < current_date) & 
                      (((df['team_1'] == team_a) & (df['team_2'] == team_b)) | 
                       ((df['team_1'] == team_b) & (df['team_2'] == team_a)))]
    return (past_matches.tail(n)['winner'] == team_a).sum()

print("Calculating recent form and H2H stats with the new dataset...")

t1_last6, t2_last6, t1_h2h, t2_h2h = [], [], [], []

for index, row in df.iterrows():
    date, t1, t2 = row['date'], row['team_1'], row['team_2']
    t1_last6.append(get_recent_wins(t1, date, n=6))
    t2_last6.append(get_recent_wins(t2, date, n=6))
    t1_h2h.append(get_h2h_wins(t1, t2, date, n=10))
    t2_h2h.append(get_h2h_wins(t2, t1, date, n=10))

df['team1_last6_wins'] = t1_last6
df['team2_last6_wins'] = t2_last6
df['team1_h2h_last10'] = t1_h2h
df['team2_h2h_last10'] = t2_h2h

output_path = os.path.join(current_dir, 'pre_match_features.csv')
df.to_csv(output_path, index=False)
print(f"Success! Saved final feature dataset to: {output_path}")