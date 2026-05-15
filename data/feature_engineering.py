import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'cleaned_base_data.csv')
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

bat_first_won = []
for _, row in df.iterrows():
    if row['toss_decision'] == 'bat':
        bat_first_won.append(1 if row['winner'] == row['toss_winner'] else 0)
    else:
        bat_first_won.append(1 if row['winner'] != row['toss_winner'] else 0)
df['bat_first_won'] = bat_first_won

def get_recent_wins(team, current_date, n=6):
    past = df[(df['date'] < current_date) & ((df['team_1'] == team) | (df['team_2'] == team))]
    return (past.tail(n)['winner'] == team).sum()

def get_h2h_wins(team_a, team_b, current_date, n=10):
    past = df[(df['date'] < current_date) & 
              (((df['team_1'] == team_a) & (df['team_2'] == team_b)) | 
               ((df['team_1'] == team_b) & (df['team_2'] == team_a)))]
    return (past.tail(n)['winner'] == team_a).sum()

def get_venue_rate(venue, current_date):
    past = df[(df['date'] < current_date) & (df['venue'] == venue)]
    if len(past) < 5: return 0.50
    return past['bat_first_won'].mean()

def get_global_win_rate(team, current_date):
    past = df[(df['date'] < current_date) & ((df['team_1'] == team) | (df['team_2'] == team))]
    if len(past) < 5: return 0.50 
    return (past['winner'] == team).sum() / len(past)

print("Calculating momentum, venue dynamics, and Advanced Elo Ratings...")

t1_last6, t2_last6, t1_h2h, t2_h2h, v_rates, t1_elo_list, t2_elo_list = [], [], [], [], [], [], []

elo_ratings = {}
K_FACTOR = 40 

def get_elo(team):
    return elo_ratings.get(team, 1000)

for index, row in df.iterrows():
    date, t1, t2, venue, winner = row['date'], row['team_1'], row['team_2'], row['venue'], row['winner']
    
    t1_last6.append(get_recent_wins(t1, date, n=6))
    t2_last6.append(get_recent_wins(t2, date, n=6))
    t1_h2h.append(get_h2h_wins(t1, t2, date, n=10))
    t2_h2h.append(get_h2h_wins(t2, t1, date, n=10))
    v_rates.append(get_venue_rate(venue, date))
    
    elo1 = get_elo(t1)
    elo2 = get_elo(t2)
    t1_elo_list.append(elo1)
    t2_elo_list.append(elo2)

    expected_1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    expected_2 = 1 - expected_1
    
    actual_1 = 1 if winner == t1 else 0
    actual_2 = 1 if winner == t2 else 0
    
    elo_ratings[t1] = elo1 + K_FACTOR * (actual_1 - expected_1)
    elo_ratings[t2] = elo2 + K_FACTOR * (actual_2 - expected_2)

df['team1_last6_wins'] = t1_last6
df['team2_last6_wins'] = t2_last6
df['team1_h2h_last10'] = t1_h2h
df['team2_h2h_last10'] = t2_h2h
df['venue_bat_first_win_rate'] = v_rates

df['team1_elo'] = t1_elo_list 
df['team2_elo'] = t2_elo_list 

output_path = os.path.join(current_dir, 'pre_match_features.csv')
df.to_csv(output_path, index=False)
print(f"Success! Saved pure algorithmic V2.1 dataset to: {output_path}")