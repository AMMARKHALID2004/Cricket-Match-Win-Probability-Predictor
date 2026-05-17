from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import joblib
import os

# 1. Initialize FastAPI
app = FastAPI(title="Cric AI API", description="Production-Ready T20 Predictor")

# 2. Add CORS Middleware (CRITICAL for Streamlit connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your Streamlit app to talk to this API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup Paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# 4. Load Data and Process ELO
try:
    data_path = os.path.join(current_dir, '..', 'data', 'cleaned_base_data.csv')
    df_history = pd.read_csv(data_path)
    
    df_history['team_1'] = df_history['team_1'].astype(str).str.lower()
    df_history['team_2'] = df_history['team_2'].astype(str).str.lower()
    df_history['venue'] = df_history['venue'].astype(str).str.lower()
    df_history['winner'] = df_history['winner'].astype(str).str.lower()
    
    VALID_TEAMS = sorted(list(set(df_history['team_1'].unique()) | set(df_history['team_2'].unique())))
    VALID_VENUES = sorted(list(df_history['venue'].dropna().unique()))
    print(f"Database connected! Loaded {len(VALID_TEAMS)} teams and {len(VALID_VENUES)} venues.")
    
    CURRENT_ELO = {}
    ELO_HISTORY = {}
    match_idx = 1
    for _, row in df_history.iterrows():
        t1, t2, winner = row['team_1'], row['team_2'], row['winner']
        
        if t1 not in ELO_HISTORY: ELO_HISTORY[t1] = [{"match": 0, "elo": 1000}]
        if t2 not in ELO_HISTORY: ELO_HISTORY[t2] = [{"match": 0, "elo": 1000}]
        
        e1, e2 = CURRENT_ELO.get(t1, 1000), CURRENT_ELO.get(t2, 1000)
        expected_1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        
        new_e1 = e1 + 40 * ((1 if winner == t1 else 0) - expected_1)
        new_e2 = e2 + 40 * ((1 if winner == t2 else 0) - (1 - expected_1))
        
        CURRENT_ELO[t1] = new_e1
        CURRENT_ELO[t2] = new_e2
        
        ELO_HISTORY[t1].append({"match": match_idx, "elo": new_e1})
        ELO_HISTORY[t2].append({"match": match_idx, "elo": new_e2})
        match_idx += 1

except Exception as e:
    print(f"Database Error: {e}")
    VALID_TEAMS, VALID_VENUES, df_history, ELO_HISTORY = [], [], None, {}

# 5. Load Machine Learning Models
try:
    models_dir = os.path.join(current_dir, '..', 'models', 'saved_models')
    model = joblib.load(os.path.join(models_dir, 'xgboost_pre_match.joblib'))
    model_columns = joblib.load(os.path.join(models_dir, 'model_columns.joblib'))
except Exception as e:
    print(f"Model Error: {e}")

# 6. Pydantic Models for Validation
class MatchData(BaseModel):
    team_1: str
    team_2: str
    venue: str
    toss_winner: str
    toss_decision: str
    team1_last6_wins: int
    team2_last6_wins: int
    team1_h2h_last10: int
    team2_h2h_last10: int

    @field_validator('team_1', 'team_2', 'toss_winner')
    def validate_teams(cls, v):
        if v.lower().strip() not in VALID_TEAMS:
            raise ValueError(f"Unknown team: {v}")
        return v.lower().strip()

# 7. Routes
@app.get("/")
def home():
    return {"status": "online", "message": "Cric AI API is active and healthy"}

@app.get("/metadata")
def get_metadata():
    return {"teams": VALID_TEAMS, "venues": VALID_VENUES}

@app.get("/stats")
def get_match_stats(team1: str, team2: str):
    t1 = team1.lower()
    t2 = team2.lower()
    
    def get_recent(team):
        past = df_history[(df_history['team_1'] == team) | (df_history['team_2'] == team)]
        return int((past.tail(6)['winner'] == team).sum())
        
    def get_h2h(ta, tb):
        past = df_history[((df_history['team_1'] == ta) & (df_history['team_2'] == tb)) | 
                          ((df_history['team_1'] == tb) & (df_history['team_2'] == ta))]
        return int((past.tail(10)['winner'] == ta).sum())

    return {
        "team1_last6": get_recent(t1),
        "team2_last6": get_recent(t2),
        "team1_h2h": get_h2h(t1, t2),
        "team2_h2h": get_h2h(t2, t1)
    }

@app.post("/predict/pre-match")
def predict_pre_match(data: MatchData):
    try:
        is_toss_winner = (data.toss_winner == data.team_1)
        chose_to_bat = (data.toss_decision == 'bat')
        team1_batting_first = 1 if ((is_toss_winner and chose_to_bat) or (not is_toss_winner and not chose_to_bat)) else 0

        venue_bat_first_win_rate = 0.50 
        if df_history is not None:
            past_venue = df_history[df_history['venue'] == data.venue]
            if len(past_venue) >= 5: 
                bat_first_wins = 0
                for _, row in past_venue.iterrows():
                    if row['toss_decision'] == 'bat':
                        if row['winner'] == row['toss_winner']: bat_first_wins += 1
                    else:
                        if row['winner'] != row['toss_winner']: bat_first_wins += 1
                venue_bat_first_win_rate = bat_first_wins / len(past_venue)

        team1_elo = CURRENT_ELO.get(data.team_1, 1000)
        team2_elo = CURRENT_ELO.get(data.team_2, 1000)

        payload_dict = {
            'team_1': data.team_1,
            'team_2': data.team_2,
            'team1_last6_wins': data.team1_last6_wins,
            'team2_last6_wins': data.team2_last6_wins,
            'team1_h2h_last10': data.team1_h2h_last10,
            'team2_h2h_last10': data.team2_h2h_last10,
            'team1_batting_first': team1_batting_first,
            'venue_bat_first_win_rate': venue_bat_first_win_rate,
            'team1_elo': team1_elo,      
            'team2_elo': team2_elo        
        }
        
        df = pd.DataFrame([payload_dict])
        df_encoded = pd.get_dummies(df)
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        pred = model.predict(df_final)[0]
        probs = model.predict_proba(df_final)[0]
        
        return {
            "predicted_winner": data.team_1 if pred == 1 else data.team_2,
            "win_probability": float(probs[1] if pred == 1 else probs[0]),
            "team_1_probability": float(probs[1]),
            "team_2_probability": float(probs[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/team-analytics/{team}")
def get_team_analytics(team: str):
    t = team.lower()
    if df_history is None:
        raise HTTPException(status_code=500, detail="Database not loaded")
        
    past = df_history[(df_history['team_1'] == t) | (df_history['team_2'] == t)].copy()
    total_matches = len(past)
    if total_matches == 0:
        return {"total_matches": 0, "win_rate": 0, "bat_first_win_rate": 0, "bowl_first_win_rate": 0}
        
    total_wins = int((past['winner'] == t).sum())
    win_rate = total_wins / total_matches
    
    def did_bat_first(row):
        is_toss_winner = (row['toss_winner'] == t)
        if is_toss_winner:
            return row['toss_decision'] == 'bat'
        else:
            return row['toss_decision'] == 'field'
            
    past['batted_first'] = past.apply(did_bat_first, axis=1)
    
    bat_first_matches = past[past['batted_first'] == True]
    bowl_first_matches = past[past['batted_first'] == False]
    
    bat_first_wins = int((bat_first_matches['winner'] == t).sum())
    bowl_first_wins = int((bowl_first_matches['winner'] == t).sum())
    
    bat_first_win_rate = bat_first_wins / len(bat_first_matches) if len(bat_first_matches) > 0 else 0
    bowl_first_win_rate = bowl_first_wins / len(bowl_first_matches) if len(bowl_first_matches) > 0 else 0
    
    return {
        "total_matches": total_matches,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "bat_first_win_rate": bat_first_win_rate,
        "bowl_first_win_rate": bowl_first_win_rate
    }

@app.get("/h2h-history")
def get_h2h_history(team1: str, team2: str):
    t1 = team1.lower()
    t2 = team2.lower()
    if df_history is None:
        raise HTTPException(status_code=500, detail="Database not loaded")
        
    past = df_history[((df_history['team_1'] == t1) & (df_history['team_2'] == t2)) | 
                      ((df_history['team_1'] == t2) & (df_history['team_2'] == t1))]
    
    t1_wins = int((past['winner'] == t1).sum())
    t2_wins = int((past['winner'] == t2).sum())
                      
    past_5 = past.tail(5) 
    
    history = []
    for _, row in past_5.iterrows():
        history.append({
            "date": row.get('date', 'Unknown'),
            "venue": row['venue'].title() if isinstance(row['venue'], str) else row['venue'],
            "winner": row['winner'].title() if isinstance(row['winner'], str) else row['winner']
        })
        
    return {
        "team1_wins": t1_wins,
        "team2_wins": t2_wins,
        "history": history[::-1]
    }

@app.get("/elo-history/{team}")
def get_elo_history(team: str):
    t = team.lower()
    if t not in ELO_HISTORY:
        raise HTTPException(status_code=404, detail="Team not found in history")
    return {"history": ELO_HISTORY[t]}