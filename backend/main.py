from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import joblib
import os

app = FastAPI(title="Cric AI API", description="Production-Ready T20 Predictor")

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load the Historical Database for Auto-Stats & Valid Options
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
except Exception as e:
    print(f"Database Error: {e}")
    VALID_TEAMS, VALID_VENUES, df_history = [], [], None

# 2. Load the AI Model
try:
    models_dir = os.path.join(current_dir, '..', 'models', 'saved_models')
    model = joblib.load(os.path.join(models_dir, 'xgboost_pre_match.joblib'))
    model_columns = joblib.load(os.path.join(models_dir, 'model_columns.joblib'))
except Exception as e:
    print(f"Model Error: {e}")

# 3. Input Schema
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

# 4. Endpoints
@app.get("/metadata")
def get_metadata():
    return {"teams": VALID_TEAMS, "venues": VALID_VENUES}

# NEW: Automated Stats Calculator!
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
        df = pd.DataFrame([data.model_dump()])
        df['team1_won_toss'] = (df['toss_winner'] == df['team_1']).astype(int)
        df = df.drop(columns=['toss_winner'])
        
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