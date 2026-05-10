import streamlit as st
import requests

st.set_page_config(page_title="Cric AI Predictor", page_icon="🏏", layout="wide")

@st.cache_data 
def load_metadata():
    try:
        res = requests.get("http://127.0.0.1:8000/metadata")
        return res.json() if res.status_code == 200 else None
    except:
        return None

metadata = load_metadata()

if not metadata or not metadata.get('venues'):
    st.error("🚨 Backend is offline or database failed to load. Please restart FastAPI.")
    st.stop()

TEAMS = [t.title() for t in metadata['teams']]
VENUES = [v.title() for v in metadata['venues']]

st.title("🏏 Cric AI: Pro Predictor")
st.markdown("Select match parameters. The AI will automatically fetch live momentum stats and calculate win probabilities.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Matchups")
    team_1 = st.selectbox("Team 1 (Batting First/Host)", TEAMS, index=TEAMS.index("Pakistan") if "Pakistan" in TEAMS else 0)
    team_2 = st.selectbox("Team 2 (Challenger)", TEAMS, index=TEAMS.index("India") if "India" in TEAMS else 1)

with col2:
    st.subheader("Environment")
    venue = st.selectbox("Venue", VENUES)
    
with col3:
    st.subheader("Toss Info")
    toss_winner = st.selectbox("Who won the toss?", [team_1, team_2])
    toss_decision = st.selectbox("Decision", ["Bat", "Field"])

st.divider()

# --- AUTOMATED STATS FETCHING ---
with st.spinner("Fetching historical team data..."):
    stats_res = requests.get(f"http://127.0.0.1:8000/stats?team1={team_1}&team2={team_2}")
    
if stats_res.status_code == 200:
    stats = stats_res.json()
    
    st.subheader("📊 Live Form & Momentum (Calculated by AI)")
    st.caption("These stats are automatically fetched from the database and fed into the prediction model.")
    
    # Displaying stats beautifully, not as inputs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{team_1} Recent Form", f"{stats['team1_last6']} / 6 Wins")
    m2.metric(f"{team_2} Recent Form", f"{stats['team2_last6']} / 6 Wins")
    m3.metric(f"{team_1} H2H Wins", f"{stats['team1_h2h']} / 10")
    m4.metric(f"{team_2} H2H Wins", f"{stats['team2_h2h']} / 10")
    
    st.divider()

    if st.button("🔮 Calculate Win Probability", type="primary", use_container_width=True):
        
        payload = {
            "team_1": team_1.lower(),
            "team_2": team_2.lower(),
            "venue": venue.lower(),
            "toss_winner": toss_winner.lower(),
            "toss_decision": toss_decision.lower(),
            "team1_last6_wins": stats["team1_last6"],
            "team2_last6_wins": stats["team2_last6"],
            "team1_h2h_last10": stats["team1_h2h"],
            "team2_h2h_last10": stats["team2_h2h"]
        }
        
        with st.spinner("Crunching historical data and decision trees..."):
            res = requests.post("http://127.0.0.1:8000/predict/pre-match", json=payload)
            
        if res.status_code == 200:
            result = res.json()
            winner = result['predicted_winner'].title()
            win_prob = result['win_probability'] * 100
            
            st.success(f"### 🏆 Predicted Winner: {winner} ({win_prob:.1f}%)")
            st.progress(win_prob / 100)
            
            r1, r2 = st.columns(2)
            r1.info(f"**{team_1}** Win Probability: {result['team_1_probability'] * 100:.1f}%")
            r2.info(f"**{team_2}** Win Probability: {result['team_2_probability'] * 100:.1f}%")
        else:
            st.error("Error calculating probability.")