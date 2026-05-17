import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

try:
    API_URL = st.secrets["API_URL"]
except:
    API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Cric AI Predictor", page_icon="🏏", layout="wide")

@st.cache_data 
def load_metadata():
    try:
        res = requests.get(f"{API_URL}/metadata")
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
st.markdown("Advanced AI-powered cricket match prediction and analytics dashboard.")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor", "📊 Team Analytics", "⚔️ Head-to-Head"])

with tab1:
    st.header("Pre-Match Predictor")
    st.markdown("Select match parameters. The AI will calculate win probabilities.")
    
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
        stats_res = requests.get(f"{API_URL}/stats?team1={team_1}&team2={team_2}")
        
    if stats_res.status_code == 200:
        stats = stats_res.json()
        
        st.subheader("Live Form & Momentum")
        
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
                res = requests.post(f"{API_URL}/predict/pre-match", json=payload)
                
            if res.status_code == 200:
                result = res.json()
                winner = result['predicted_winner'].title()
                win_prob = result['win_probability'] * 100
                
                st.success(f"### 🏆 Predicted Winner: {winner} ({win_prob:.1f}%)")
                
                # Visual Chart
                labels = [team_1, team_2]
                values = [result['team_1_probability'] * 100, result['team_2_probability'] * 100]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Error calculating probability.")

with tab2:
    st.header("Team Analytics")
    selected_team = st.selectbox("Select Team for Analytics", TEAMS, key="team_analytics")
    
    if st.button("Fetch Analytics"):
        with st.spinner(f"Loading {selected_team} stats..."):
            res = requests.get(f"{API_URL}/team-analytics/{selected_team}")
            
        if res.status_code == 200:
            data = res.json()
            st.subheader(f"{selected_team} Overall Performance")
            
            c1, c2 = st.columns(2)
            c1.metric("Total Matches", data['total_matches'])
            c2.metric("Total Wins", data['total_wins'])
            
            st.markdown("#### Win Rates")
            wr1, wr2, wr3 = st.columns(3)
            wr1.metric("Overall Win %", f"{data['win_rate']*100:.1f}%")
            wr2.metric("Win % (Batting First)", f"{data['bat_first_win_rate']*100:.1f}%")
            wr3.metric("Win % (Bowling First)", f"{data['bowl_first_win_rate']*100:.1f}%")
            
            chart_data = pd.DataFrame({
                "Situation": ["Overall", "Batting First", "Bowling First"],
                "Win Rate (%)": [data['win_rate']*100, data['bat_first_win_rate']*100, data['bowl_first_win_rate']*100]
            })
            fig2 = px.bar(chart_data, x="Situation", y="Win Rate (%)", color="Situation", text_auto='.1f')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Fetch and display Elo History
            res_elo = requests.get(f"{API_URL}/elo-history/{selected_team}")
            if res_elo.status_code == 200:
                elo_data = res_elo.json().get('history', [])
                if elo_data:
                    st.markdown("#### Elo Rating History")
                    df_elo = pd.DataFrame(elo_data)
                    fig_elo = px.line(df_elo, x="match", y="elo", title=f"{selected_team} Form Progression")
                    fig_elo.update_layout(xaxis_title="Matches Played", yaxis_title="Elo Rating")
                    st.plotly_chart(fig_elo, use_container_width=True)
                    
        else:
            st.error("Could not fetch team analytics.")

with tab3:
    st.header("Head-to-Head History")
    col1, col2 = st.columns(2)
    h2h_team1 = col1.selectbox("Team 1", TEAMS, index=TEAMS.index("Pakistan") if "Pakistan" in TEAMS else 0, key="h2h1")
    h2h_team2 = col2.selectbox("Team 2", TEAMS, index=TEAMS.index("India") if "India" in TEAMS else 1, key="h2h2")
    
    if st.button("View History"):
        if h2h_team1 == h2h_team2:
            st.warning("Please select two different teams.")
        else:
            with st.spinner("Fetching history..."):
                res = requests.get(f"{API_URL}/h2h-history?team1={h2h_team1}&team2={h2h_team2}")
            
            if res.status_code == 200:
                json_data = res.json()
                h2h_data = json_data.get('history', [])
                t1_wins = json_data.get('team1_wins', 0)
                t2_wins = json_data.get('team2_wins', 0)
                
                if t1_wins > 0 or t2_wins > 0:
                    st.subheader("Overall H2H Distribution")
                    pie_data = pd.DataFrame({
                        "Team": [h2h_team1, h2h_team2],
                        "Wins": [t1_wins, t2_wins]
                    })
                    fig_pie = px.pie(pie_data, values='Wins', names='Team', hole=0.4, title=f"All-Time Matchups ({t1_wins + t2_wins} matches)")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.divider()

                if not h2h_data:
                    st.info("No recent matches found between these teams.")
                else:
                    st.subheader(f"Last {len(h2h_data)} Matches")
                    for match in h2h_data:
                        with st.container():
                            st.markdown(f"**Date:** {match['date']} | **Venue:** {match['venue']}")
                            st.success(f"🏆 Winner: {match['winner']}")
                            st.divider()
            else:
                st.error("Could not fetch head-to-head history.")