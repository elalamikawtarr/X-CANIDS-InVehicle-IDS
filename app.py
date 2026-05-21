# DESCRIPTION GÉNÉRALE :
# Ce script est une application Streamlit qui sert de tableau de bord de sécurité 
# pour le réseau interne (Bus CAN) d'un véhicule connecté. Son objectif est 
# d'analyser les flux de données, de détecter les anomalies à l'aide de l'IA, 
# et de classifier les types d'attaques en temps réel

import streamlit as st
import pandas as pd
import numpy as np

from google.cloud import bigquery
from google.oauth2 import service_account

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="X-CANIDS",
    layout="wide"
)

st.title("X-CANIDS - Automotive IDS")
st.markdown("Intelligent CAN Bus Intrusion Detection System")


KEY_PATH = r"C:\Users\elala\Downloads\Projet\key_cloud.json"
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
    location="europe-southwest1"
)

st.success("BigQuery Connected")

query = """
SELECT
    session_id,
    arbitration_id,
    signals_count,
    messages_count,
    uniq_data_count,
    interval_mean,
    interval_std,
    uniq_dlc
FROM `project-e6de9b55-41d5-4f13-ae0.can_ids_gold.vw_can_id_profile_by_session`
LIMIT 15000
"""

try:
    df = client.query(query).to_dataframe()
except Exception as e:
    st.error(f"Query Error: {e}")
    st.stop()

df.columns = df.columns.str.strip()

numeric_cols = [
    "signals_count",
    "messages_count",
    "uniq_data_count",
    "interval_mean",
    "interval_std",
    "uniq_dlc"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(0)

st.success(f"Dataset Loaded: {df.shape[0]} rows")

df["data_ratio"] = df["uniq_data_count"] / (df["messages_count"] + 1)

df["burst_score"] = df["messages_count"] / (df["interval_mean"] + 1e-6)

df["timing_score"] = df["interval_std"] / (df["interval_mean"] + 1e-6)

df["signal_density"] = df["signals_count"] / (df["uniq_dlc"] + 1)

df["repeat_ratio"] = 1 - df["data_ratio"]

features = [
    "signals_count",
    "messages_count",
    "uniq_data_count",
    "interval_mean",
    "interval_std",
    "uniq_dlc",
    "data_ratio",
    "burst_score",
    "timing_score",
    "signal_density",
    "repeat_ratio"
]

X = df[features].copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=500,
    contamination=0.20,
    max_samples="auto",
    bootstrap=True,
    random_state=42
)

model.fit(X_scaled)

scores = -model.decision_function(X_scaled)

scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

df["risk_score"] = scores_norm * 100

threshold = np.percentile(df["risk_score"], 80)

df["is_attack"] = np.where(df["risk_score"] >= threshold, 1, 0)

df["attack_type"] = "Normal Traffic"

msg_high = df["messages_count"].quantile(0.97)
uniq_high = df["uniq_data_count"].quantile(0.97)
interval_low = df["interval_mean"].quantile(0.05)
std_high = df["interval_std"].quantile(0.95)
burst_high = df["burst_score"].quantile(0.97)
timing_high = df["timing_score"].quantile(0.95)

dos_condition = (
    (df["messages_count"] > msg_high)
    & (df["interval_mean"] < interval_low)
    & (df["burst_score"] > burst_high)
    & (df["data_ratio"] < 0.4)
)

df.loc[dos_condition, "attack_type"] = "DoS Attack"

fuzz_condition = (
    (df["uniq_data_count"] > uniq_high)
    & (df["data_ratio"] > 0.75)
    & (df["timing_score"] > timing_high)
)

df.loc[fuzz_condition, "attack_type"] = "Fuzzing Attack"

replay_condition = (
    (df["messages_count"] > msg_high * 0.6)
    & (df["repeat_ratio"] > 0.80)
    & (df["interval_std"] < std_high * 0.5)
)

df.loc[replay_condition, "attack_type"] = "Replay Attack"

df.loc[
    (df["risk_score"] > 65) & (df["attack_type"] == "Normal Traffic"),
    "attack_type"
] = "Suspicious Activity"

df.loc[df["attack_type"] != "Normal Traffic", "is_attack"] = 1

normal_pct = round((df["is_attack"] == 0).mean() * 100, 2)
attack_pct = round((df["is_attack"] == 1).mean() * 100, 2)
global_risk = round(df["risk_score"].mean(), 2)

c1, c2, c3 = st.columns(3)

c1.metric("Normal Traffic", f"{normal_pct}%")
c2.metric("Attacks Detected", f"{attack_pct}%")
c3.metric("Global Risk", f"{global_risk}%")

alerts = df[df["is_attack"] == 1].sort_values("risk_score", ascending=False)

st.subheader("Detected Threats")

st.dataframe(
    alerts[
        [
            "session_id",
            "arbitration_id",
            "messages_count",
            "uniq_data_count",
            "risk_score",
            "attack_type"
        ]
    ],
    use_container_width=True
)

st.subheader("Attack Distribution")

fig1 = px.histogram(alerts, x="attack_type", color="attack_type")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Risk Timeline")

fig2 = px.line(df.head(1000), y="risk_score")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Traffic Classification")

fig3 = px.pie(df, names="attack_type")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Top Dangerous Sessions")

top = df.sort_values("risk_score", ascending=False).head(20)

st.dataframe(
    top[
        [
            "session_id",
            "arbitration_id",
            "risk_score",
            "attack_type"
        ]
    ],
    use_container_width=True
)

st.subheader("Global Threat Level")

fig4 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=global_risk,
    title={'text': "Threat Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "red"},
        'steps': [
            {'range': [0, 30], 'color': "green"},
            {'range': [30, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}
        ]
    }
))

st.plotly_chart(fig4, use_container_width=True)

st.subheader("Feature Correlation")

corr = df[features].corr()

fig5 = px.imshow(corr, text_auto=True, aspect="auto")

st.plotly_chart(fig5, use_container_width=True)

st.success("X-CANIDS Detection Engine Running")
