# DESCRIPTION GÉNÉRALE :
# Ce script est une application Streamlit qui sert de tableau de bord de sécurité 
# pour le réseau interne d'un véhicule connecté. Son objectif est 
# d'analyser les flux de données, de détecter les anomalies à l'aide de l'IA, 
# et de classifier les types d'attaques en temps réel
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

from google.cloud import bigquery
from google.oauth2 import service_account

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="X-CANIDS",
    layout="wide"
)

st.title("X-CANIDS")
st.subheader("Cloud Automotive Intrusion Detection System")

# =========================================================
# STATUS BOX CLEAN
# =========================================================

def status_box(label, state="INFO"):
    if state == "INFO":
        st.info(label)
    elif state == "SUCCESS":
        st.success(label)
    elif state == "ERROR":
        st.error(label)

# =========================================================
# GOOGLE AUTH (IMPORTANT: DO NOT HARD CODE IN PRODUCTION)
# =========================================================

credentials = service_account.Credentials.from_service_account_file(
    "key_cloud.json"
)

client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id
)

status_box("BigQuery connection established", "SUCCESS")

# =========================================================
# CACHE DATA LOADING (CRITICAL PERFORMANCE FIX)
# =========================================================

@st.cache_data
def load_data():
    query = """
    SELECT *
    FROM `project-e6de9b55-41d5-4f13-ae0.can_ids_bqnative_gold_ml.signal_big_table`
    LIMIT 50000
    """
    return client.query(query).to_dataframe()

# =========================================================
# PIPELINE BUTTON
# =========================================================

run = st.button("Run IDS Pipeline")

if not run:
    st.warning("Click 'Run IDS Pipeline' to start analysis")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================

status_box("Loading dataset from BigQuery...", "INFO")

df = load_data()

status_box(f"Dataset loaded: {len(df)} records", "SUCCESS")

st.write("Shape:", df.shape)
st.dataframe(df.head())

# =========================================================
# LABEL DETECTION
# =========================================================

label_candidates = ['label', 'Label', 'attack', 'class', 'is_attack']
label_col = next((c for c in label_candidates if c in df.columns), None)

if not label_col:
    st.stop()

status_box(f"Target column detected: {label_col}", "SUCCESS")

# =========================================================
# ATTACK TYPE
# =========================================================

attack_candidates = ['attack_type', 'type', 'category']
attack_col = next((c for c in attack_candidates if c in df.columns), None)

if attack_col:
    status_box(f"Attack type detected: {attack_col}", "SUCCESS")

# =========================================================
# SPLIT
# =========================================================

y = df[label_col]
X = df.drop(columns=[label_col] + ([attack_col] if attack_col else []))

# =========================================================
# TIMESTAMP ENGINEERING
# =========================================================

time_candidates = ['timestamp', 'time', 'datetime']
time_col = next((c for c in time_candidates if c in X.columns), None)

if time_col:
    X[time_col] = pd.to_datetime(X[time_col])
    X = X.sort_values(time_col)

    X["delta_t"] = X[time_col].diff().dt.total_seconds().fillna(0)
    X["dt_spike"] = (X["delta_t"] > 0.5).astype(int)
    X["high_freq_attack"] = (X["delta_t"] < 0.0005).astype(int)

    X = X.drop(columns=[time_col])

status_box(f"Features retained: {X.shape[1]}", "SUCCESS")

# =========================================================
# CLEAN
# =========================================================

X = X.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)
X = X.loc[:, X.nunique() > 1]

# =========================================================
# TRAIN TEST SPLIT (FIXED RANDOM STATE = STABLE RESULTS)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================================
# UNDERSAMPLING
# =========================================================

train_df = X_train.copy()
train_df["label"] = y_train.values

minor = train_df[train_df["label"] == 1]
major = train_df[train_df["label"] == 0].sample(
    n=min(len(train_df[train_df["label"] == 0]), len(minor) * 3),
    random_state=42
)

train_balanced = pd.concat([minor, major]).sample(frac=1, random_state=42)

X_train_bal = train_balanced.drop(columns=["label"])
y_train_bal = train_balanced["label"]

status_box(f"Balanced dataset: {len(major)} normal / {len(minor)} attack", "SUCCESS")

# =========================================================
# MODEL (CACHED)
# =========================================================

@st.cache_resource
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

status_box("Training model...", "INFO")
model = train_model(X_train_bal, y_train_bal)
status_box("Model training completed", "SUCCESS")

# =========================================================
# PREDICTION
# =========================================================

probs = model.predict_proba(X_test)[:, 1]

# hybrid rules
score = probs.copy()
if "dt_spike" in X_test.columns:
    score += X_test["dt_spike"] * 0.2

if "high_freq_attack" in X_test.columns:
    score += X_test["high_freq_attack"] * 0.2

# =========================================================
# THRESHOLD (STABLE FIX)
# =========================================================

prec, rec, thr = precision_recall_curve(y_test, score)
f1 = (2 * prec * rec) / (prec + rec + 1e-10)

best_thr = thr[np.argmax(f1)]

y_pred = (score >= best_thr).astype(int)

# =========================================================
# METRICS
# =========================================================

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

# =========================================================
# KPIS
# =========================================================

st.subheader("Security KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Messages", len(df))
col2.metric("Alerts", int(y_pred.sum()))
col3.metric("Detected attacks", TP)
col4.metric("False positives", FP)

# =========================================================
# CLEAN DASHBOARD
# =========================================================

st.subheader("Dashboard")

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "domain"}, {"type": "indicator"}],
           [{"type": "table"}, {"type": "scatter"}]]
)

fig.add_trace(go.Pie(values=[TN+FN, FP+TP], labels=["Normal", "Alerts"], hole=0.5), 1, 1)

fig.add_trace(go.Indicator(value=(TP/(TP+FP))*100 if TP+FP>0 else 0), 1, 2)

fig.add_trace(go.Table(
    header=dict(values=["TN", "FP", "FN", "TP"]),
    cells=dict(values=[[TN], [FP], [FN], [TP]])
), 2, 1)

fig.add_trace(go.Scatter(y=score[:300]), 2, 2)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TABLE RESULTS
# =========================================================

result = X_test.copy()
result["true"] = y_test.values
result["pred"] = y_pred
result["score"] = score

if attack_col:
    result["attack_type"] = df.loc[X_test.index, attack_col]

st.subheader("Prediction Table")
st.dataframe(result.sort_values("score", ascending=False).head(500))

# =========================================================
# REPORT
# =========================================================

st.subheader("Classification Report")

st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

status_box("Pipeline executed successfully", "SUCCESS")
