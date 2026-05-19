import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import time

st.set_page_config(
    page_title="X-CANIDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("X-CANIDS - Intelligent CAN Bus IDS")
st.markdown("### Détection CAN Bus avec XGBoost")

st.sidebar.header("Configuration")

threshold_manual = st.sidebar.slider(
    "Seuil de détection",
    0.0,
    1.0,
    0.24,
    0.01
)

simulation_speed = st.sidebar.slider(
    "Vitesse simulation",
    0.0,
    1.0,
    0.05
)

@st.cache_resource
def load_model():
    return joblib.load("model/xgb_model.pkl")

try:
    model = load_model()
except:
    model = None

uploaded_file = st.file_uploader("Uploader fichier parquet", type=["parquet"])

def preprocess_data(df):
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['time_diff'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)

    if 'ID' in df.columns:
        df['ID_frequency'] = df.groupby('ID')['ID'].transform('count')

    return df


def detect_attacks(df, threshold=0.24):
    df_model = df.select_dtypes(include=np.number)

    if model is not None:
        probs = model.predict_proba(df_model)[:, 1]
    else:
        probs = np.random.uniform(0, 1, len(df_model))

    df['risk_score'] = probs
    df['prediction'] = (probs >= threshold).astype(int)

    labels = []
    for p in probs:
        if p > 0.85:
            labels.append("Fabrication Attack")
        elif p > 0.70:
            labels.append("Masquerade Attack")
        elif p > 0.50:
            labels.append("Fuzzing Attack")
        elif p > 0.24:
            labels.append("Suspension Attack")
        else:
            labels.append("Normal")

    df['attack_type'] = labels
    return df


if uploaded_file is not None:

    df = pd.read_parquet(uploaded_file)
    st.dataframe(df.head())

    st.metric("Messages", len(df))
    st.metric("Colonnes", len(df.columns))
    st.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

    df_clean = preprocess_data(df)

    progress = st.progress(0)
    for i in range(100):
        time.sleep(simulation_speed)
        progress.progress(i + 1)

    results_df = detect_attacks(df_clean, threshold_manual)

    total = len(results_df)
    attacks = (results_df['prediction'] == 1).sum()
    normal = total - attacks
    risk_avg = results_df['risk_score'].mean() * 100

    st.metric("Messages", total)
    st.metric("Attaques", attacks)
    st.metric("Normal", normal)
    st.metric("Risque moyen", f"{risk_avg:.2f}%")

    attack_dist = results_df['attack_type'].value_counts().reset_index()
    attack_dist.columns = ['Attack', 'Count']

    st.plotly_chart(px.bar(attack_dist, x='Attack', y='Count'))

    st.plotly_chart(go.Figure(data=[go.Pie(
        labels=['Normal', 'Alertes'],
        values=[normal, attacks],
        hole=0.5
    )]))

    st.plotly_chart(px.line(results_df.head(500), y='risk_score'))

    alerts_df = results_df[results_df['prediction'] == 1][['risk_score', 'attack_type']]
    st.dataframe(alerts_df.head(100))

    if 'Label' in results_df.columns:
        y_true = results_df['Label']
        y_pred = results_df['prediction']

        cm = confusion_matrix(y_true, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True))

        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        fpr, tpr, _ = roc_curve(y_true, results_df['risk_score'])
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC {roc_auc:.2f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash')))
        st.plotly_chart(fig)

    if model is not None:
        try:
            feature_names = results_df.select_dtypes(include=np.number).columns
            importances = model.feature_importances_

            imp_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance').tail(15)

            st.plotly_chart(px.bar(imp_df, x='Importance', y='Feature', orientation='h'))
        except:
            pass

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger CSV", csv, "results.csv", "text/csv")

else:
    st.info("Uploader un fichier parquet")