# PIPELINE D'INFÉRENCE CLOUD : DÉPLOIEMENT ET PUBLICATION
# Ce script automatise le cycle complet : 
# 1. Extraction des caractéristiques "Gold" depuis BigQuery.
# 2. Prédiction en temps réel via le modèle XGBoost/LGBM optimisé.
# 3. Ré-injection des scores d'anomalies dans le Cloud pour le monitoring.


import pandas as pd
import numpy as np
import joblib
import os
from google.cloud import bigquery
from google.oauth2 import service_account


# Utilisation de ta clé JSON pour l'authentification
KEY_PATH = r"C:\Users\elala\Downloads\Projet\key_cloud.json"

if not os.path.exists(KEY_PATH):
    raise FileNotFoundError(f"La clé JSON n'est pas au chemin : {KEY_PATH}")

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)


MODEL_PATH = "x_canids_model_vfinal.pkl"

try:
    model = joblib.load(MODEL_PATH)
    SEUIL_OPTIMAL = 0.2415
    print(f" Modèle chargé. Seuil de détection : {SEUIL_OPTIMAL}")
except:
    print("  Erreur : Le fichier .pkl du modèle est introuvable !")

# LECTURE DES DONNÉES DEPUIS BIGQUERY ---
# On utilise la table 'vw_gold_features_analyst' qui existe réellement
query = """
SELECT * FROM `project-e6de9b55-41d5-4f13-ae0.can_ids_gold.vw_gold_features_analyst`
LIMIT 50000
"""

print(" Récupération des données en cours depuis BigQuery...")
df_brut = client.query(query).to_dataframe()

#  PRÉPARATION ET PRÉDICTION IA ---
print("  Analyse du trafic CAN par l'IA...")

# On ne garde que les colonnes numériques pour le modèle
X = df_brut.select_dtypes(include=[np.number])

# On aligne les colonnes avec ce que le modèle attend (important !)
expected_features = model.feature_names_in_
X = X.reindex(columns=expected_features, fill_value=0)

# Calcul des probabilités d'attaque
probs = model.predict_proba(X)[:, 1]

# Ajout des résultats au DataFrame
df_brut['anomaly_score'] = probs
df_brut['is_alert'] = (probs >= SEUIL_OPTIMAL).astype(int)

# PUBLICATION DES RÉSULTATS DANS BIGQUERY ---
# On crée une nouvelle table de prédictions finales
DESTINATION_TABLE = "project-e6de9b55-41d5-4f13-ae0.can_ids_gold.final_predictions_results"

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE" # On écrase la table à chaque fois pour la mettre à jour
)

print(f"  Envoi des résultats vers {DESTINATION_TABLE}...")
client.load_table_from_dataframe(df_brut, DESTINATION_TABLE, job_config=job_config).result()

print("============================================================")
print(" ANALYSE TERMINÉE AVEC SUCCÈS")
print(f" Total messages analysés : {len(df_brut)}")
