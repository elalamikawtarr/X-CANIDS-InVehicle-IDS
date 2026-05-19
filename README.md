# X-CANIDS – In-Vehicle Intrusion Detection System

##  Présentation

X-CANIDS est un système intelligent de détection d’intrusion (IDS) pour les réseaux CAN automobile.  
Il combine des techniques de Machine Learning et de Deep Learning afin de détecter à la fois les attaques connues et les anomalies zero-day.

Le système est conçu pour améliorer la cybersécurité des véhicules modernes en analysant les communications internes du bus CAN en temps réel.

---

## Problématique

Les véhicules connectés sont de plus en plus exposés à des attaques informatiques ciblant le réseau CAN :

- Injection de messages falsifiés  
- Attaques par usurpation  
- Anomalies invisibles (zero-day)  
- Manipulation des signaux véhicules  

Les approches classiques basées sur des règles ne suffisent plus.

---

##  Architecture du système

- Prétraitement des données CAN  
- Feature Engineering  
- Machine Learning (LightGBM / XGBoost)  
- Deep Learning (Autoencoder TensorFlow)  
- Moteur de diagnostic intelligent  
- Dashboard Streamlit  
- Module Cloud (BigQuery + Worker)  

---

## Modèles utilisés

### Machine Learning
- LightGBM  
- XGBoost  


### Deep Learning
- Autoencoder (TensorFlow/Keras)  

---

## Pipeline

1. Chargement dataset CAN  
2. Nettoyage et prétraitement  
3. Feature engineering  
4. Entraînement ML  
5. Entraînement Deep Learning  
6. Détection d’anomalies  
7. Diagnostic intelligent  
8. Visualisation (Dashboard)  

---

## Fonctionnalités

- Détection d’intrusions CAN en temps réel  
- Système hybride ML + DL  
- Dashboard Streamlit interactif  
- Module Cloud BigQuery  
- Simulation SecOC (sécurité automobile)  
- Explicabilité des décisions  

---

## Structure du projet

- `app.py` → Dashboard Streamlit  
- `train.py` → Entraînement modèles ML  
- `inference.py` → Prédiction temps réel  
- `preprocessing.py` → Nettoyage données  
- `deep_autoencoder.py` → Modèle Deep Learning  
- `deep_diagnostic_engine.py` → Analyse anomalies  
- `secoc_module.py` → Sécurité SecOC  
- `cloud_inference_worker.py` → Cloud processing  
- `evaluate.py` → Évaluation modèles  
- `explain.py` → Interprétabilité  

---

## Installation

```bash
git clone https://github.com/elalamikawtarr/X-CANIDS-InVehicle-IDS
cd X-CANIDS-InVehicle-IDS
pip install -r requirements.txt
