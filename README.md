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
- Autoencoder 

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
##  Benchmark & Justification du Modèle
Pour garantir la sécurité du véhicule, nous avons comparé trois modèles sur un dataset de **313 909 messages CAN**.

| Modèle | Détection (Recall) | Précision | F1-Score | Temps (sec) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Selected)** | **99.34%** | **98.01%** | **0.9867** | **49.28** |
| Random Forest | 99.69% | 97.45% | 0.9855 | 101.70 |
| Decision Tree | 97.19% | 98.01% | 0.9760 | 48.26

----

## References
[CANet] Hanselmann, M., Strauss, T., Dormann, K., & Ulmer, H. (2020). CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data. Ce papier justifie ton choix d'utiliser des Auto-encodeurs (Deep Learning non-supervisé).

[ROAD Dataset] Verma, M. E., et al. (2020). Addressing the lack of comparability & testing in CAN intrusion detection research: A comprehensive guide to CAN IDS data & introduction of the ROAD dataset. arXiv preprint arXiv:2012.14600. Ce papier explique pourquoi les anciens datasets étaient limités et pourquoi le filtrage des données est crucial.

[X-CANIDS] X-CANIDS Dataset (In-Vehicle Signal Dataset). IEEE DataPort. C'est ta source de données principale, celle qui contient les attaques de type Fuzzing, Spoofing et Replay.

---

## Installation

```bash
git clone https://github.com/elalamikawtarr/X-CANIDS-InVehicle-IDS
cd X-CANIDS-InVehicle-IDS
pip install -r requirements.txt

---

