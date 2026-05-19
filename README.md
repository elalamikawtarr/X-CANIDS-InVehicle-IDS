#  X-CANIDS : In-Vehicle Intrusion Detection System

##  Présentation
Système de détection d'intrusion (IDS) hybride pour bus CAN automobile. Ce projet combine l'efficacité du Machine Learning (LightGBM) pour la détection rapide et la puissance du Deep Learning (Auto-encodeurs) pour l'analyse fine des signaux.

##  Architecture ML & MLOps
* **Core Engine :** LightGBM & XGBoost (Accuracy 98%).
* **Anomaly Detection :** Auto-encodeur TensorFlow pour la détection Zero-Day.
* **Pipeline Cloud :** Inférence automatisée sur Google BigQuery via Service Accounts.
* **Monitoring :** Dashboard temps réel avec Streamlit.

## Fichiers Clés
* `app.py` : Dashboard de visualisation.
* `deep_diagnostic_engine.py` : Moteur de diagnostic par auto-encodeur.
* `cloud_inference_worker.py` : Worker pour l'analyse des données Cloud.
* `secoc_module.py` : Simulation de la protection SecOC (Authentification).

##  Installation
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
