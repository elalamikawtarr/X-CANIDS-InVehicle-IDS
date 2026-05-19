# X-CANIDS-InVehicle-IDS Système de Détection d'Intrusion Automobile

## Présentation du Projet
Ce projet implémente un IDS (Intrusion Detection System) pour le réseau CAN (Controller Area Network) des véhicules,  Le système combine du Machine Learning (LightGBM) et du Deep Learning (Auto-encodeurs) pour détecter des anomalies en temps réel.

##  Architecture Technique
- **Modèle Global** : LightGBM Classifier (98% d'accuracy).
- **Seuil Optimal** : 0.2415 (établi par courbe Precision-Recall).
- **Explicabilité** : Analyse de l'importance des features pour identifier les signaux CAN compromis.
- **Interface** : Dashboard interactif réalisé avec Streamlit.

## Résultats 
- **Volume d'entraînement** : +300 000 messages CAN.
- **Détection des menaces** : Fuzzing, Masquerade, et Suspension.
- **Performance** : Temps d'inférence < 2ms par message.

## Installation et Utilisation
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
    pour lancez le dashboard : streamlit run app11.py   **MLops**
