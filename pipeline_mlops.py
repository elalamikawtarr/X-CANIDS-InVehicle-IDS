import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import precision_recall_curve, confusion_matrix
import mlflow
import mlflow.xgboost

# Configuration initiale de l'environnement
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_experiment("X-CANIDS-IDS-Global")

print("--- DEBUT DU PIPELINE PRATIQUE MLOps ---")

with mlflow.start_run(run_name="XGBoost_Pipeline_Dynamique"):
    
    # ==========================================
    # ÉTAPE A : ALIGNEMENT DES FEATURES & CALCUL DYNAMIQUE
    # ==========================================
    
    # 1. Extraction ordonnée des features attendues par le modèle
    features_attendues = final_model.get_booster().feature_names
    
    # 2. Alignement strict et reconstruction du DataFrame dans l'ordre EXACT
    # Si des colonnes manquent dans X_test_G, on les crée remplies de 0 pour ne pas planter
    X_test_G_aligne = X_test_G.reindex(columns=features_attendues, fill_value=0)
    
    print(f"🔄 Alignement effectué : {X_test_G_aligne.shape[1]} colonnes alignées dans le bon ordre.")

    # 3. CONVERSION NUMPY CRITIQUE : Évite le "feature_names mismatch" de l'interface Pandas de XGBoost
    X_test_matrix = X_test_G_aligne.values
    
    # Calcul des probabilités sur la matrice NumPy
    probs_global = final_model.predict_proba(X_test_matrix)[:, 1]
    
    # Calcul de la courbe Précision-Recall pour trouver le meilleur seuil
    precisions, recalls, thresholds = precision_recall_curve(y_test_G, probs_global)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    seuil_optimal = thresholds[np.argmax(f1_scores)]
    
    # Application du seuil calculé
    y_pred_final = (probs_global >= seuil_optimal).astype(int)
    cm = confusion_matrix(y_test_G, y_pred_final)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Envoi des paramètres réels calculés à MLflow
    mlflow.log_param("seuil_optimal_calcule", float(seuil_optimal))
    mlflow.log_param("nombre_features_can", X_test_G_aligne.shape[1])
    
    # Envoi des métriques réelles à MLflow
    mlflow.log_metric("vrais_positifs", int(TP))
    mlflow.log_metric("faux_positifs", int(FP))
    mlflow.log_metric("faux_negatifs", int(FN))
    mlflow.log_metric("vrais_negatifs", int(TN))
    mlflow.log_metric("recall_final_pourcent", float((TP / (TP + FN)) * 100))

    print(f"   Seuil optimal détecté automatiquement : {seuil_optimal:.4f}")
    print(f"   Nombre de Faux Positifs calculé : {FP}")

    # ==========================================
    # ÉTAPE B : GÉNÉRATION & SAUVEGARDE AUTOMATIQUE DES ARTÉFACTS
    # ==========================================
    os.makedirs("mlflow_artifacts", exist_ok=True)
    
    # 1. Matrice de Confusion Interactive
    fig_cm = px.imshow(cm, text_auto=True,
                       labels=dict(x="Prédit", y="Réel"),
                       x=['Normal', 'Attaque'], y=['Normal', 'Attaque'],
                       title="Matrice de Confusion Dynamique - Seuil Optimisé")
    
    cm_path = "mlflow_artifacts/matrice_confusion.html"
    fig_cm.write_html(cm_path)
    mlflow.log_artifact(cm_path, "evaluation_plots")
    
    # 2. Importance des variables (Feature Importance)
    importances = final_model.feature_importances_
    df_importance = pd.DataFrame({'Feature': X_test_G_aligne.columns, 'Importance': importances})
    df_importance = df_importance.sort_values(by='Importance').tail(15)
    
    fig_imp = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                     title="Top 15 des signaux CAN détecteurs d'intrusions",
                     color='Importance', color_continuous_scale='Reds')
    
    imp_path = "mlflow_artifacts/feature_importance.html"
    fig_imp.write_html(imp_path)
    mlflow.log_artifact(imp_path, "evaluation_plots")
    
    print("   Graphiques interactifs (HTML) générés et poussés dans MLflow !")

    # ==========================================
    # ÉTAPE C : LOG DU MODÈLE LOGICIEL DANS LE REGISTRE
    # ==========================================
    mlflow.xgboost.log_model(
        xgb_model=final_model, 
        artifact_path="model_ids",
        registered_model_name="X-CANIDS-InVehicle-IDS"
    )
    
    print("   Modèle enregistré avec succès dans l'onglet 'Models' de MLflow !")

print("--- FIN DU PIPELINE : VÉRIFIER http://127.0.0.1:5000 ---")
