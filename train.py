import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("---  PRÉPARATION ET ENTRAÎNEMENT DU MODÈLE FINAL ---")

#  détection automatique du nom de la colonne cible 
if 'Label' in dataset_final.columns:
    nom_cible = 'Label'
elif 'label' in dataset_final.columns:
    nom_cible = 'label'
else:
    # Au cas où, on prend la dernière colonne 
    nom_cible = dataset_final.columns[-1]

print(f" Colonne cible détectée : '{nom_cible}'")

# Préparation des colonnes (X) et de la cible (y)
X_global = dataset_final.drop(columns=[nom_cible])
y_global = dataset_final[nom_cible]

#  Conversion des colonnes textuelles en nombres
X_global_clean = X_global.copy()
colonnes_objets = X_global_clean.select_dtypes(include=['object']).columns

if len(colonnes_objets) > 0:
    print(f" Conversion des colonnes textuelles : {list(colonnes_objets)}")
    for col in colonnes_objets:
        le = LabelEncoder()
        X_global_clean[col] = le.fit_transform(X_global_clean[col].astype(str))


split_index = int(len(X_global_clean) * 0.8)
X_train_G = X_global_clean.iloc[:split_index]
X_test_G = X_global_clean.iloc[split_index:]
y_train_G = y_global.iloc[:split_index]
y_test_G = y_global.iloc[split_index:]

print(f" Entraînement sur : {len(X_train_G)} messages")
print(f"Test sur : {len(X_test_G)} messages")

# Entraînement de XGBoost
print("\n   Entraînement en cours (XGBoost)...")
modele_global = xgb.XGBClassifier(
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1
)
modele_global.fit(X_train_G, y_train_G)

#  Évaluation finale
y_pred_G = modele_global.predict(X_test_G)
print("\n--- RAPPORT DE CLASSIFICATION FINAL ---")
print(classification_report(y_test_G, y_pred_G))

print(" --------Modèle entraîné avec succès sur l'ensemble des fichiers")

import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("---  COMPROMIS PARFAIT ET DASHBOARD GLOBAL ---")

# 1. Calcul des proba
probs_global = modele_global.predict_proba(X_test_G)[:, 1]

# 2. Recherche du meilleur seuil
precisions, recalls, thresholds = precision_recall_curve(y_test_G, probs_global)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) # Éviter division par 0
seuil_parfait_G = thresholds[np.argmax(f1_scores)]

# 3. Application du nouveau seuil
y_pred_final_G = (probs_global >= seuil_parfait_G).astype(int)
cm_final = confusion_matrix(y_test_G, y_pred_final_G)

TN, FP, FN, TP = cm_final[0][0], cm_final[0][1], cm_final[1][0], cm_final[1][1]
nouv_precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
nouv_recall = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0


print(f" Seuil d'équilibre trouvé : {seuil_parfait_G:.4f}")
print(f" Faux Positifs (Fausses alertes) : {FP}")
print(f" Attaques ratées (Faux Négatifs) : {FN}")
print(f" Attaques bloquées (Vrais Positifs) : {TP}")

print("\n---  GÉNÉRATION DU DASHBOARD MULTI-ATTAQUES ---")

# 4. Création du Dashboard (Adapté pour les résultats globaux)
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Volume : Trafic vs Alertes", "Taux de Détection (Recall %)", "Matrice de Confusion Globale", "Score de Risque (300 msgs)"),
    specs=[[{"type": "domain"}, {"type": "indicator"}],
           [{"type": "table"}, {"type": "scatter"}]]
)

# Graphique Donut
fig.add_trace(go.Pie(
    labels=['Trafic Normal', 'Alertes Déclenchées'], 
    values=[TN+FN, FP+TP], 
    hole=.5, marker_colors=['#00CC96', '#EF553B'],
    textposition='inside' 
), row=1, col=1)

# Jauge (Taux de détection - Recall)
fig.add_trace(go.Indicator(
    mode="number+gauge", value=nouv_recall,
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkgreen"},
           'steps': [{'range': [0, 80], 'color': "lightgray"}, {'range': [80, 100], 'color': "lightgreen"}]}
), row=1, col=2)

# Tableau (Matrice de confusion)
fig.add_trace(go.Table(
    header=dict(values=['', 'Prédit Normal (Légitime)', 'Prédit Attaque (Alerte)'], fill_color='paleturquoise', align='center'),
    cells=dict(values=[['Vrai Normal', 'Vraie Attaque'], [TN, FN], [FP, TP]], fill_color='lavender', align='center')
), row=2, col=1)

# Graphique en ligne (Score d'anomalie sur un échantillon)
fig.add_trace(go.Scatter(y=probs_global[:300], mode='lines', name="Score Modèle", line=dict(color='orange')), row=2, col=2)
fig.add_trace(go.Scatter(
    x=[0, 300], y=[seuil_parfait_G, seuil_parfait_G], 
    mode='lines', name="Seuil Optimal", 
    line=dict(color='red', dash='dash')
), row=2, col=2)

# Mise en page finale
fig.update_layout(
    title_text=" X-CANIDS Dashboard - IDS Multi-Attaques ", 
    height=750,
    margin=dict(t=120, b=40, l=40, r=40)
)
fig.show()

print("-----------------Félicitations -- Le système IDS global est terminé et opérationnel.")



#  ENTRAÎNEMENT DU MODÈLE GLOBAL 


X = df_test.drop(columns=['label'])    #X = Signaux
y = df_test['label']   #y = label d'attaque

# Nettoyage rapide 
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Division Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" Lancement de l'entraînement sur {len(X_train)} messages CAN...")

model_global = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1 # Utilise tous les cœurs de ton processeur
)

model_global.fit(X_train, y_train)


joblib.dump(model_global, "x_canids_model.pkl")
print("  Modèle sauvegardé sous 'x_canids_model.pkl'")
