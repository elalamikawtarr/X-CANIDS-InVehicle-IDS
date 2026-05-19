import pandas as pd
import plotly.express as px

print("--- EXPLICABILITÉ DU MODÈLE  ")

# 1. Récupérer l'importance calculée par XGBoost
importances = modele_global.feature_importances_
colonnes = X_train_G.columns

# 2. Créer un tableau et le trier pour garder les 15 plus importants
df_importance = pd.DataFrame({'Feature': colonnes, 'Importance': importances})
df_importance = df_importance.sort_values(by='Importance', ascending=True).tail(15)

# 3. Création du graphique interactif
fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
             title="Les signaux les plus importants pour la détection (X-CANIDS)",
             color='Importance', color_continuous_scale='Reds')

fig.update_layout(height=600, margin=dict(l=20, r=20, t=50, b=20))
fig.show()

print(" ---   Graphique généré ----- ")
