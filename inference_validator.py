# VALIDATION DE LA DIMENSIONNALITÉ ET INFÉRENCE IA
# Ce bloc assure la conformité entre les données d'entrée et le modèle chargé.
# Il vérifie que le nombre de signaux (dimensions) est identique à celui utilisé 
# lors de l'entraînement pour éviter les erreurs de calcul matriciel (Crash).

# Sélectionner uniquement le numérique
X_numeric = df_brut.select_dtypes(include=[np.number]).fillna(0)

#  vérification de la dimension 
expected_dim = autoencoder.input_shape[-1]
current_dim = X_numeric.shape[1]

print(f"Dimension attendue : {expected_dim}")
print(f"Dimension actuelle : {current_dim}")

if current_dim != expected_dim:
    print(f" Attention : Décalage de colonnes ! ({current_dim} au lieu de {expected_dim})")
    # Optionnel : réaligner si tu as la liste des colonnes d'origine
    # X_numeric = X_numeric.reindex(columns=original_feature_list, fill_value=0)

#  Normalisation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 4. Inférence (Prédiction)
S_pred = autoencoder.predict(X_scaled)
print(" Prédiction réussie ")
