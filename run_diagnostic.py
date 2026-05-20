# DIAGNOSTIC CIBLÉ : IDENTIFICATION DU SIGNAL SUSPECT
# Ce module utilise la distance entre le signal réel (X_test_scaled) et la 
# reconstruction de l'IA (S_pred). Le signal présentant l'erreur quadratique 
# moyenne (MSE) la plus élevée est identifié comme le vecteur de l'attaque
# Préparation des données (X_test_scaled)
X_test = df_brut.select_dtypes(include=[np.number])
X_test_scaled = scale_signals_xcanids(X_test)

#  Instanciation du modèle (Correction du NameError)
input_dim = X_test_scaled.shape[1]
autoencoder = build_x_canids_autoencoder(input_dim)

# Obtenir la reconstruction
print(" Analyse du trafic en cours...")
S_pred = autoencoder.predict(X_test_scaled)

# Calcul de l'erreur par signal
signalwise_error = np.mean(np.power(X_test_scaled - S_pred, 2), axis=0)

# Identifier le signal suspect
target_signal_idx = np.argmax(signalwise_error)
print(f"Attaque détectée sur le signal : {X_test.columns[target_signal_idx]}")
