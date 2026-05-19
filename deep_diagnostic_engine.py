import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # Correction de l'erreur NameError


def scale_signals(df):
    """DRY & Robust: Gère les valeurs manquantes pour éviter les 'nan'"""
    # Remplir les NaN par 0 avant de scaler pour éviter que le modèle explose
    df_clean = df.select_dtypes(include=[np.number]).fillna(0)
    scaler = MinMaxScaler()
    return scaler.fit_transform(df_clean)

def build_model(input_dim):
    """KISS: Structure avec initialisation stable"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    # Adam est bon, mais on s'assure que le learning rate n'est pas trop haut
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# --- EXÉCUTION (Pipeline Final) ---

try:
    # 1. Préparation
    X_scaled = scale_signals(df_brut)
    
    # 2. Construction
    autoencoder = build_model(X_scaled.shape[1])
    
    # 3. Entraînement (Essentiel pour éviter les fausses alertes)
    print(" Entraînement de l'IA (Patientez...)")
    autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=32, verbose=1)
    
    # 4. Diagnostic
    print(" Analyse en cours...")
    S_pred = autoencoder.predict(X_scaled)
    
    # 5. Calcul de l'erreur (Explicabilité X-CANIDS)
    errors = np.mean(np.power(X_scaled - S_pred, 2), axis=0)
    target_idx = np.argmax(errors)
    
    print("="*50)
    print(f" ANALYSE RÉUSSIE")
    print(f" Signal le plus suspect : {df_brut.select_dtypes(include=[np.number]).columns[target_idx]}")
    print("="*50)

except Exception as e:
    print(f"Erreur Critique : {e}")
