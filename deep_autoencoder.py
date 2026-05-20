# Archi DEEP LEARNING : AUTOENCODER POUR DÉTECTION D'ANOMALIES
# Ce modèle apprend à compresser (Encoder) et reconstruire  le signal normal
# Une erreur de reconstruction élevée indique une divergence par rapport au 



import tensorflow as tf
from tensorflow.keras import layers, models

def build_x_canids_autoencoder(input_dim):
    # Architecture basée sur le papier (Encoder-Decoder)
    model = models.Sequential([
        # ENCODER
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'), # Latent space (h)
        
        # DECODER
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid') # Reconstruction de S'
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model
