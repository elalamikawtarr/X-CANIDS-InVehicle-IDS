# ARCHITECTURE DU CLASSIFIEUR NEURONAL (X-CANIDS DEEP LEARNING)
# Ce modèle utilise une structure d'encodeur pour extraire les caractéristiques
# essentielles (Bottleneck) suivies d'une tête de classification Softmax.
# Objectif : Catégoriser avec précision l'attaque (Normal, Replay, Fuzzing, etc.)

import tensorflow as tf
from tensorflow.keras import layers, models

def build_pfe_model(input_dim, num_classes):
    # --- ENCODEUR ---
    # C'est ici que l'IA apprend à compresser les 21 features
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dense(32, activation='relu')(x)
    latent_space = layers.Dense(16, activation='relu', name='bottleneck')(x)

    # --- TÊTE DE CLASSIFICATION ---
    # Cette partie décide du type d'attaque SANS voir le nom du fichier
    y = layers.Dense(16, activation='relu')(latent_space)
    y = layers.Dropout(0.2)(y)
    output_layer = layers.Dense(num_classes, activation='softmax', name='class_output')(y)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# input_dim = 21, num_classes = 4 (0:Normal, 1:Replay, 2:Fuzzing, 3:Suspicious)
my_model = build_pfe_model(21, 4)
