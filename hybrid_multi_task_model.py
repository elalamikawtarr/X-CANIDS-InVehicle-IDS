import tensorflow as tf
from tensorflow.keras import layers, models

def build_multi_attack_model(input_dim, num_classes):
    # --- PARTIE ENCODEUR (Compression) ---
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(encoded)
    latent_space = layers.Dense(8, activation='relu')(encoded) # Espace latent

    #  DÉTECTION (Reconstruction) ---
    decoded = layers.Dense(16, activation='relu')(latent_space)
    decoded = layers.Dense(32, activation='relu')(decoded)
    reconstruction = layers.Dense(input_dim, activation='sigmoid', name='reconstruction')(decoded)

    #  CLASSIFICATION (Type d'attaque) ---
    classification = layers.Dense(16, activation='relu')(latent_space)
    classification = layers.Dropout(0.2)(classification)
    attack_type = layers.Dense(num_classes, activation='softmax', name='classification')(classification)

    # Modèle à deux sorties
    model = models.Model(inputs=input_layer, outputs=[reconstruction, attack_type])
    
    model.compile(optimizer='adam', 
                  loss={'reconstruction': 'mse', 'classification': 'categorical_crossentropy'},
                  metrics={'classification': 'accuracy'})
    return model

# input_dim = 21 (tes colonnes), num_classes = 4 (Normal, Replay, Fuzzing, DoS)
model = build_multi_attack_model(21, 4)
