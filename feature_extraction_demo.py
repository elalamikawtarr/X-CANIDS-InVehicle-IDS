# EXTRACTION DES FEATURES 
# Ce module transforme les trames CAN brutes en vecteurs statistiques.
# L'objectif est de capturer le comportement temporel (Intervalles) et la 
# variabilité des données pour permettre la détection par Machine Learning.

import pandas as pd
import glob


chemin_dossier = r"C:\Users\elala\Downloads\Projet\Signal\*.parquet"
fichiers = glob.glob(chemin_dossier)

if fichiers:
    df_brut = pd.read_parquet(fichiers[0])
    
    print("---  COLONNES DU DATASET BRUT (.parquet) ---")
    print(df_brut.columns.tolist())
    print("\n---  APERÇU DES 5 PREMIÈRES LIGNES ---")
    print(df_brut.head())

    #  Démonstration de l'extraction des FEATURES
    # On simule l'extraction pour voir à quoi ressemble le Dataset "IA"
    if 'timestamp' not in df_brut.columns and 'time' in df_brut.columns:
        df_brut['timestamp'] = df_brut['time']

    delta_t = df_brut['timestamp'].diff().fillna(0) if 'timestamp' in df_brut.columns else [0]
    
    features_demo = pd.DataFrame([{
        "arbitration_id": df_brut['arbitration_id'].iloc[0] if 'arbitration_id' in df_brut.columns else "N/A",
        "messages_count": len(df_brut),
        "interval_mean": delta_t.mean() if not isinstance(delta_t, list) else 0,
        "interval_std": delta_t.std() if not isinstance(delta_t, list) else 0,
        "uniq_data_count": df_brut['data'].nunique() if 'data' in df_brut.columns else "N/A",
    }])

    print("\n" + "="*50)
    print(" VOTRE DATASET DE FEATURES (PRÊT POUR L'IA)")
    print("="*50)
    print(features_demo.to_string(index=False))
else:
    print("Aucun fichier trouvé pour la démonstration.")
