import pandas as pd
import glob
import gc

chemin_fichiers = r"C:\Users\elala\Downloads\Projet\Signal\*.parquet"
fichiers = glob.glob(chemin_fichiers)

dfs_propres = []

for f in fichiers:
    nom = f.split("\\")[-1]
    print(f" Traitement de : {nom}...")
    
    # 1. Chargement d'un seul fichier
    df = pd.read_parquet(f)
    df = df.ffill().fillna(0)
    
    # 2. Définition du Label
    col_label = 'label'
    if col_label not in df.columns and 'Label' in df.columns:
        col_label = 'Label'
    elif col_label not in df.columns:
        df[col_label] = 0 if 'dump1' in nom.lower() else 1
        
    # 3. LE SECRET ANTI-CRASH : Sous-échantillonnage précoce
    attaques = df[df[col_label] == 1]
    normaux = df[df[col_label] == 0]
    
    # On garde 100% des attaques, mais seulement 5% du trafic normal pour sauver la RAM
    if len(normaux) > 0:
        normaux = normaux.sample(frac=0.05, random_state=42)
        
    df_reduit = pd.concat([attaques, normaux])
    
    # 4. Calcul du dt_spike sur le fichier réduit
    if 'timestamp' in df_reduit.columns:
        df_reduit = df_reduit.sort_values('timestamp')
        df_reduit['dt_spike'] = df_reduit['timestamp'].diff().dt.total_seconds().fillna(0)
        
    # 5. Optimisation finale (Float32)
    for col in df_reduit.select_dtypes(include=['float64']).columns:
        df_reduit[col] = df_reduit[col].astype('float32')

    dfs_propres.append(df_reduit)
    
    # Nettoyage absolu de la mémoire 
    del df
    del attaques
    del normaux
    gc.collect()

# 6. Fusion Finale
print("\n Fusion verticale des données réduites...")
dataset_final = pd.concat(dfs_propres, axis=0, ignore_index=True)

print("--Tri chronologique final...")
if 'timestamp' in dataset_final.columns:
    dataset_final = dataset_final.sort_values('timestamp')
elif isinstance(dataset_final.index, pd.DatetimeIndex):
    dataset_final = dataset_final.sort_index()

print(f" DATASET PRÊT : {dataset_final.shape[0]} lignes .")
print(f" Répartition : {dataset_final[col_label].value_counts(normalize=True)[1]*100:.2f}% d'attaques.")



# ------ pour les données cloud
from sklearn.preprocessing import MinMaxScaler
import joblib

def scale_signals_xcanids(df, save_path="models/xcanids_scaler.pkl"):
    #  Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    #  Initialisation du scaler (Normalisation 0-1)
    scaler = MinMaxScaler()
    
    # Fit (Apprentissage des min/max) et Transform
    scaled_data = scaler.fit_transform(numeric_df)
    
    # SAUVEGARDE : Crucial pour l'inférence Cloud/Temps Réel
    joblib.dump(scaler, save_path)
    print(f" Scaler sauvegardé sous {save_path}")
    
    return scaled_data, numeric_df.columns

