import pandas as pd
import glob
import gc

print("    DIAGNOSTIC AVANCÉ PAR FENÊTRAGE (X-CANIDS V5) ---")

chemin_dossier = r"C:\Users\elala\Downloads\Projet\Signal\*.parquet"
fichiers = glob.glob(chemin_dossier)

resultats_pfe = []

for index, fichier in enumerate(fichiers):
    nom_fichier = fichier.split("\\")[-1]
    df = pd.read_parquet(fichier)
    
    # --- Calcul du  DELTA_T ---
    if 'delta_t' not in df.columns and 'timestamp' in df.columns:

        
        temp_ts = pd.to_numeric(df['timestamp'], errors='coerce')
        delta_t = temp_ts.diff().fillna(0)
    elif 'delta_t' in df.columns:
        delta_t = df['delta_t']
    else:
        delta_t = pd.Series([0.1] * len(df)) # Valeur par défaut
    
    # --- analyse
    # Fenêtre de 100 messages pour détecter les rafales 
    window_size = 100
    min_locaux = delta_t.rolling(window=window_size).min()
    max_locaux = delta_t.rolling(window=window_size).max()
    
    # --- DIAGNOSTIC 
    # On compte combien de fois l'anomalie apparaît
    nb_anomalies_fast = (min_locaux < 0.0001).sum()
    nb_anomalies_slow = (max_locaux > 1.5).sum()
    
    if nb_anomalies_fast > 5:
        diag = "Fabrication / DoS (Avalanche de messages)"
        security = "Rate Limiting & Filtrage Gateway"
    elif nb_anomalies_slow > 2:
        diag = "Suspension (Silence anormal)"
        security = "Watchdog Temporel"
    else:
        diag = "Fuzzing / Masquerade / Replay (Anomalie de données)"
        security = "SecOC (Authentification MAC & Freshness)"
        
    resultats_pfe.append({
        "Fichier": nom_fichier, 
        "Diagnostic": diag, 
        "Alertes Detectées": max(nb_anomalies_fast, nb_anomalies_slow),
        "Sécurité Recommandée": security
    })
    
    # Nettoyage strict
    del df, delta_t, min_locaux, max_locaux
    gc.collect()
    print(f" {nom_fichier} analysé.")

# Affichage final
df_final = pd.DataFrame(resultats_pfe)
print("\n" + "="*80)
print(df_final.to_string(index=False))
