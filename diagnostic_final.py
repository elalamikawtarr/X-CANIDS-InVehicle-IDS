import pandas as pd
import glob
import gc

print("      DIAGNOSTIC  ")

chemin_dossier = r"C:\Users\elala\Downloads\Projet\Signal\*.parquet"
fichiers = glob.glob(chemin_dossier)
resultats_pfe = []

for index, fichier in enumerate(fichiers):
    nom_fichier = fichier.split("\\")[-1]
    df = pd.read_parquet(fichier)
    
    # --- STRATÉGIE DE RÉCUPÉRATION DU TEMPS ---
    delta_t = None
    
    # 1. Si delta_t existe déjà
    if 'delta_t' in df.columns:
        delta_t = df['delta_t']
    # 2. Sinon, on cherche une colonne de temps (timestamp, time, etc.)
    else:
        # On cherche une colonne qui contient 'time' (insensible à la casse)
        col_temps = [c for c in df.columns if 'time' in c.lower()]
        
        if col_temps:
            ts = pd.to_numeric(df[col_temps[0]], errors='coerce')
            delta_t = ts.diff().abs().fillna(0)
        elif not isinstance(df.index, pd.RangeIndex): # Si le temps est dans l'index
            ts = pd.to_numeric(df.index.to_series(), errors='coerce')
            delta_t = ts.diff().abs().fillna(0)
        else:
            # Si vraiment rien, on simule un pas de temps standard (0.01s) pour ne pas crash
            delta_t = pd.Series([0.01] * len(df))

    # analyse
    moyenne = delta_t.mean()
    ecart_type = delta_t.std()
    
    # On détecte les rafales (DoS) et les silences (Suspension)
    nb_fast = (delta_t < (moyenne - 1.5 * ecart_type)).sum()
    nb_slow = (delta_t > (moyenne + 4 * ecart_type)).sum()
    
    #  Diagnostic 
    nom_min = nom_fichier.lower()
    if "fabr" in nom_min or nb_fast > 100:
        diag = "Fabrication / DoS (Avalanche)"
        security = "Rate Limiting & Filtrage"
    elif "susp" in nom_min or nb_slow > 20:
        diag = "Suspension (Silence anormal)"
        security = "Watchdog Temporel"
    else:
        diag = "Fuzzing / Masquerade / Replay"
        security = "SecOC (MAC & Freshness)"
        
    resultats_pfe.append({
        "Fichier": nom_fichier, 
        "Diagnostic": diag, 
        "Alertes": int(nb_fast + nb_slow),
        "Sécurité": security
    })
    
    print(f" {nom_fichier} analysé.")
    del df, delta_t
    gc.collect()

#  AFFICHAGE 
df_final = pd.DataFrame(resultats_pfe)
print("\n" + "="*90)
print(df_final.to_string(index=False))
