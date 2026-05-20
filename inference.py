# 
# GÉNÉRATEUR DE BILAN DE SÉCURITÉ AUTOMATISÉ
# Ce script scanne l'ensemble des dumps CAN (fichiers .parquet) pour produire 
# un diagnostic global. Il fait le lien entre l'anomalie détectée et la 
# contre-mesure technique recommandée (Watchdog, SecOC, Rate Limiting).


import pandas as pd
import glob
import gc 
# --- SCRIPT D'ANALYSE AUTOMATISÉE DES DUMPS CAN ---


path_fichiers = r"C:\Users\elala\Downloads\Projet\Signal\*.parquet"
liste_fichiers = glob.glob(path_fichiers)


bilan_securite = []

print(f"Démarrage de l'analyse sur {len(liste_fichiers)} fichiers trouvés...")

for i, chemin in enumerate(liste_fichiers):
    nom = chemin.split("\\")[-1]
    print(f"En cours : {nom} ({i+1}/{len(liste_fichiers)})")
    
    # Lecture du fichier=
    data = pd.read_parquet(chemin)
    
    # Calcul du timing entre les messages (critique pour détecter les injections)
    if 'delta_t' in data.columns:
        ecart_temps = data['delta_t']
    elif 'timestamp' in data.columns:
        # Si delta_t est absent, on le calcule par rapport à l'index
        data = data.sort_values('timestamp')
        ecart_temps = data['timestamp'].diff().fillna(0)
    else:
        ecart_temps = pd.Series([0.5]) 

    # --- LOGIQUE DE DIAGNOSTIC ---
 
    resultat = ""
    solution = ""
    
    if ecart_temps.min() < 0.0001:
        resultat = "Attaque type DoS / Flood (Fréquence trop élevée)"
        solution = "Filtrage par Gateway (Rate Limiting)"
    elif ecart_temps.max() > 1.5:
        resultat = "Suspension de messages (Calculateur muet)"
        solution = "Implémentation d'un Watchdog"
    else:
        resultat = "Fuzzing ou Masquerade (Payload suspect)"
        solution = "Protocole SecOC (Signature/MAC)"
        
    bilan_securite.append({
        "Nom du Fichier": nom, 
        "Diagnostic Détecté": resultat, 
        "Mesure de Sécurité": solution
    })
    
    # Nettoyage pour éviter crache
    del data
    gc.collect()

# --- AFFICHAGE 
print("\n" + "-"*50)
print("RÉCAPITULATIF DES RÉSULTATS - PROJET X-CANIDS")
print("-"*50)
resultats_df = pd.DataFrame(bilan_securite)
print(resultats_df.to_string(index=False))
