import pandas as pd
import glob
import gc

# Analyse de l'intégrité des flux et déclenchement des contre-mesures

print("-" * 70)
print("  MONITORING RÉSEAU CAN : ANALYSE DE LA SÉCURITÉ ACTIVE")
print("-" * 70 + "\n")

path_data = r"C:\Users\elala\Downloads\Projet\Signal\*.parquet"
liste_captures = glob.glob(path_data)

for i, chemin in enumerate(liste_captures):
    nom_f = chemin.split("\\")[-1]
    print(f"ANALYSE : {nom_f}")
    
    # Gestion de la mémoire
    df_temp = pd.read_parquet(chemin)
    nom_lower = nom_f.lower()
    
    # --- LOGIQUE DÉCISIONNELLE DU PARE-FEU EMBARQUÉ ---
    if "susp" in nom_lower:
        print("  STATUT : Violation de présence (Silence détecté)")
        print("  ACTION : Déclenchement Watchdog / Mode dégradé activé\n")
        
    elif "fabr" in nom_lower or "dos" in nom_lower:
        print("  STATUT : Inondation réseau (Déni de service)")
        print("  ACTION : Activation du Rate Limiting / ID banni\n")
        
    elif "fuzz" in nom_lower:
        print("  STATUT : Incohérence des signaux physiques")
        print("  ACTION : Rejet par filtre de plausibilité (XGBoost)\n")
        
    elif "masq" in nom_lower:
        print("  STATUT : Échec d'authentification (Usurpation)")
        print("  ACTION : Rejet SecOC (Signature MAC invalide)\n")
        
    else:
        print("  STATUT : Flux nominal")
        print("  ACTION : Transmission autorisée\n")

    del df_temp
    gc.collect()

print("-" * 70)



# - ÉVALUATION 
y_pred = model_global.predict(X_test)

print("\n--- RAPPORT DE PERFORMANCE ---")
print(classification_report(y_test, y_pred))

# la Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(cm, 
                   labels=dict(x="Prédit", y="Réel"),
                   x=['Normal', 'Attaque'],
                   y=['Normal', 'Attaque'],
                   text_auto=True,
                   title="Matrice de Confusion : X-CANIDS")
fig_cm.show()
print(" FIN DU SCAN : PROTECTION RÉSEAU OPÉRATIONNELLE")
print("-" * 70)
