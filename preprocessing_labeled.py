# MODULE D'AUDIT QUALITÉ ET PRÉPARATION FINALE 
# Ce bloc assure l'intégrité du flux de données avant l'inférence.
# Il traite les incohérences temporelles, les doublons et convertit les types
# de données pour garantir la compatibilité avec les algorithmes de Gradient Boosting.
def verifier_qualite(df, nom_fichier):
    print(f"--- Audit Qualité : {nom_fichier} ---")
    
    # 1. Vérification des valeurs manquantes
    manquants = df.isnull().sum().sum()
    if manquants > 0:
        print(f" Attention : {manquants} valeurs manquantes détectées.")
    
    # 2. Vérification de la cohérence temporelle (Pas de temps négatif)
    if 'delta_t' in df.columns:
        negatifs = (df['delta_t'] < 0).sum()
        if negatifs > 0:
            print(f" Erreur : {negatifs} timestamps incohérents (négatifs).")

    # 3. Vérification des doublons (Messages CAN identiques au même instant)
    doublons = df.duplicated().sum()
    if doublons > 0:
        print(f" Doublons détectés : {doublons} lignes.")
        
    return df.dropna().drop_duplicates() # Nettoyage rapide

print("--- Analyse de la colonne Label ---")
repartition = df_test['label'].value_counts()
pourcentage = df_test['label'].value_counts(normalize=True) * 100

print(f"Messages Normaux (0) : {repartition[0]} ({pourcentage[0]:.2f}%)")
if 1 in repartition:
    print(f"Messages d'Attaque (1) : {repartition[1]} ({pourcentage[1]:.2f}%)")
else:
    print(" Attention : Aucune attaque détectée dans ce fichier ")

# 1. On sépare les features (X) et la cible (y)
X = df_test.drop(columns=['label'])
y = df_test['label']

# 2. Conversion : On s'assure que tout est numérique

for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(0)

print("\n Données nettoyées et formatées pour XGBoost")
