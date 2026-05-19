from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

print("faire l'optimisation  ")


param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

clf = xgb.XGBClassifier(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=5, scoring='f1', cv=3, verbose=1)


search.fit(X_train_G, y_train_G)

modele_final = search.best_estimator_
print(f"---   Meilleurs paramètres trouvés : {search.best_params_}")
