import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from xgboost import XGBClassifier, XGBRegressor

# ================================
# 1️⃣ Chargement des données
# ================================
TRAIN_TEST_PATH = "data/train_test/"
CLEAN_PATH      = "data/processed/dataset_cleaned.csv"
MODELS_PATH     = "models/"
REPORTS_PATH    = "reports/"

os.makedirs(MODELS_PATH,  exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

X_train = pd.read_csv(TRAIN_TEST_PATH + "X_train.csv")
X_test  = pd.read_csv(TRAIN_TEST_PATH + "X_test.csv")
y_train = pd.read_csv(TRAIN_TEST_PATH + "y_train.csv").squeeze()
y_test  = pd.read_csv(TRAIN_TEST_PATH + "y_test.csv").squeeze()

# On charge AUSSI les données brutes (non normalisées) pour le clustering
df_clean = pd.read_csv(CLEAN_PATH)

print(f"[OK] Données chargées")
print(f"     X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"     df_clean (brutes) : {df_clean.shape}")

# ================================
# 2️⃣ CLUSTERING K-MEANS (CORRIGÉ)
# ================================
# ✅ FIX : On utilise les 3 features RFM sur données BRUTES
# avec un scaler dédié pour éviter que tout soit dans un seul cluster
print("\n" + "="*50)
print("PARTIE 1 — CLUSTERING K-MEANS (corrigé)")
print("="*50)

# Récupérer les 3 features RFM depuis les données brutes
rfm_cols_full = ['Recency', 'Frequency', 'MonetaryTotal']
rfm_available = [c for c in rfm_cols_full if c in df_clean.columns]
print(f"     Features RFM utilisées : {rfm_available}")

X_rfm_raw = df_clean[rfm_available].copy()

# ✅ FIX : Retirer les outliers extrêmes qui créent des mini-clusters
# (ces 10 clients ultra-riches faussaient tout)
for col in rfm_available:
    q99 = X_rfm_raw[col].quantile(0.99)
    X_rfm_raw[col] = X_rfm_raw[col].clip(upper=q99)

# ✅ FIX : Scaler DÉDIÉ au clustering (pas celui du classificateur)
scaler_rfm = StandardScaler()
X_rfm_scaled = scaler_rfm.fit_transform(X_rfm_raw)

# Méthode Elbow
inertias = []
silhouettes = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_rfm_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_rfm_scaled, km.labels_)
    silhouettes.append(sil)
    print(f"  k={k} → inertie={km.inertia_:.0f} | silhouette={sil:.4f}")

# Double graphique : Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(list(k_range), inertias, 'o-', color='steelblue', linewidth=2)
axes[0].set_xlabel("Nombre de clusters (k)")
axes[0].set_ylabel("Inertie")
axes[0].set_title("Méthode Elbow")
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), silhouettes, 'o-', color='orange', linewidth=2)
axes[1].set_xlabel("Nombre de clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Score de Silhouette")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_PATH + "elbow_curve.png", dpi=150)
plt.close()
print(f"[OK] Graphique Elbow+Silhouette → {REPORTS_PATH}elbow_curve.png")

# ✅ Choix k=4 pour avoir 4 segments métier clairs
K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans.fit(X_rfm_scaled)

sil_score = silhouette_score(X_rfm_scaled, kmeans.labels_)
print(f"\n[OK] K-Means k={K_OPTIMAL} | Silhouette={sil_score:.4f}")

cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
print(f"     Distribution : {cluster_counts.to_dict()}")

# Profil moyen par cluster (sur données brutes pour interprétabilité)
X_rfm_profile = X_rfm_raw.copy()
X_rfm_profile['Cluster'] = kmeans.labels_
profile = X_rfm_profile.groupby('Cluster')[rfm_available].mean().round(2)
print(f"\n     Profil moyen par cluster (valeurs brutes) :")
print(profile.to_string())

# Assignation automatique des noms de segments selon les profils
# Champions = haute Frequency + haute Monetary + basse Recency
# VIP = ultra haute Monetary
# Réguliers = moyens partout
# Dormants = haute Recency + basse Frequency
print(f"\n     Interprétation métier suggérée :")
for cluster_id in profile.index:
    r = profile.loc[cluster_id, 'Recency'] if 'Recency' in profile.columns else 0
    f = profile.loc[cluster_id, 'Frequency']
    m = profile.loc[cluster_id, 'MonetaryTotal']
    print(f"       Cluster {cluster_id}: R={r:.0f}j, F={f:.1f}, M={m:.0f}£ "
          f"({cluster_counts[cluster_id]} clients)")

# Visualisation (Recency vs MonetaryTotal)
plt.figure(figsize=(9, 6))
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
for i in range(K_OPTIMAL):
    mask = kmeans.labels_ == i
    plt.scatter(
        X_rfm_raw.loc[mask, 'Recency'] if 'Recency' in X_rfm_raw.columns else X_rfm_raw.iloc[mask, 0],
        X_rfm_raw.loc[mask, 'MonetaryTotal'],
        c=colors[i], label=f'Cluster {i} (n={mask.sum()})',
        alpha=0.5, s=25, edgecolors='white', linewidth=0.3
    )
plt.xlabel("Recency (jours)")
plt.ylabel("MonetaryTotal (£)")
plt.title(f"Segmentation K-Means (k={K_OPTIMAL}) — Silhouette={sil_score:.3f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_PATH + "kmeans_clusters.png", dpi=150)
plt.close()
print(f"[OK] Visualisation clusters → {REPORTS_PATH}kmeans_clusters.png")

# ✅ Sauvegarder kmeans + scaler_rfm + colonnes
joblib.dump(kmeans, MODELS_PATH + "kmeans_model.pkl")
joblib.dump(scaler_rfm, MODELS_PATH + "scaler_rfm.pkl")
joblib.dump(rfm_available, MODELS_PATH + "rfm_features.pkl")
print(f"[OK] K-Means + scaler RFM sauvegardés")

# ================================
# 3️⃣ SUPPRESSION COLONNES LEAKAGE
# ================================
LEAKAGE_COLS = [
    'ChurnRiskCategory',
    'LoyaltyLevel',
    'RFMSegment_Dormants',
    'RFMSegment_Fidèles',
    'RFMSegment_Potentiels',
    'CustomerType_Perdu',
    'CustomerType_Nouveau',
    'CustomerType_Occasionnel',
    'CustomerType_Régulier',
    'CustomerType_Hyperactif',
    'Recency',
    'CustomerTenureDays',
    'SpendingCategory',
    'PreferredMonth',
    'FavoriteSeason_Printemps',
]

leakage_present = [c for c in LEAKAGE_COLS if c in X_train.columns]
X_train_clf = X_train.drop(columns=leakage_present)
X_test_clf  = X_test.drop(columns=leakage_present)

print(f"\n[OK] {len(leakage_present)} colonnes leakage supprimées pour classification")
print(f"     X_train_clf={X_train_clf.shape} | X_test_clf={X_test_clf.shape}")

# ================================
# 4️⃣ CLASSIFICATION — Prédire le Churn
# ================================
print("\n" + "="*50)
print("PARTIE 2 — CLASSIFICATION (Prédiction Churn)")
print("="*50)

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42, C=0.1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2,
        random_state=42, eval_metric='logloss', verbosity=0
    ),
}

best_clf_name  = None
best_clf_f1    = 0
best_clf_model = None
results_clf    = {}

for name, clf in classifiers.items():
    clf.fit(X_train_clf, y_train)
    y_pred = clf.predict(X_test_clf)
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    
    # ✅ Validation croisée
    cv_scores = cross_val_score(clf, X_train_clf, y_train, cv=5,
                                 scoring='f1_weighted', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()
    
    results_clf[name] = {
        'accuracy': acc, 'f1': f1,
        'cv_f1_mean': cv_mean, 'cv_f1_std': cv_std
    }
    
    print(f"\n--- {name} ---")
    print(f"  Accuracy test    : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 test          : {f1:.4f}")
    print(f"  F1 CV (5-folds)  : {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(classification_report(y_test, y_pred,
                                target_names=['Fidèle (0)', 'Churné (1)']))
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Fidèle', 'Churné']).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f"Confusion — {name}")
    plt.tight_layout()
    safe_name = name.replace(' ', '_').lower()
    plt.savefig(REPORTS_PATH + f"confusion_{safe_name}.png", dpi=150)
    plt.close()
    
    if f1 > best_clf_f1:
        best_clf_f1    = f1
        best_clf_name  = name
        best_clf_model = clf

print(f"\n[OK] Meilleur classificateur : {best_clf_name} (F1={best_clf_f1:.4f})")

# Feature importance
if best_clf_name in ["Random Forest", "XGBoost"]:
    importances = best_clf_model.feature_importances_
    feat_df = pd.DataFrame({
        'feature':    X_train_clf.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_df, y='feature', x='importance',
                hue='feature', palette='viridis', legend=False)
    plt.title(f"Top 15 features — {best_clf_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH + "feature_importance_classification.png", dpi=150)
    plt.close()
    print(f"[OK] Feature importance → {REPORTS_PATH}feature_importance_classification.png")
    print(f"\n     Top 5 features :")
    print(feat_df[['feature', 'importance']].head(5).to_string(index=False))

# ✅ Sauvegarder modèle + liste de features
joblib.dump(best_clf_model, MODELS_PATH + "best_classifier.pkl")
joblib.dump(X_train_clf.columns.tolist(), MODELS_PATH + "clf_features.pkl")
print(f"[OK] Classificateur + features sauvegardés")

# ================================
# 5️⃣ RÉGRESSION (CORRIGÉ)
# ================================
# ✅ FIX : On utilise X_train/X_test NORMALISÉS (cohérent avec predict.py)
# et on reconstruit y_reg depuis df_clean via l'index
print("\n" + "="*50)
print("PARTIE 3 — RÉGRESSION (Prédiction MonetaryTotal) — CORRIGÉE")
print("="*50)

# On prépare y_reg (cible = MonetaryTotal brute) en suivant le même split
# On refait le split avec le même random_state pour garantir la cohérence
df_for_reg = df_clean.copy()

# Retirer Churn et CustomerID de df_clean pour avoir le même X
cols_to_drop_reg = ['Churn']
if 'CustomerID' in df_for_reg.columns:
    cols_to_drop_reg.append('CustomerID')

X_full = df_for_reg.drop(columns=cols_to_drop_reg)
y_churn_full = df_for_reg['Churn']

# Même split que preprocessing.py (random_state=42, stratify)
from sklearn.model_selection import train_test_split as tts
X_tr_raw, X_te_raw, y_tr_ch, y_te_ch = tts(
    X_full, y_churn_full, test_size=0.2, random_state=42, stratify=y_churn_full
)

# y_reg = MonetaryTotal correspondant à chaque ligne
y_reg_train = X_tr_raw['MonetaryTotal'].values
y_reg_test  = X_te_raw['MonetaryTotal'].values

# Features pour la régression (on retire toutes les colonnes Monetary + target)
monetary_cols = ['MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
                 'MonetaryMin', 'MonetaryMax']

# On utilise les X normalisés existants, en retirant les colonnes Monetary
reg_features = [c for c in X_train.columns if c not in monetary_cols]

X_reg_train = X_train[reg_features].copy()
X_reg_test  = X_test[reg_features].copy()

# Vérifier cohérence des tailles
assert len(X_reg_train) == len(y_reg_train), "Mismatch train"
assert len(X_reg_test) == len(y_reg_test), "Mismatch test"

print(f"[OK] X_reg_train={X_reg_train.shape} | X_reg_test={X_reg_test.shape}")
print(f"     y_reg stats: min={y_reg_train.min():.0f}, "
      f"median={np.median(y_reg_train):.0f}, max={y_reg_train.max():.0f}")

# ✅ Log-transform de y pour stabiliser les outliers extrêmes
# MonetaryTotal a des valeurs de 0 à 280k£ → skew très fort
y_reg_train_log = np.log1p(np.maximum(y_reg_train, 0))
y_reg_test_log  = np.log1p(np.maximum(y_reg_test, 0))

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    ),
}

best_reg_name  = None
best_reg_r2    = -np.inf
best_reg_model = None
results_reg    = {}

for name, reg in regressors.items():
    # Entraîner sur log(y)
    reg.fit(X_reg_train, y_reg_train_log)
    
    # Prédire et repasser en échelle originale
    y_pred_log = reg.predict(X_reg_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)  # pas de montants négatifs
    
    mae  = mean_absolute_error(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2   = r2_score(y_reg_test, y_pred)
    
    results_reg[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    print(f"\n--- {name} ---")
    print(f"  MAE  : {mae:.2f} £")
    print(f"  RMSE : {rmse:.2f} £")
    print(f"  R²   : {r2:.4f}")
    
    if r2 > best_reg_r2:
        best_reg_r2    = r2
        best_reg_name  = name
        best_reg_model = reg

print(f"\n[OK] Meilleur régresseur : {best_reg_name} (R²={best_reg_r2:.4f})")

# Visualisation
y_pred_best = np.expm1(best_reg_model.predict(X_reg_test))
y_pred_best = np.maximum(y_pred_best, 0)

plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_pred_best, alpha=0.4, s=18, color='steelblue')
max_val = float(max(y_reg_test.max(), y_pred_best.max()))
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Prédiction parfaite')
plt.xlabel("Valeur réelle (£)")
plt.ylabel("Valeur prédite (£)")
plt.title(f"Régression — {best_reg_name} (R²={best_reg_r2:.3f})")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig(REPORTS_PATH + "regression_pred_vs_real.png", dpi=150)
plt.close()
print(f"[OK] Graphique → {REPORTS_PATH}regression_pred_vs_real.png")

# ✅ Sauvegarder modèle + features + flag log
joblib.dump(best_reg_model, MODELS_PATH + "best_regressor.pkl")
joblib.dump(reg_features, MODELS_PATH + "reg_features.pkl")
joblib.dump(True, MODELS_PATH + "reg_uses_log.pkl")  # flag pour predict.py
print(f"[OK] Régresseur + features sauvegardés")

# ================================
# 6️⃣ RÉSUMÉ FINAL
# ================================
print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)
print(f"  Clustering    : K-Means k={K_OPTIMAL} | Silhouette={sil_score:.4f}")
print(f"                  Distribution : {cluster_counts.to_dict()}")

for name, scores in results_clf.items():
    print(f"  {name:<25} | Acc={scores['accuracy']:.4f} | "
          f"F1={scores['f1']:.4f} | CV={scores['cv_f1_mean']:.4f}")

print(f"  Meilleur classif. : {best_clf_name}")
print(f"  Meilleur régression : {best_reg_name} | R²={best_reg_r2:.4f}")
print(f"\n  Modèles → {MODELS_PATH}")
print(f"  Rapports → {REPORTS_PATH}")
print("\n✅ train_model.py terminé avec succès !")