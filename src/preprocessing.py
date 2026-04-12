import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

# ================================
# 1️⃣ Chargement des données
# ================================

DATA_PATH = "data/raw/dataset ml.csv"
OUTPUT_PATH = "data/processed/dataset_cleaned.csv"

if not os.path.exists(DATA_PATH):
    print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"[OK] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.head())
# ================================
# SUPPRESSION DES COLONNES LEAKAGE
# ================================
# Ces colonnes ont été calculées À PARTIR du Churn dans le dataset
# Les inclure = donner la réponse au modèle → 100% accuracy artificielle

## ================================
# SUPPRESSION DES COLONNES LEAKAGE
# ================================
leakage_cols = [
    'ChurnRiskCategory',      # ordinal 0=Faible→3=Critique, calculé depuis Churn
    'LoyaltyLevel',           # ordinal, directement lié au statut churn
    'RFMSegment_Dormants',    # Dormant = client churné par définition
    'RFMSegment_Fidèles',     # Fidèle = client non churné par définition
    'RFMSegment_Potentiels',  # segment lié au comportement churn
]

leakage_present = [c for c in leakage_cols if c in X_train.columns]
X_train = X_train.drop(columns=leakage_present)
X_test  = X_test.drop(columns=leakage_present)
print(f"[OK] {len(leakage_present)} colonnes leakage supprimées : {leakage_present}")
print(f"     X_train={X_train.shape} | X_test={X_test.shape}")
# ================================
# 2️⃣ Analyse initiale des données
# ================================

print(df.info())
print(df.describe(include='all'))

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Valeurs manquantes :")
print(missing if not missing.empty else "Aucune valeur manquante détectée")

nb_duplicates = df.duplicated().sum()
print(f"Nombre de doublons : {nb_duplicates}")

# ================================
# 3️⃣ Nettoyage des données
# ================================

df = df.drop_duplicates()
print(f"[OK] Doublons supprimés. Lignes restantes : {len(df)}")

# Imputation Age par la médiane (30% de NaN, robuste aux valeurs extrêmes)
if 'Age' in df.columns:
    nb_missing_age = df['Age'].isnull().sum()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    print(f"[OK] 'Age' : {nb_missing_age} NaN imputés par la médiane ({df['Age'].median():.1f})")
else:
    print("[AVERTISSEMENT] Colonne 'Age' introuvable.")

# Imputation AvgDaysBetweenPurchases par la médiane (79 NaN détectés)
if 'AvgDaysBetweenPurchases' in df.columns:
    nb_missing = df['AvgDaysBetweenPurchases'].isnull().sum()
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(df['AvgDaysBetweenPurchases'].median())
    print(f"[OK] 'AvgDaysBetweenPurchases' : {nb_missing} NaN imputés par la médiane")

# ================================
# 4️⃣ Valeurs sentinelles + aberrantes
# ================================

# --- SupportTicketsCount ---
# -1 et 999 = codes d'erreur → NaN → médiane → capping IQR
COL_TICKETS = 'SupportTicketsCount'
if COL_TICKETS in df.columns:
    nb_sentinel = df[COL_TICKETS].isin([-1, 999]).sum()
    df[COL_TICKETS] = df[COL_TICKETS].replace([-1, 999], np.nan)
    print(f"[OK] '{COL_TICKETS}' : {nb_sentinel} valeurs sentinelles (-1, 999) → NaN")

    nb_nan = df[COL_TICKETS].isnull().sum()
    df[COL_TICKETS] = df[COL_TICKETS].fillna(df[COL_TICKETS].median())
    print(f"[OK] '{COL_TICKETS}' : {nb_nan} NaN imputés par la médiane ({df[COL_TICKETS].median():.1f})")

    Q1, Q3 = df[COL_TICKETS].quantile(0.25), df[COL_TICKETS].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    nb_outliers = ((df[COL_TICKETS] < lower) | (df[COL_TICKETS] > upper)).sum()
    df[COL_TICKETS] = df[COL_TICKETS].clip(lower, upper)
    print(f"[OK] '{COL_TICKETS}' : {nb_outliers} outliers → capping [{lower:.2f}, {upper:.2f}]")
else:
    print(f"[AVERTISSEMENT] Colonne '{COL_TICKETS}' introuvable.")

# --- SatisfactionScore ---
# -1, 0, 99 = codes d'erreur. Valeurs valides : 1 à 5
COL_SATIS = 'SatisfactionScore'
if COL_SATIS in df.columns:
    nb_sentinel_s = df[COL_SATIS].isin([-1, 0, 99]).sum()
    df[COL_SATIS] = df[COL_SATIS].replace([-1, 0, 99], np.nan)
    print(f"[OK] '{COL_SATIS}' : {nb_sentinel_s} valeurs sentinelles (-1, 0, 99) → NaN")

    nb_nan_s = df[COL_SATIS].isnull().sum()
    df[COL_SATIS] = df[COL_SATIS].fillna(df[COL_SATIS].median())
    print(f"[OK] '{COL_SATIS}' : {nb_nan_s} NaN imputés par la médiane ({df[COL_SATIS].median():.1f})")
else:
    print(f"[AVERTISSEMENT] Colonne '{COL_SATIS}' introuvable.")

# ================================
# 5️⃣ Formats inconsistants + Feature Engineering date
# ================================

if 'RegistrationDate' in df.columns:
    nb_before = df['RegistrationDate'].isnull().sum()
    df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"], format="mixed", dayfirst=True, errors='coerce')    
    nb_after = df['RegistrationDate'].isnull().sum()
    nb_failed = nb_after - nb_before

    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday

    # Supprimer la colonne date originale (remplacée par 4 features numériques)
    df = df.drop(columns=["RegistrationDate"])
    print(f"[OK] 'RegistrationDate' : features extraites + colonne supprimée ({nb_failed} NaT)")
else:
    print("[AVERTISSEMENT] Colonne 'RegistrationDate' introuvable.")

# ================================
# 6️⃣ Suppression colonnes inutiles
# ================================

if 'NewsletterSubscribed' in df.columns:
    df = df.drop(columns=["NewsletterSubscribed"])
    print("[OK] 'NewsletterSubscribed' supprimée (variance nulle).")

# ================================
# 7️⃣ Feature Engineering sur LastLoginIP
# ================================

if 'LastLoginIP' in df.columns:
    df[["IP1","IP2","IP3","IP4"]] = df["LastLoginIP"].str.split(".", expand=True).astype(int)

    def ip_type(ip):
        first = int(ip.split('.')[0])
        return "Private" if first in [10, 172, 192] else "Public"

    df["IP_Type"] = df["LastLoginIP"].apply(ip_type)
    df = df.drop(columns=["LastLoginIP"])
    print(f"[OK] IP features créées : IP1-4 + IP_Type\n{df['IP_Type'].value_counts()}")
else:
    print("[AVERTISSEMENT] Colonne 'LastLoginIP' introuvable.")

# ================================
# 8️⃣ Analyse distribution colonnes catégorielles
# ================================

# CORRECTION : 'str' explicite pour compatibilité Pandas 3
cat_cols = df.select_dtypes(include=['str', 'object']).columns.tolist()
print(f"\n[INFO] Colonnes catégorielles ({len(cat_cols)}) : {cat_cols}")

for col in cat_cols:
    print(f"\n--- Distribution de '{col}' ---")
    print(df[col].value_counts())
    print(f"  Valeurs manquantes : {df[col].isnull().sum()}")

# ================================
# 9️⃣ Encodage des variables catégorielles
# ================================

# --- Encodage Ordinal (noms corrigés selon le vrai dataset) ---
ordinal_mappings = {
    'AgeCategory':        ['Inconnu', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'SpendingCategory':   ['Low', 'Medium', 'High', 'VIP'],
    'PreferredTimeOfDay': ['Nuit', 'Matin', 'Midi', 'Après-midi', 'Soir'],
    'LoyaltyLevel':       ['Nouveau', 'Jeune', 'Établi', 'Ancien'],
    'ChurnRiskCategory':  ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSizeCategory': ['Petit', 'Moyen', 'Grand'],
}

for col, categories in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].fillna(categories[0])
        enc = OrdinalEncoder(
            categories=[categories],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df[[col]] = enc.fit_transform(df[[col]])
        print(f"[OK] Encodage ordinal → '{col}'")
    else:
        print(f"[AVERTISSEMENT] Colonne '{col}' introuvable.")

# --- Encodage One-Hot (noms corrigés selon le vrai dataset) ---
onehot_cols = [
    'CustomerType',
    'FavoriteSeason',
    'Region',
    'WeekendPreference',
    'ProductDiversity',
    'Gender',
    'AccountStatus',
    'IP_Type',
]

onehot_present = [c for c in onehot_cols if c in df.columns]
if onehot_present:
    df = pd.get_dummies(df, columns=onehot_present, drop_first=True)
    print(f"[OK] One-Hot Encoding → {onehot_present}")
    print(f"     Dimensions après One-Hot : {df.shape}")

# RFMSegment : One-Hot (pas d'ordre entre Champions/Fidèles/etc.)
if 'RFMSegment' in df.columns:
    df = pd.get_dummies(df, columns=['RFMSegment'], drop_first=True)
    print("[OK] One-Hot Encoding → 'RFMSegment'")

# Country : fréquence (37+ pays → One-Hot créerait trop de colonnes)
if 'Country' in df.columns:
    freq_map = df['Country'].value_counts(normalize=True)
    df['Country_freq'] = df['Country'].map(freq_map)
    df = df.drop(columns=['Country'])
    print("[OK] 'Country' → encodage par fréquence (Country_freq)")

# ================================
# 🔟 Sauvegarde dataset nettoyé + encodé
# ================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] dataset_cleaned.csv → {OUTPUT_PATH}")
print(f"     Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ================================
# 1️⃣1️⃣ Séparation X / y
# ================================

if 'Churn' not in df.columns:
    print("[ERREUR] Colonne cible 'Churn' introuvable.")
    sys.exit(1)

cols_to_drop = ['Churn']
if 'CustomerID' in df.columns:
    cols_to_drop.append('CustomerID')

X = df.drop(columns=cols_to_drop)
y = df['Churn']

print(f"\n[INFO] X : {X.shape} | y : {y.shape}")
print(f"[INFO] Distribution Churn :\n{y.value_counts()}")
print(f"[INFO] Taux de churn : {y.mean()*100:.1f}%")

# ================================
# 1️⃣2️⃣ Train/Test Split (80/20 stratifié)
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"[OK] Split : Train {X_train.shape} | Test {X_test.shape}")

# ================================
# 1️⃣3️⃣ Normalisation — StandardScaler
# ================================

# RÈGLE ABSOLUE :
#   fit_transform sur X_train uniquement
#   transform seulement sur X_test
#   ne jamais normaliser y

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

print(f"[OK] StandardScaler sur {len(num_cols)} colonnes numériques")
print(f"     Moyenne X_train après scaling (doit être ~0) : {X_train[num_cols].mean().mean():.4f}")

# ================================
# 1️⃣4️⃣ Sauvegarde Train/Test
# ================================

TRAIN_TEST_PATH = "data/train_test/"
os.makedirs(TRAIN_TEST_PATH, exist_ok=True)

X_train.to_csv(TRAIN_TEST_PATH + "X_train.csv", index=False)
X_test.to_csv(TRAIN_TEST_PATH  + "X_test.csv",  index=False)
y_train.to_csv(TRAIN_TEST_PATH + "y_train.csv", index=False)
y_test.to_csv(TRAIN_TEST_PATH  + "y_test.csv",  index=False)

print(f"\n[OK] Fichiers sauvegardés dans '{TRAIN_TEST_PATH}'")
print(f"     X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"     y_train : {y_train.shape} | y_test : {y_test.shape}")
print("\n✅ Pipeline preprocessing terminé avec succès !")