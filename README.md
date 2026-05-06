🚀 ChurnSight — Customer Behavior Analysis & Prediction
📌 Description

ChurnSight est une solution de Machine Learning dédiée à l’analyse du comportement client dans le secteur du retail.
Elle permet de transformer les données en décisions stratégiques à travers trois objectifs principaux :

🔍 Prédiction du churn (clients susceptibles de partir)
💰 Estimation de la valeur client (MonetaryTotal)
👥 Segmentation client (approche RFM avec K-Means)

Le projet implémente un pipeline complet :

Prétraitement → Modélisation → Évaluation → Prédiction → Déploiement (Flask)
⚙️ Installation
# Cloner le projet
git clone https://github.com/Ranim638515/machine-learning-project.git
cd churnsight

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# Installer les dépendances
pip install -r exigences.txt

▶️ Utilisation
🔹 1. Prétraitement
python src/preprocessing.py
✔ Nettoyage des données
✔ Feature engineering
✔ Encodage + normalisation
✔ Split train/test

🔹 2. Entraînement
python src/train_model.py
✔ Classification (churn)
✔ Régression (valeur client)
✔ Clustering (segmentation)
✔ Génération de graphiques

🔹 3. Prédictions
python src/predict.py

✔ Prédiction du churn
✔ Estimation du montant client
✔ Attribution d’un segment

🔹 4. Application Web
python app/app.py

👉 Accès : http://localhost:5000

Fonctionnalités :

Prédiction en temps réel
Analyse batch (CSV)
Visualisation du risque client
📊 Résultats
🔸 Classification
Modèle : XGBoost
Accuracy : 96.2%
F1-score : 0.962
🔸 Régression
Modèle : Random Forest
R² : 0.683
MAE : 690 £
🔸 Clustering
Algorithme : K-Means (k=4)
Segments :
Dormants
Réguliers
Champions
VIP
⚠️ Limitations
Régression limitée par la forte dispersion des montants
Dataset partiellement synthétique
Déséquilibre dans certains segments
🔮 Perspectives
Optimisation des modèles (SMOTE, Optuna)
Intégration de données temps réel
Monitoring (data drift, retraining automatique)
Déploiement en API REST pour CRM
🎯 Conclusion

ChurnSight permet de :

Anticiper le churn
Optimiser les stratégies marketing
Améliorer la gestion client
👩‍💻 Auteur

Ranim Bouguila
