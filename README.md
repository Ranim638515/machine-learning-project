🚀 ChurnSight — Customer Analytics & Prediction Solution
📌 Description

ChurnSight est une solution de Machine Learning dédiée à l’analyse du comportement client dans le retail.
Elle permet de transformer les données en décisions stratégiques en répondant à trois objectifs principaux :

🔍 Prédire le churn (clients susceptibles de partir)
💰 Estimer la valeur client (MonetaryTotal)
👥 Segmenter les clients (approche RFM avec K-Means)

Le projet implémente une pipeline complète :
👉 Préprocessing → Modélisation → Évaluation → Prédiction → Déploiement (Flask)

⚙️ Installation
1. Cloner le projet
git clone https://github.com/Ranim638515/machine-learning-project.git
cd churnsight

2. Créer un environnement virtuel
python -m venv venv

3. Activer l’environnement
Windows :
venv\Scripts\activate

Linux / Mac :
source venv/bin/activate

4. Installer les dépendances
pip install -r requirements.txt

🗂️ Structure du projet
projet_ml_retail/
│
├── data/                      # Données du projet
│   ├── raw/                   # Données brutes
│   ├── processed/             # Données nettoyées
│   └── train_test/            # Données train/test
│
├── notebooks/                 # Exploration & prototypage
│
├── src/                       # Pipeline Machine Learning
│   ├── preprocessing.py       # Nettoyage & feature engineering
│   ├── train_model.py         # Entraînement & évaluation
│   ├── predict.py             # Prédictions & validation
│   └── utils.py               # Fonctions utilitaires
│
├── models/                    # Modèles sauvegardés (.pkl)
├── app/                       # Application web Flask
├── reports/                   # Graphiques & résultats
│
├── requirements.txt           # Dépendances
├── README.md                  # Documentation
└── .gitignore

▶️ Utilisation
🔹 1. Préprocessing
python src/preprocessing.py


✔ Nettoyage des données
✔ Feature engineering
✔ Encodage + normalisation
✔ Split train/test

🔹 2. Entraînement des modèles
python src/train_model.py


✔ Classification (churn)
✔ Régression (valeur client)
✔ Clustering (segmentation)
✔ Génération de graphiques (confusion matrix, elbow, etc.)

🔹 3. Prédictions
python src/predict.py


✔ Prédiction du churn
✔ Estimation du montant client
✔ Attribution d’un segment

🔹 4. Application web
python app/app.py


👉 Accès : http://localhost:5000

Fonctionnalités :

prédiction en temps réel
analyse batch (CSV)
visualisation du risque client
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
Segments : Dormants, Réguliers, Champions, VIP
⚠️ Limitations
Régression limitée par la forte dispersion des montants
Dataset partiellement synthétique
Déséquilibre dans certains segments
🔮 Perspectives
Optimisation des modèles (SMOTE, Optuna)
Ajout de données comportementales temps réel
Monitoring (data drift, retraining automatique)
Déploiement en API REST pour intégration CRM
🎯 Conclusion

ChurnSight est une solution complète qui permet de :
👉 anticiper le churn, optimiser le marketing et améliorer la gestion client

👩‍💻 Auteur

Ranim Bouguila