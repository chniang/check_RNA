Markdown

# 🌸 Classification de Fleurs Iris avec Réseau Neuronal Dense (DNN)

Ce dépôt contient un script Python simple (`model.py`) qui implémente et entraîne un **Réseau Neuronal Dense (DNN)** utilisant **TensorFlow/Keras** pour la classification des célèbres fleurs du jeu de données **Iris** de Scikit-learn.

## 🎯 Objectif du Projet

L'objectif est de prédire l'espèce d'une fleur Iris (Setosa, Versicolor, ou Virginica) en se basant sur quatre caractéristiques morphologiques (longueur/largeur des sépales et des pétales).

---

## ⚙️ Configuration et Dépendances

Ce projet nécessite les bibliothèques Python suivantes : `tensorflow`, `scikit-learn`, `pandas`, et `numpy`.

### Installation

Pour installer les dépendances nécessaires, utilisez `pip` dans votre environnement virtuel :

```bash
pip install tensorflow scikit-learn pandas numpy
🚀 Guide d'Exécution
Cloner le dépôt :

Bash

git clone [https://github.com/chniang/check_RNA.git](https://github.com/chniang/check_RNA.git)
cd CHECK_RNA
Activer l'environnement virtuel (tf_venv) :
Assurez-vous que votre environnement virtuel est actif pour exécuter le script avec les bonnes dépendances.

Bash

# Sous PowerShell (Windows)
.\tf_venv\Scripts\Activate.ps1
# Sous MINGW64 / Linux
source tf_venv/bin/activate
Lancer le script d'entraînement :

Bash

python model.py
Résultat de l'Évaluation
Le script effectue l'entraînement sur 100 époques. Le résultat final sur l'ensemble de test (20% des données) démontre l'efficacité du modèle :

--- Évaluation du Modèle ---
Perte(loss) sur l'ensemble de test:0.0331
Precision(accuracy) sur l'ensemble de test:100.00%
🧠 Détails du Modèle
Préparation des Données
Encodage One-Hot : Les étiquettes de classe (0, 1, 2) sont converties en format One-Hot ([1, 0, 0], etc.) en utilisant to_categorical.

Division : Les données sont divisées en 80% (entraînement) et 20% (test) via train_test_split.

Normalisation : Toutes les caractéristiques d'entrée sont normalisées à l'aide de StandardScaler pour optimiser la convergence du réseau.

Architecture
Le modèle utilise une architecture séquentielle simple :

Couche	Type	Neurones	Activation	Rôle
Input	Dense	10	ReLU	Couche d'entrée (4 features)
Cachée	Dense	8	ReLU	Couche intermédiaire
Output	Dense	3	Softmax	Couche de sortie (3 classes)

Exporter vers Sheets
Compilation du Modèle :

Optimiseur : adam

Fonction de Perte : categorical_crossentropy

Métriques : accuracy
