#importation des bibliotheques
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from import to_categorical
from sklearn.datasets import load_iris

#chargement et preparation des donnees
iris = load_iris()
X = iris.data
y = iris.target
print(f"Nombre de caracteristiques d'entree:{X.shape[1]}")
print(f"Nombre de classes de sortie:{len(np.unique(y))}")

#encodage des variables categorielles par One Hote Encoding
y_encoded = to_categorical(y)

#division des donnees en ensemble d'entrainnement et de test
X_train,X_test,y_train,y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=42)

#normalisation des donnees
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#definir le nombre d'entree et de sortie pour le model
input_dim = X_train_scaled.shape[1]
output_dim = y_encoded.shape[1]

#construction du model
model = Sequential([
    Dense(10, input_dim=input_dim,activation='relu'),
    Dense(8,activation='relu'),
    Dense(output_dim,activation='softmax')
])

#affichage du resumee du model
print("\n--- Resume du Modele ---")
model.summary()

#compilation du model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#entrainnement du model
print("\n--- Début de l'Entraînement ---")
history = model.fit(
    X_train_scaled,
    y_train,
    epochs = 100,
    batch_size = 5,
    verbose = 0
)
print("--- Fin de l'Entraînement ---")

#evaluation du model
print("\n--- Évaluation du Modèle ---")
loss, accuracy = model.evaluate(X_test_scaled,y_test,verbose = 0)
print(f"Perte(loss) sur l'ensemble de test:{loss:.4f}")
print(f"Precision(accuracy) sur l'ensemble de test:{accuracy*100:.2f}%")