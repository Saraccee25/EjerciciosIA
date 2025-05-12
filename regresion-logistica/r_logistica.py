"""regresion_logistica_obesidad.ipynb

Predicción de obesidad usando regresión logística
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


data = pd.read_csv("regresion-logistica/dataset2.csv")
print("Descripción del dataset:")
print(data.describe())
print("\nColumnas disponibles:")
print(data.columns)


print("\nValores nulos por columna:")
print(data.isna().sum())


data.plot(kind='scatter', x='IMC', y='Obesidad', figsize=(8,5))
plt.title('Relación entre IMC y Obesidad')
plt.show()



X = data[['IMC']]  
y = data['Obesidad']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTamaño del conjunto de prueba:")
print(X_test.count())


modelo = LogisticRegression()
modelo.fit(X_train, y_train)


prediccion_individual = modelo.predict([[35]])
print(f"\nPredicción para IMC=35: {'Obesidad' if prediccion_individual[0] == 1 else 'No obesidad'}")


y_pred = modelo.predict(X_test)



matriz_confusion = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(matriz_confusion)


cm_display = ConfusionMatrixDisplay(matriz_confusion, display_labels=['No Obeso', 'Obeso'])
cm_display.plot()
plt.title('Matriz de Confusión - Predicción de Obesidad')
plt.show()


print("\nMétricas del modelo:")
print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precisión: {precision_score(y_test, y_pred):.2f}")
print(f"Exhaustividad (Recall): {recall_score(y_test, y_pred):.2f}")
print(f"Puntuación F1: {f1_score(y_test, y_pred):.2f}")


print("\nCoeficiente del modelo (pendiente):", modelo.coef_[0][0])
print("Intercepto:", modelo.intercept_[0])


import numpy as np
imc_range = np.linspace(data['IMC'].min(), data['IMC'].max(), 300)
probabilidades = modelo.predict_proba(imc_range.reshape(-1, 1))[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Datos reales')
plt.plot(imc_range, probabilidades, color='red', label='Probabilidad predicha')
plt.axhline(0.5, color='black', linestyle='--', label='Umbral de decisión')
plt.xlabel('IMC')
plt.ylabel('Probabilidad de Obesidad')
plt.title('Regresión Logística para Predicción de Obesidad')
plt.legend()
plt.show()