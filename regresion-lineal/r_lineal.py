# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 1. Generar un dataset sintético
np.random.seed(42)
n_samples = 100

# Simulamos altura en cm (entre 150 y 200) y edad en años (entre 18 y 60)
altura = np.random.uniform(150, 200, n_samples)
edad = np.random.uniform(18, 60, n_samples)

# Simulamos el peso en kg con una relación lineal + ruido
peso = 0.5 * altura + 0.8 * edad + np.random.normal(0, 5, n_samples)

# Crear DataFrame
data = pd.DataFrame({
    'Altura': altura,
    'Edad': edad,
    'Peso': peso
})

print("Primeras filas del dataset:")
print(data.head())

# 2. Separar variables independientes y dependiente
X = data[['Altura', 'Edad']]  # Variables independientes
y = data['Peso']              # Variable dependiente

# 3. Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# 6. Evaluar el modelo
print("\nEvaluación del modelo:")
print(f"Coeficientes: {modelo.coef_}")
print(f"Intercepto: {modelo.intercept_}")
print(f"MSE (error cuadrático medio): {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE (error absoluto medio): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# 7. Visualizar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Peso Real")
plt.ylabel("Peso Predicho")
plt.title("Regresión lineal")
plt.grid(True)
plt.show()
