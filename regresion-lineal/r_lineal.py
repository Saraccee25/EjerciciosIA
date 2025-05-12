import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


path = "regresion-lineal/body_measurements_dataset.csv"  
datos = pd.read_csv(path)


print(datos.tail())
print(datos.describe())
print("Valores nulos por columna:\n", datos.isna().sum())
print("Distribución por género:\n", datos["Gender"].value_counts())


subDataframe = datos[['Height_cm', 'Age', 'Weight_kg']]
matrix = subDataframe.corr()
print("Matriz de correlación:\n", matrix)


subDataframe.plot(kind='scatter', x='Height_cm', y='Weight_kg', figsize=(8, 5), color='teal')
plt.title("Relación entre altura y peso")
plt.grid(True)
plt.show()


reg = LinearRegression()
reg.fit(datos[["Height_cm"]], datos["Weight_kg"])


print("B1 (pendiente):", reg.coef_)
print("B0 (intercepto):", reg.intercept_)


print("Predicción para altura = 170 cm:", reg.predict([[170]])[0])


predicted = reg.predict(datos[["Height_cm"]])


mse = mean_squared_error(datos["Weight_kg"], predicted)
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de correlación altura-peso:", np.corrcoef(datos["Height_cm"], predicted)[0, 1])


plt.figure(figsize=(10, 6))
plt.scatter(datos["Height_cm"], datos["Weight_kg"], color='blue', label='Datos reales')
plt.plot(datos["Height_cm"], predicted, color='red', label='Regresión lineal')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.title('Regresión Lineal: Peso en función de Altura')
plt.legend()
plt.grid(True)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(datos[["Height_cm"]], datos["Weight_kg"], test_size=0.2, random_state=42)


reg.fit(X_train, y_train)


y_pred = reg.predict(X_test)


plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='green')
plt.title('Regresión Lineal (Test Set)')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.grid(True)
plt.show()


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Error cuadrático medio:", mse)
print("Error medio absoluto:", mae)
print("Coeficiente de determinación R²:", r2)
