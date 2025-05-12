import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("redes-neuronales/dataset2.csv")
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Obesidad'] = LabelEncoder().fit_transform(df['Obesidad'])

X = df[['Height_cm', 'Age', 'Gender', 'Weight_kg', 'IMC']]
y = df['Obesidad']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=0)


plt.figure(figsize=(10, 5))


plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Error de Entrenamiento', color='blue')
plt.plot(history.history['val_loss'], label='Error de Validación', color='red', linestyle='--')
plt.title('Evolución del Error durante el Entrenamiento')
plt.ylabel('Error (Pérdida)')
plt.xlabel('Iteración (Época)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'\nResultados finales:')
print(f'- Error en prueba: {test_loss:.4f}')
print(f'- Precisión en prueba: {test_acc:.2%}')