import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset (puede ser "iris.csv" sin etiquetas o el que tengas)
df = pd.read_csv("iris.csv")

# Visualizar las primeras filas para confirmar la carga
print("Primeras filas del dataset:")
print(df.head())

# Graficar la forma del pétalo
plt.figure(figsize=(8, 6))
plt.scatter(df['petal_length'], df['petal_width'], color='magenta', alpha=0.7)
plt.title("Visualización de la forma del pétalo")
plt.xlabel("Longitud del pétalo")
plt.ylabel("Ancho del pétalo")
plt.grid(True)
plt.show()
