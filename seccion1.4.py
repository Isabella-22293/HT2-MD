import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Cargar el dataset
df = pd.read_csv("iris.csv")

#Seleccionar las variables de interés: sepal_length y sepal_width
X = df[['sepal_length', 'sepal_width']]

#Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Aplicar el método del codo: evaluar KMeans para k de 1 a 10
inercia = []  
rango_k = range(1, 11)

for k in rango_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

#Graficar la inercia en función del número de clusters
plt.figure(figsize=(8, 6))
plt.plot(rango_k, inercia, marker='o')
plt.title("Método del Codo: Determinación del Número Óptimo de Clusters")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inercia (Suma de Distancias Cuadráticas)")
plt.grid(True)
plt.show()
