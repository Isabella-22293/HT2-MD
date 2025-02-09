import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator  

#Cargar el dataset 
df = pd.read_csv("iris.csv")

#Seleccionar las variables de interés
X = df[['sepal_length', 'sepal_width']]

#Estandarizar los datos 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Calcular la inercia para un rango de k 
inercia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

#Graficar la curva del codo
plt.figure(figsize=(8, 6))
plt.plot(k_range, inercia, marker='o')
plt.title("Método del Codo (forma del sépalo)")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.grid(True)
plt.show()

#Usar la librería kneed para detectar automáticamente el "codo"
knee_locator = KneeLocator(k_range, inercia, curve="convex", direction="decreasing")
optimal_k = knee_locator.elbow
print("El número óptimo de clusters detectado por kneed es:", optimal_k)
