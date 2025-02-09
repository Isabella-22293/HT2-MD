import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Cargar el dataset y seleccionar variables
df = pd.read_csv("iris.csv")  # Asegúrate de tener iris.csv en la ruta correcta

#Seleccionar las variables de interés: sepal_length y sepal_width
X = df[['sepal_length', 'sepal_width']]

#K-Means sin estandarización (para comparación)
kmeans_original = KMeans(n_clusters=2, random_state=42)
clusters_original = kmeans_original.fit_predict(X)
df['cluster_original'] = clusters_original

plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['sepal_width'], c=df['cluster_original'], cmap='viridis', alpha=0.7)
plt.title("K-Means (2 clusters) sin estandarización")
plt.xlabel("Longitud del Sépalo")
plt.ylabel("Ancho del Sépalo")
plt.grid(True)
plt.show()

#Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Aplicar K-Means sobre datos estandarizados
kmeans_scaled = KMeans(n_clusters=2, random_state=42)
clusters_scaled = kmeans_scaled.fit_predict(X_scaled)

#Guardamos los clusters obtenidos en el DataFrame
df['cluster_scaled'] = clusters_scaled

#Para graficar en el espacio original, transformamos los centroides
centroids_scaled = kmeans_scaled.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['sepal_width'], c=df['cluster_scaled'], cmap='viridis', alpha=0.7)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', s=200, marker='X', label='Centroides')
plt.title("K-Means (2 clusters) con estandarización")
plt.xlabel("Longitud del Sépalo")
plt.ylabel("Ancho del Sépalo")
plt.legend()
plt.grid(True)
plt.show()
