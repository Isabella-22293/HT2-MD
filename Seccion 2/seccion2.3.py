import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("iris.csv")
X = df[['petal_length', 'petal_width']]

#Estandarizar las variables de la forma del pétalo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Aplicar K-Means sobre los datos estandarizados (2 clusters)
kmeans_scaled = KMeans(n_clusters=2, random_state=42)
clusters_scaled = kmeans_scaled.fit_predict(X_scaled)

#Guardar los clusters obtenidos en el DataFrame
df['cluster_scaled'] = clusters_scaled

#Graficar los resultados (transformando los centroides a la escala original)
centroids_scaled = kmeans_scaled.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(df['petal_length'], df['petal_width'], c=df['cluster_scaled'], cmap='coolwarm', alpha=0.7)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='black', s=200, marker='X', label='Centroides')
plt.title("K-Means (2 clusters) con estandarización - Forma del Pétalo")
plt.xlabel("Longitud del pétalo")
plt.ylabel("Ancho del pétalo")
plt.legend()
plt.grid(True)
plt.show()
