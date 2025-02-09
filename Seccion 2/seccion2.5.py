import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Cargar el dataset
df = pd.read_csv("iris.csv")

#Seleccionar las variables de interés
X = df[['sepal_length', 'sepal_width']]

#Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir los diferentes números de clusters a probar
clusters_to_try = [2, 3, 4, 5]

plt.figure(figsize=(16, 12))

for i, k in enumerate(clusters_to_try):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Transformar los centroides a la escala original para graficarlos
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    plt.subplot(2, 2, i+1)
    plt.scatter(df['petal_length'], df['petal_width'], c=cluster_labels, cmap='coolwarm', alpha=0.6, edgecolor='k')
    plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='black', s=200, marker='X', label='Centroides')
    plt.title(f'K-Means con {k} clusters - Forma del Pétalo')
    plt.xlabel("Longitud del pétalo")
    plt.ylabel("Ancho del pétalo")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
