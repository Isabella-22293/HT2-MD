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

#Definir los diferentes números de clusters a probar
clusters_to_try = [2, 3, 4, 5]  

#Crear una figura para mostrar varias gráficas
plt.figure(figsize=(16, 12))

#Para cada número de clusters, aplicar K-Means y graficar los resultados
for i, k in enumerate(clusters_to_try):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    #Obtener los centroides y transformarlos a la escala original para graficarlos
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    #Crear un subplot para cada valor de k
    plt.subplot(2, 2, i + 1)
    plt.scatter(df['sepal_length'], df['sepal_width'], 
                c=cluster_labels, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
                marker='X', s=200, c='red', label='Centroides')
    plt.title(f'K-Means con {k} clusters')
    plt.xlabel('Longitud del sépalo')
    plt.ylabel('Ancho del sépalo')
    plt.legend()
    plt.grid(True)

# Ajustar el layout para que no se superpongan las gráficas
plt.tight_layout()
plt.show()
