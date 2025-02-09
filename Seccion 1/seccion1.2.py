import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Cargar el dataset
df = pd.read_csv("iris.csv")
print("Primeras filas del dataset:")
print(df.head())

#Seleccionar las variables: sepal_length y sepal_width
X = df[['sepal_length', 'sepal_width']]

#Aplicar K-Means Clustering con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

#Agregar la etiqueta del cluster al DataFrame
df['cluster'] = clusters

#Graficar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['sepal_width'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title("K-Means Clustering: 2 clusters (Forma del Sépalo)")
plt.xlabel("Longitud del Sépalo")
plt.ylabel("Ancho del Sépalo")
plt.grid(True)
plt.show()
