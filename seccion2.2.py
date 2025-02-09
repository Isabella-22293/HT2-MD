import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("iris.csv")
#Seleccionar las variables: petal_length y petal_width
X = df[['petal_length', 'petal_width']]

#Aplicar K-Means Clustering con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

#Agregar la etiqueta del cluster al DataFrame
df['cluster'] = clusters

#Graficar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(df['petal_length'], df['petal_width'], c=df['cluster'], cmap='coolwarm', alpha=0.7)
plt.title("K-Means Clustering (2 clusters) - Forma del Pétalo")
plt.xlabel("Longitud del pétalo")
plt.ylabel("Ancho del pétalo")
plt.grid(True)
plt.show()
