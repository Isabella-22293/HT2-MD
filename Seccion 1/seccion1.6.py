import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Cargar el dataset real con etiquetas
df_real = pd.read_csv("iris-con-respuestas.csv")

#Seleccionar las variables de interés (forma del sépalo)
X = df_real[['sepal_length', 'sepal_width']]

#Aplicar K-Means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df_real['cluster'] = clusters

#Graficar los clusters obtenidos
plt.figure(figsize=(8,6))
plt.scatter(df_real['sepal_length'], df_real['sepal_width'], c=df_real['cluster'], cmap='viridis', alpha=0.7)
plt.title("K-Means (3 clusters) usando solo la forma del sépalo")
plt.xlabel("Longitud del sépalo")
plt.ylabel("Ancho del sépalo")
plt.grid(True)
plt.show()

# Graficar las etiquetas reales
plt.figure(figsize=(8,6))
# Asignamos colores a las especies (por ejemplo, setosa=0, versicolor=1, virginica=2)
species_mapping = {'setosa':0, 'versicolor':1, 'virginica':2}
real_labels = df_real['species'].map(species_mapping)
plt.scatter(df_real['sepal_length'], df_real['sepal_width'], c=real_labels, cmap='viridis', alpha=0.7)
plt.title("Etiquetas reales (especies)")
plt.xlabel("Longitud del sépalo")
plt.ylabel("Ancho del sépalo")
plt.grid(True)
plt.show()
