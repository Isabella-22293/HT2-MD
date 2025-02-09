import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el archivo con respuestas reales
df_respuestas = pd.read_csv("iris-con-respuestas.csv")

# Seleccionar las variables de la forma del pétalo
X_real = df_respuestas[['petal_length', 'petal_width']]

# Crear la instancia de StandardScaler
scaler = StandardScaler()

# Aplicar K-Means con 3 clusters (ya que sabemos que hay 3 especies)
kmeans_real = KMeans(n_clusters=3, random_state=42)
clusters_real = kmeans_real.fit_predict(scaler.fit_transform(X_real))
df_respuestas['cluster'] = clusters_real

# Graficar el clustering obtenido
plt.figure(figsize=(8,6))
plt.scatter(df_respuestas['petal_length'], df_respuestas['petal_width'], c=df_respuestas['cluster'], cmap='viridis', alpha=0.7)
plt.title("K-Means (3 clusters) - Forma del Pétalo (Clusterings obtenidos)")
plt.xlabel("Longitud del pétalo")
plt.ylabel("Ancho del pétalo")
plt.grid(True)
plt.show()

# Graficar las etiquetas reales
plt.figure(figsize=(8,6))
# Mapear cada especie a un número para graficar (por ejemplo, setosa:0, versicolor:1, virginica:2)
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
real_labels = df_respuestas['species'].map(species_mapping)
plt.scatter(df_respuestas['petal_length'], df_respuestas['petal_width'], c=real_labels, cmap='viridis', alpha=0.7)
plt.title("Etiquetas reales - Forma del Pétalo")
plt.xlabel("Longitud del pétalo")
plt.ylabel("Ancho del pétalo")
plt.grid(True)
plt.show()
