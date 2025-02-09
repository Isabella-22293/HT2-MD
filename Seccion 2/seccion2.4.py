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

inercia = []
rango_k = range(1, 11)

for k in rango_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)  # Se utiliza X_scaled si se desea trabajar con datos estandarizados
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(rango_k, inercia, marker='o')
plt.title("Método del Codo - Forma del Pétalo")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inercia (Suma de Distancias Cuadráticas)")
plt.grid(True)
plt.show()
