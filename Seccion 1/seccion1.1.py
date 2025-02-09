import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

#Cargar el dataset
df = pd.read_csv("iris.csv")
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

#Visualizar la forma del sépalo con un gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(df['sepal_length'], df['sepal_width'], color='blue', alpha=0.7)
plt.title("Visualización de la forma del sépalo")
plt.xlabel("Longitud del sépalo")
plt.ylabel("Ancho del sépalo")
plt.grid(True)
plt.show()

#Pairplot para observar la relación entre todas las variables
sns.pairplot(df)
plt.suptitle("Pairplot del Dataset Iris", y=1.02)
plt.show()
