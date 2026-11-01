import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris
iris=load_iris()
print(iris)  #importer les jeux de donnes iris et Affiche tout l’objet : la matrice des mesures, les labels, les noms, etc
print(iris.target_names) #det les classes de fleures
print(iris.feature_names)  #caracteristiq en liste
print(iris.DESCR)
df_iris = pd.DataFrame(iris.data,
columns=iris.feature_names)
df_iris["species"] = [iris.target_names[i] for i in
iris.target]

# Afficher les cinq premières observations
print("Les cinq premières observations :")
print(df_iris.head())

# Afficher les informations générales
print("\nInformations générales :")
print(df_iris.info())
print(df_iris.describe())
print(df_iris.isnull().sum())
#6. Déterminer la dimension du vecteur label de la base.
dim_label=iris.target.shape
print("dimention label",dim_label)
plt.figure(figsize=(8, 6))
plt.hist(iris.data[:, iris.feature_names.index("petal width (cm)")], bins=30, edgecolor='k')
plt.xlabel("Largeur des pétales (petal_width)")
plt.ylabel("Fréquence")
plt.title("Histogramme de la répartition des largeurs de pétales")
plt.show()
plt.figure(figsize=(8, 6))
sn.kdeplot(iris.data[:, iris.feature_names.index("petal width (cm)")], fill=True)
plt.xlabel("Largeur des pétales (petal_width)")
plt.ylabel("Distribution de probabilité")
plt.title("Distribution de probabilité des largeurs de pétales")
plt.show()
plt.figure(figsize=(8, 6))
sn.countplot(x="species", data=pd.DataFrame(iris.target,columns=["species"]))
plt.xlabel("Espèces")
plt.ylabel("Fréquence")
plt.title("Distribution de probabilité des espèces")
plt.show()
df_iris = pd.DataFrame(iris.data,columns=iris.feature_names)
correlation_matrix = df_iris.corr()
plt.figure(figsize=(10, 8))
sn.heatmap(correlation_matrix, annot=True,cmap="coolwarm", square=True)
plt.title("Matrice de corrélation")
plt.show()
