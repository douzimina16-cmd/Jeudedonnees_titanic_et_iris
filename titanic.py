import matplotlib.pyplot as plt
import seaborn as sns
titanic=sns.load_dataset("titanic")
print(titanic)
selected_features=["survived","pclass","sex","age","sibsp","parch","fare","embarked","deck"]
titanic=titanic[selected_features]
print(titanic.head()) #5 premiere ligne  dands les datasets
print(titanic.info())
print(titanic.describe())
print(titanic.isnull().sum())
# Supprimer la caractéristique deck
titanic = titanic.drop(columns=["deck"])
# Imputer les valeurs manquantes de age par la médiane
titanic["age"]=titanic["age"].fillna(titanic["age"].median())
print(titanic["age"])
# Supprimer les observations ayant une valeur manquante pour embarked
titanic = titanic.dropna(subset=["embarked"])
# Vérifier les valeurs manquantes après traitement
print("Valeurs manquantes après traitement :")
print(titanic.isnull().sum())
gender_counts = titanic["sex"].value_counts()
print("Nombre d'hommes et de femmes :\n", gender_counts)
sns.countplot(x='sex',data=titanic)
plt.title("Répartition des passagers par sexe")
plt.show()
sns.countplot(x='sex',hue='pclass',data=titanic)
plt.title("Répartition des passagers par classe et sexe")
plt.show()
sns.countplot(x='sex',hue='survived',data=titanic)
plt.title("Répartition des passagers par sexe et survie")
plt.show()
sns.catplot(x="pclass", y="age", data=titanic, kind="box")
plt.xlabel("Classe")
plt.ylabel("Âge")
plt.title("Relation entre la classe et l'âge")
plt.show()
sns.catplot(x="pclass", y="age", hue="sex", data=titanic,kind="box")
plt.xlabel("Classe")
plt.ylabel("Âge")
plt.title("Relation entre la classe, l'âge et le sexe")
plt.show()


