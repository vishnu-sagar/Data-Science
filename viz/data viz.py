import pandas as pd
import seaborn as sns
from sklearn import preprocessing
titanic_train = pd.read_csv(r"C:\Users\vichu\Downloads\train.csv")
print(titanic_train.shape)

print(titanic_train.info())

####################
#EDA-explantory data analaysis


imputable_cont_features = ['Age', 'Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])


#univariate-cat
sns.countplot(x='Age',data=titanic_train)

sns.countplot(x='Embarked',data=titanic_train)

titanic_train.groupby('Survived').size()
sns.countplot(x='Pclass',data=titanic_train)
sns.countplot(x='Sex',data=titanic_train)

#univariate-cont
sns.distplot(titanic_train['Age'],kde=True,hist=False)
#box-whisker
sns.boxplot(x='Age',data=titanic_train)


#bivariate-cat vs cat
sns.factorplot(x="Sex", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Pclass", hue="Survived", data=titanic_train, kind="count", size=6)
sns.factorplot(x="Embarked", hue="Survived", data=titanic_train, kind="count", size=6)

#continuous vs categorigal
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Age").add_legend()

#cont vs cont
#continuous vs continuous: scatter plot
sns.jointplot(x="Fare", y="Age", data=titanic_train)


#sns.pairplot( titanic_train,hue="Y", size=6 )


##multi-variate plots
#3-categorical features
g = sns.FacetGrid(titanic_train, row="Sex", col="Pclass") 
g.map(sns.countplot, "Survived")
g = sns.FacetGrid(titanic_train, row="Embarked", col="Sex") 
g.map(sns.countplot, "Survived")



g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Fare")

#is age have an impact on survived for each pclass and sex group?
g = sns.FacetGrid(titanic_train, row="Pclass", col="Sex", hue="Survived") 
g.map(sns.kdeplot, "Age")