import pandas as pd

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/titanic_train.csv")
print(titanic_train.shape)

print(titanic_train.info())

#discover pattern: which class is majority?
titanic_train.groupby('Survived').size()

titanic_train.groupby('Survived').size()
titanic_train.groupby(['Sex','Survived']).size()
titanic_train.groupby(['Sex','Pclass','Survived']).size()
titanic_train.groupby(['Sex','Embarked','Survived']).size()

titanic_test = pd.read_csv(r"C:\Users\vichu\Documents\ML\titanic_test.csv")
print(titanic_test.shape)

print(titanic_test.columns)


titanic_test['Survived'] = 0
titanic_test.loc[(titanic_test.Sex =='female') & (titanic_test.Pclass=='1'), 'Survived'] = 1
titanic_test.to_csv(r"C:\Users\vichu\Documents\ML\titanic_test1.csv",columns=['PassengerId', 'Survived'],index=False)
