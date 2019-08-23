import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
dir=r"C:\\Users\\vichu\\Documents\loan"
#os.chdir(r'C:\Users\prana\Desktop')
df = pd.read_csv(os.path.join(dir,"train.csv"))
df_test = pd.read_csv(os.path.join(dir,"test.csv"))
df.head()

pd.options.display.max_seq_items = 4000

df_test1 = pd.read_csv(os.path.join(dir,"test.csv"))


def get_continuous_features(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_features(df):
    return df.select_dtypes(exclude=['number']).columns

def cast_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')
        
        
        
cat_features=get_categorical_features(df)
cont_features=get_continuous_features(df)
print(cat_features)
print(cont_features)


from sklearn import preprocessing
for col in df[cat_features]:
    le_all=preprocessing.LabelEncoder()
    le_all.fit(df[col])
    print(le_all.classes_)
    df[col]=le_all.transform(df[col])
    
    
df_test=df_test.replace(['Apr-12','Feb-12','Mar-12','May-12'],['04/2012','02/2012','03/2012','05/2012'])

df_test=df_test.replace(['01/01/12','01/02/12','01/03/12'],['2012-01-01','2012-02-01','2012-03-01'])


for col in df_test[cat_features]:
    le_all=preprocessing.LabelEncoder()
    le_all.fit(df_test[col])
    print(le_all.classes_)
    df_test[col]=le_all.transform(df_test[col])

df_train=df
    
import numpy as np
df_train['loan_to_value_log'] = np.log(df_train['loan_to_value'])
df_test['loan_to_value_log'] = np.log(df_test['loan_to_value'])
df_train['loan_to_value_log'].hist(bins=25)

df_train['unpaid_principal_bal_log'] = np.log(df_train['unpaid_principal_bal'])
df_test['unpaid_principal_bal_log'] = np.log(df_test['unpaid_principal_bal'])
df_train['unpaid_principal_bal_log'].hist(bins=25)


df_train['total_amount']=df_train['unpaid_principal_bal']*(1+((df_train['interest_rate']/100)*(df_train['loan_term']/365)))
df_test['total_amount']=df_test['unpaid_principal_bal']*(1+((df_test['interest_rate']/100)*(df_test['loan_term']/365)))

df_train['debt_to_income_ratio_log'] = np.log(df_train['debt_to_income_ratio'])
df_test['debt_to_income_ratio_log'] = np.log(df_test['debt_to_income_ratio'])
df_test['debt_to_income_ratio_log'].hist(bins=25)

df_train=df_train.drop(['loan_id','unpaid_principal_bal','loan_to_value'], axis=1)
#df_train=df_train.drop('Loan_ID',axis=1) 
df_test=df_test.drop(['loan_id','unpaid_principal_bal','loan_to_value'],axis=1)
#df_train = df_train.drop('LoanAmount',axis=1)
#df_test = df_test.drop('LoanAmount',axis=1)
X = df_train 
y = df_train.m13
X=pd.get_dummies(X,columns=cat_features)

df_train.columns

df_train=pd.get_dummies(df_train,columns=cat_features) 
df_test=pd.get_dummies(df_test,columns=cat_features)


df_train=df_train.drop(['number_of_borrowers'], axis=1) 
df_test=df_test.drop(['number_of_borrowers'], axis=1)

df_train['number_of_borrowers']=df['number_of_borrowers']
df_test['number_of_borrowers']=df_test1['number_of_borrowers']

from sklearn.linear_model import LogisticRegression
#import xgboost as xgb
from sklearn import ensemble,tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
features=['m7', 'm8', 'm9', 'm10', 'm11', 'm12','borrower_credit_score']
X = df_train.drop('m13',axis=1)
y=df_train.m13
i=1
f=[]
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))    
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    #dt_estimator = ensemble.GradientBoostingClassifier(learning_rate=1,n_estimators=300,random_state=100)
    model2 = ensemble.GradientBoostingClassifier(learning_rate=0.01,n_estimators=1000,random_state=100,max_depth=4,min_samples_split=10)
    model2.fit(xtr, ytr)     
    pred_test1 = model2.predict(xvl)    
    score = accuracy_score(yvl,pred_test1)
    matrix=confusion_matrix(yvl,pred_test1)
    f1=f1_score(yvl,pred_test1)
    f.append(f1)
    print(np.mean(f))
    p=precision_score(yvl,pred_test1)
    r=recall_score(yvl,pred_test1)
    print('accuracy_score',score)
    print('matrix',matrix)
    print('f1',f1)
    print('pres',p)
    print('rec',r)
    i+=1
  
    
importances=pd.Series(model2.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8))


 
df_test.columns
pred_test1 = model2.predict(df_test)


np.bincount(pred_test1)

df_test1['m13']=pred_test1
df_test1.to_csv(os.path.join(dir,'submission.csv'), columns=['loan_id','m13'], index=False)



df_train['m9'].value_counts()
df_test['m9'].value_counts()

df_train['m9']=df_train['m9'].replace(7,6)

