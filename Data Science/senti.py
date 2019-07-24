import pandas as pd
import os
import io



dir=r"C:\\Users\\vichu\\Documents\ML"
census=pd.read_csv(os.path.join(dir,"train_senti.csv"))
print(census.info())


census.isna()

census1=census.dropna()

census1.concat()
print(census.columns)
print(census1['unique_hash'])

census1['new'] = census1.values.sum(axis=1)

census1['new'] = census1.sum(axis=1).astype(int).astype(str)

name = list(census1.columns)
name.remove('unique_hash')


print(name)

for x in name :
    census1['unique_hash']= census1['unique_hash'].map(str)+census1[x].map(str)
    
df= pd.DataFrame(census1,columns=name)

census['unique_hash'].apply(len)

df = census[census['unique_hash'].apply(lambda x: len(x) <= 40)]

census['unique_hash'].apply(lambda x: x.str.len().gt(40))

census.select_dtypes(['object']).apply(lambda x: x.str.len().gt(40)).axis=1


bad_data=census[~(census.unique_hash.str.len() == 40)]
temp=list(bad_data.index)

for x in temp:
    print(x)
    temp1=x-1
