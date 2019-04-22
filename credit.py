import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv(r"C:\Users\vichu\Documents\ML\card.csv")
pd.set_option('display.max_columns', None)
train.info()
train.describe()
train.columns

print(train.skew())
train.groupby('Y').size()

print(train.shape)
drop_features=['ID','X6', 'X7', 'X8', 'X9', 'X10', 'X11']
train=train.drop(drop_features,axis=1)

sns.FacetGrid(train, hue="Y",size=8).map(sns.kdeplot, "X1").add_legend()

#import plotting libraries


# Scatter plot of only the highly correlated pairs

# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore Wilderness_Area and Soil_Type as they are binary

#sets the number of features considered
size = 24 

#create a dataframe with only 'size' features
data=train.iloc[:,:size] 

#get the names of all the columns
cols=data.columns 

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()
# Set the threshold to select only only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size) : #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

for v,i,j in s_corr_list:
    sns.pairplot(train, hue="Y", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()

    
data_corr.iloc[0,1]


sns.pairplot(train, size=6, x_vars="X12",hue="Y")

sns.scatterplot()
