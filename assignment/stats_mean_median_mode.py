import pandas as pd
import numpy as np
import os
from scipy.stats import mode


#using numpy
def cal(x):
    mean= np.mean(x)
    median = np.median(x)
    m= mode(x)
    variance=np.var(x)
    stddev=np.std(x)
    return mean,median,m,variance,stddev


dir=r"C:\\Users\\vichu\\Documents\assign\Inputfiles"
stats=pd.read_excel(os.path.join(dir,"Stats.xlsx"),index_col=0)
print(stats.info())

mean,median,Mode,Variance,Stddev=cal(stats)

print('MEAN\n',mean)
print('MEDIAN\n',median)
print('MODE\n',Mode)
print('VARIANCE\n',Variance)
print('STDDEV\n',Stddev)

