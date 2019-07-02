#apriori algorithum

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting data set
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transaction=[]
for i in range(7501):
    transaction.append([str(dataset.values[i,j]) for j in range(20)])
    
#training apriori to data
from apyori import apriori
res=apriori(transaction,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#fitting and visualisinresult
result=list(res)