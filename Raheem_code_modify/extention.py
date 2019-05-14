from __future__ import division
print(__doc__)
import time
import itertools
import numpy as np
#from sklearn import preprocessing
#from sklearn.datasets.samples_generator import make_blobs
from collections import defaultdict
import math
import random
import multiprocessing
from numpy import genfromtxt
from numpy import loadtxt
import linecache
#import smjModule
import heapq
from scipy.spatial.distance import euclidean
import csv
from scipy import stats
#import os






my_data = loadtxt('./crosslingual_features_final.csv', delimiter=',')
my_data=np.asarray(my_data)
#print("Data:", my_data)
group=5
labels=my_data[:,0]
DD_Lable=np.split(my_data, np.where(np.diff(my_data[:,0]))[0]+1)
#print(DD_Lable)
full_data=list()
for x in DD_Lable:
    x=x.tolist()
    if(len(x)%group!=0):
        d=len(x)%group
        del x[-d:]
    full_data.append(x)

full_data = [y for x in full_data for y in x]
#for x in full_data:
#    print(x)

#print(len(full_data))
lst = range(1,int(len(full_data)/group)+1)
lst=list(itertools.chain.from_iterable(itertools.repeat(x, group) for x in lst))
dataset=list()
i=0
#print(lst)
for y in full_data:
    z=float(lst[i])
    dataset.append([z]+y)
    i=i+1
	
#print (dataset)

with open('raheem.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(dataset)
    
