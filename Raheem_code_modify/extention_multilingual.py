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




training='crosslingual_vr_features_short_training.csv'
testing='crosslingual_stats_short_testing.csv'

my_data = loadtxt('/home/grads/raheem/multilingual_vr_features/'+str(training), delimiter=',')
my_data=np.asarray(my_data)
group=40
labels=my_data[:,0]
DD_Lable=np.split(my_data, np.where(np.diff(my_data[:,0]))[0]+1)
q = loadtxt('/home/grads/raheem/multilingual_vr_features/'+str(testing), delimiter=',')
q=np.asarray(q)
print (q)
full_data=list()
q_data=list()
for x in DD_Lable:
    x=x.tolist()
    if (x[0][0] in q):
        q_data.append(x)
    if (x[0][0] not in q):
        if(len(x)%group!=0):
            d=len(x)%group
            del x[-d:]
        full_data.append(x)
        
q_data = [y for x in q_data for y in x]
full_data = [y for x in full_data for y in x]
abc=list()
lst = range(1,int(len(full_data)/group)+1)
for x in lst:
    abc.append(x+max(q))

lst=list(itertools.chain.from_iterable(itertools.repeat(x, group) for x in abc))
dataset=list()
query_set=list()
i=0
for y in full_data:
    z=float(lst[i])
    dataset.append([z]+y)
    i=i+1

for y in q_data:
    query_set.append([y[0]]+y)

with open('raheem_'+str(training), 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(dataset)
    
with open('raheem_'+str(training), 'a') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(query_set)
    
